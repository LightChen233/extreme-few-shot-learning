import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_agent import FeatureAgent


# -----------------------------
# Utils
# -----------------------------
class StandardScalerTorch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def extract_temp_time_features(df):
    temp = df['temp'].values.astype(np.float32).reshape(-1, 1) if 'temp' in df.columns else np.zeros((len(df), 1), dtype=np.float32)
    ttime = df['time'].values.astype(np.float32).reshape(-1, 1) if 'time' in df.columns else np.zeros((len(df), 1), dtype=np.float32)

    log_time = np.log1p(np.maximum(ttime, 0.0)).astype(np.float32)
    temp_time = (temp * ttime).astype(np.float32)
    temp_logt = (temp * log_time).astype(np.float32)

    return np.concatenate([temp, ttime, log_time, temp_time, temp_logt], axis=1)


def build_features(train_df, val_df):
    agent = FeatureAgent()
    X_train_main = np.asarray(agent.engineer_features(train_df), dtype=np.float32)
    X_val_main = np.asarray(agent.engineer_features(val_df), dtype=np.float32)

    X_train_aux = extract_temp_time_features(train_df)
    X_val_aux = extract_temp_time_features(val_df)

    X_train = np.concatenate([X_train_main, X_train_aux], axis=1).astype(np.float32)
    X_val = np.concatenate([X_val_main, X_val_aux], axis=1).astype(np.float32)

    x_scaler = StandardScalerTorch().fit(X_train)
    X_train = x_scaler.transform(X_train).astype(np.float32)
    X_val = x_scaler.transform(X_val).astype(np.float32)

    y_train = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    y_scaler = StandardScalerTorch().fit(y_train)
    y_train_s = y_scaler.transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    return X_train, y_train, X_val, y_val, y_train_s, y_val_s, y_scaler


# -----------------------------
# Model
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.08):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        h = F.silu(self.norm1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return F.silu(x + self.dropout(h))


class Expert(nn.Module):
    def __init__(self, hidden_dim, out_dim=3, dropout=0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class TaskHead(nn.Module):
    def __init__(self, hidden_dim, out_dim=1, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 128
        n_experts = 5

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU()
        )

        self.experts = nn.ModuleList([
            Expert(hidden, out_dim=3, dropout=0.06) for _ in range(n_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, n_experts)
        )

        self.shared_head = nn.Sequential(
            ResidualBlock(hidden, dropout=0.05),
            nn.Linear(hidden, 3)
        )

        self.task_heads = nn.ModuleList([
            TaskHead(hidden, out_dim=1, dropout=0.04),
            TaskHead(hidden, out_dim=1, dropout=0.04),
            TaskHead(hidden, out_dim=1, dropout=0.04),
        ])

        self.logit_temp = nn.Parameter(torch.tensor(1.2))

    def forward(self, x):
        h = self.stem(x)

        gate_logits = self.gate(x)
        temp = torch.clamp(torch.exp(self.logit_temp), min=0.5, max=3.0)
        gate_w = torch.softmax(gate_logits / temp, dim=-1)

        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=1)  # [B, E, 3]
        moe_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)

        shared_out = self.shared_head(h)
        task_out = torch.cat([head(h) for head in self.task_heads], dim=1)

        out = 0.60 * moe_out + 0.20 * shared_out + 0.20 * task_out
        return out, gate_w


# -----------------------------
# Hyperparameters
# -----------------------------
lr = 8e-4
batch_size = 16
weight_decay = 8e-5


# -----------------------------
# Loss
# -----------------------------
def pairwise_rank_loss(pred_col, true_col):
    n = pred_col.shape[0]
    if n <= 1:
        return pred_col.new_tensor(0.0)

    diff_true = true_col.unsqueeze(1) - true_col.unsqueeze(0)
    diff_pred = pred_col.unsqueeze(1) - pred_col.unsqueeze(0)

    mask = (torch.abs(diff_true) > 0.10 * (torch.std(true_col) + 1e-6)).float()
    sign_true = torch.sign(diff_true)
    margin_term = F.relu(0.05 - sign_true * diff_pred)
    loss = (margin_term * mask).sum() / (mask.sum() + 1e-6)
    return loss


def multi_task_loss(pred_s, target_s, gate_w, epoch_ratio=0.0):
    err = pred_s - target_s
    abs_err = torch.abs(err).detach()

    task_weights = torch.tensor([1.15, 1.0, 1.0], device=pred_s.device).view(1, 3)

    focal_strength = 0.35 + 1.65 * epoch_ratio
    hard_weights = (1.0 + abs_err) ** focal_strength
    weighted_mse = (task_weights * hard_weights * err.pow(2)).mean()

    huber = F.smooth_l1_loss(pred_s, target_s, beta=0.8)

    # physical constraint in standardized space: tensile >= yield
    violation = F.relu(pred_s[:, 2] - pred_s[:, 1])
    phys_rank = (violation ** 2).mean()

    # pairwise ordering consistency for tensile/yield trends inside batch
    rank_tensile = pairwise_rank_loss(pred_s[:, 1], target_s[:, 1])
    rank_yield = pairwise_rank_loss(pred_s[:, 2], target_s[:, 2])

    # gate regularization: avoid collapse, but not too strong
    avg_gate = gate_w.mean(dim=0)
    entropy = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
    gate_balance = -entropy

    # sample-wise sparse routing, stronger later
    sample_entropy = -(gate_w * torch.log(gate_w + 1e-8)).sum(dim=1).mean()

    loss = (
        weighted_mse
        + 0.22 * huber
        + 0.10 * phys_rank
        + 0.05 * rank_tensile
        + 0.05 * rank_yield
        + 0.006 * gate_balance
        + (0.002 + 0.01 * epoch_ratio) * sample_entropy
    )
    return loss


# -----------------------------
# Eval
# -----------------------------
def evaluate_and_print(model, X_val_t, y_val_t_raw, y_scaler, val_df):
    model.eval()
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    with torch.no_grad():
        pred_s, gate_w = model(X_val_t)
        pred = torch.from_numpy(y_scaler.inverse_transform(pred_s.cpu().numpy())).float()

        overall_mse = mse_fn(pred, y_val_t_raw).item()
        strain_mse = mse_fn(pred[:, 0], y_val_t_raw[:, 0]).item()
        tensile_mse = mse_fn(pred[:, 1], y_val_t_raw[:, 1]).item()
        yield_mse = mse_fn(pred[:, 2], y_val_t_raw[:, 2]).item()

        strain_mae = mae_fn(pred[:, 0], y_val_t_raw[:, 0]).item()
        tensile_mae = mae_fn(pred[:, 1], y_val_t_raw[:, 1]).item()
        yield_mae = mae_fn(pred[:, 2], y_val_t_raw[:, 2]).item()

        strain_rel = (torch.abs(pred[:, 0] - y_val_t_raw[:, 0]) / (y_val_t_raw[:, 0].abs() + 1e-8)).mean().item()
        tensile_rel = (torch.abs(pred[:, 1] - y_val_t_raw[:, 1]) / (y_val_t_raw[:, 1].abs() + 1e-8)).mean().item()
        yield_rel = (torch.abs(pred[:, 2] - y_val_t_raw[:, 2]) / (y_val_t_raw[:, 2].abs() + 1e-8)).mean().item()

        print(f"Val Loss: {overall_mse:.4f}")
        print(f"METRICS strain_mse={strain_mse:.4f} tensile_mse={tensile_mse:.4f} yield_mse={yield_mse:.4f}")
        print(f"METRICS strain_mae={strain_mae:.4f} tensile_mae={tensile_mae:.4f} yield_mae={yield_mae:.4f}")
        print(f"METRICS strain_rel={strain_rel:.4f} tensile_rel={tensile_rel:.4f} yield_rel={yield_rel:.4f}")

        pred_np = pred.cpu().numpy()
        y_np = y_val_t_raw.cpu().numpy()

        for i in range(len(val_df)):
            temp = val_df.iloc[i]['temp'] if 'temp' in val_df.columns else np.nan
            ttime = val_df.iloc[i]['time'] if 'time' in val_df.columns else np.nan
            strain_err = y_np[i, 0] - pred_np[i, 0]
            tensile_err = y_np[i, 1] - pred_np[i, 1]
            yield_err = y_np[i, 2] - pred_np[i, 2]
            print(
                f"VAL_PRED temp={temp} time={ttime} "
                f"strain_err={strain_err:.4f} tensile_err={tensile_err:.4f} yield_err={yield_err:.4f}"
            )

    return overall_mse


# -----------------------------
# Train
# -----------------------------
def train_model(time_limit=300):
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    X_train, y_train_raw, X_val, y_val_raw, y_train_s, y_val_s, y_scaler = build_features(train_df, val_df)

    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    y_train_t_raw = torch.FloatTensor(y_train_raw)
    y_val_t_raw = torch.FloatTensor(y_val_raw)
    y_train_t_s = torch.FloatTensor(y_train_s)
    y_val_t_s = torch.FloatTensor(y_val_s)

    model = Net(X_train.shape[1])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=400,
        pct_start=0.18,
        anneal_strategy='cos',
        div_factor=8.0,
        final_div_factor=50.0
    )

    best_state = None
    best_metric = float('inf')

    start_time = time.time()
    epoch = 0
    n = len(X_train_t)
    steps_done = 0

    while time.time() - start_time < time_limit - 4:
        model.train()
        perm = torch.randperm(n)
        epoch_ratio = min(1.0, epoch / 180.0)

        for i in range(0, n, batch_size):
            if time.time() - start_time >= time_limit - 4:
                break

            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb_s = y_train_t_s[idx]

            optimizer.zero_grad()
            pred_s, gate_w = model(xb)
            loss = multi_task_loss(pred_s, yb_s, gate_w, epoch_ratio=epoch_ratio)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if steps_done < 400:
                scheduler.step()
            steps_done += 1

        epoch += 1

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_val_s, gate_w_val = model(X_val_t)
                pred_val = torch.from_numpy(y_scaler.inverse_transform(pred_val_s.cpu().numpy())).float()

                mse_all = nn.MSELoss()(pred_val, y_val_t_raw).item()

                # emphasize the previously problematic tensile/yield underestimation zones
                tensile_mse = nn.MSELoss()(pred_val[:, 1], y_val_t_raw[:, 1]).item()
                yield_mse = nn.MSELoss()(pred_val[:, 2], y_val_t_raw[:, 2]).item()
                val_metric = 0.55 * mse_all + 0.25 * tensile_mse + 0.20 * yield_mse

                if val_metric < best_metric:
                    best_metric = val_metric
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Epochs: {epoch}")
    overall_mse = evaluate_and_print(model, X_val_t, y_val_t_raw, y_scaler, val_df)
    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')