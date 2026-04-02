import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_agent import FeatureAgent


# -----------------------------
# Model
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
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
    def __init__(self, hidden_dim, out_dim=3, dropout=0.1):
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


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 128
        n_experts = 4

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU()
        )

        self.experts = nn.ModuleList([
            Expert(hidden, out_dim=3, dropout=0.08) for _ in range(n_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, n_experts)
        )

        # shared correction head
        self.shared_head = nn.Sequential(
            ResidualBlock(hidden, dropout=0.05),
            nn.Linear(hidden, 3)
        )

    def forward(self, x):
        h = self.stem(x)
        gate_logits = self.gate(x)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [B, E]

        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=1)  # [B, E, 3]
        moe_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)  # [B, 3]
        shared_out = self.shared_head(h)

        out = moe_out + 0.3 * shared_out
        return out, gate_w


# -----------------------------
# Hyperparameters
# -----------------------------
lr = 0.001
batch_size = 16
weight_decay = 1e-4


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


def build_features(train_df, val_df):
    agent = FeatureAgent()
    X_train = agent.engineer_features(train_df)
    X_val = agent.engineer_features(val_df)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)

    x_scaler = StandardScalerTorch().fit(X_train)
    X_train = x_scaler.transform(X_train).astype(np.float32)
    X_val = x_scaler.transform(X_val).astype(np.float32)

    y_train = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    y_scaler = StandardScalerTorch().fit(y_train)
    y_train_s = y_scaler.transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    return X_train, y_train, X_val, y_val, y_train_s, y_val_s, y_scaler


def multi_task_loss(pred_s, target_s, target_raw, gate_w, epoch_ratio=0.0):
    # base weighted MSE by inverse target variance in standardized space -> equalized already
    err = pred_s - target_s
    abs_err = torch.abs(err).detach()

    # Hard-example reweighting, stronger later in training
    focal_strength = 0.5 + 1.5 * epoch_ratio
    weights = (1.0 + abs_err) ** focal_strength
    mse = (weights * err.pow(2)).mean()

    # robust auxiliary MAE
    mae = F.l1_loss(pred_s, target_s)

    # physical constraint: tensile strength should generally be >= yield strength
    # target index: 1 tensile, 2 yield
    pred_raw = pred_s
    violation = F.relu(pred_raw[:, 2] - pred_raw[:, 1])
    phys_rank = (violation ** 2).mean()

    # encourage expert diversity but avoid collapse
    avg_gate = gate_w.mean(dim=0)
    entropy = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
    gate_reg = -entropy

    return mse + 0.25 * mae + 0.08 * phys_rank + 0.01 * gate_reg


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

        # required per-sample print
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    best_state = None
    best_metric = float('inf')

    start_time = time.time()
    epoch = 0

    n = len(X_train_t)
    while time.time() - start_time < time_limit - 3:
        model.train()
        perm = torch.randperm(n)

        epoch_ratio = min(1.0, epoch / 200.0)

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb_s = y_train_t_s[idx]
            yb_raw = y_train_t_raw[idx]

            optimizer.zero_grad()
            pred_s, gate_w = model(xb)
            loss = multi_task_loss(pred_s, yb_s, yb_raw, gate_w, epoch_ratio=epoch_ratio)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()
        epoch += 1

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_val_s, gate_w_val = model(X_val_t)
                pred_val = torch.from_numpy(y_scaler.inverse_transform(pred_val_s.cpu().numpy())).float()
                val_metric = nn.MSELoss()(pred_val, y_val_t_raw).item()
                if val_metric < best_metric:
                    best_metric = val_metric
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Epochs: {epoch}")
    overall_mse = evaluate_and_print(model, X_val_t, y_val_t_raw, y_scaler, val_df)
    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')