import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_agent import FeatureAgent


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_np(X_train, X_val):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


# -----------------------------
# Model blocks
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = self.dropout(h)
        return self.act(x + h)


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Model: stronger piecewise MoE + task-specific gating
# -----------------------------
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = 5

        self.stem = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(96, 64),
            nn.SiLU(),
        )

        self.trunk = nn.Sequential(
            ResidualBlock(64, dropout=0.04),
            ResidualBlock(64, dropout=0.04),
        )

        self.experts = nn.ModuleList([Expert(64) for _ in range(self.num_experts)])

        # shared gate
        self.gate_shared = nn.Sequential(
            nn.Linear(64, 48),
            nn.SiLU(),
            nn.Linear(48, self.num_experts)
        )
        # task-specific gates for discontinuous/process-sensitive regions
        self.gate_strain = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, self.num_experts)
        )
        self.gate_tensile = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, self.num_experts)
        )
        self.gate_yield = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, self.num_experts)
        )

        self.post_shared = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.04),
        )

        self.head_strain = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        self.head_tensile = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        self.head_yield = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def mix_experts(self, base_feat, gate_logits):
        gate_w = torch.softmax(gate_logits, dim=-1)
        expert_outs = torch.stack([expert(base_feat) for expert in self.experts], dim=1)  # [B, E, 64]
        feat = torch.sum(expert_outs * gate_w.unsqueeze(-1), dim=1)
        return feat, gate_w

    def forward(self, x, return_gate=False):
        h = self.stem(x)
        h = self.trunk(h)

        shared_logits = self.gate_shared(h)
        strain_logits = shared_logits + self.gate_strain(h)
        tensile_logits = shared_logits + self.gate_tensile(h)
        yield_logits = shared_logits + self.gate_yield(h)

        feat_strain, gate_w_strain = self.mix_experts(h, strain_logits)
        feat_tensile, gate_w_tensile = self.mix_experts(h, tensile_logits)
        feat_yield, gate_w_yield = self.mix_experts(h, yield_logits)

        feat_strain = self.post_shared(feat_strain)
        feat_tensile = self.post_shared(feat_tensile)
        feat_yield = self.post_shared(feat_yield)

        strain = self.head_strain(feat_strain)
        tensile = self.head_tensile(feat_tensile)
        yield_s = self.head_yield(feat_yield)

        out = torch.cat([strain, tensile, yield_s], dim=1)
        if return_gate:
            gates = {
                'strain': gate_w_strain,
                'tensile': gate_w_tensile,
                'yield': gate_w_yield,
            }
            return out, gates
        return out


# -----------------------------
# Loss
# -----------------------------
class RobustMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none', beta=1.0)

    def forward(self, pred, target, gates=None, epoch_ratio=1.0):
        err = pred - target
        abs_err = torch.abs(err)
        base_loss = self.huber(pred, target)  # [B, 3]

        # task scale normalization from target statistics
        with torch.no_grad():
            target_scale = target.std(dim=0, unbiased=False) + 1e-6
            task_weight = 1.0 / target_scale
            task_weight = task_weight / task_weight.mean()

        # prioritize strength while keeping strain stable
        task_bias = torch.tensor([0.95, 1.10, 1.20], device=pred.device)
        task_weight = task_weight * task_bias

        # hard-example reweighting:
        # emphasize local abrupt-process regions / outliers without fully exploding
        with torch.no_grad():
            scale = abs_err.mean(dim=0, keepdim=True) + 1e-6
            hard_ratio = abs_err / scale
            hard_weight = 1.0 + 0.45 * torch.tanh(hard_ratio) + 0.20 * torch.log1p(hard_ratio)

        data_loss = (base_loss * hard_weight * task_weight.view(1, 3)).mean()

        # physical soft constraint: tensile >= yield
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # soft covariance-style consistency:
        # tensile and yield should co-vary positively across samples
        pred_center = pred - pred.mean(dim=0, keepdim=True)
        cov_ty = (pred_center[:, 1] * pred_center[:, 2]).mean()
        cov_penalty = F.relu(-cov_ty)

        # keep MoE from collapsing too early
        gate_reg = torch.tensor(0.0, device=pred.device)
        if gates is not None:
            for gw in gates.values():
                entropy = -(gw * torch.log(gw + 1e-8)).sum(dim=1).mean()
                # encourage moderate entropy early, weaker later
                target_entropy = math.log(gw.shape[1]) * (0.75 - 0.35 * epoch_ratio)
                gate_reg = gate_reg + (entropy - target_entropy).pow(2)

        total_loss = (
            data_loss
            + 0.10 * order_penalty
            + 0.02 * cov_penalty
            + 0.01 * gate_reg
        )
        return total_loss


# -----------------------------
# Training
# -----------------------------
lr = 0.001
batch_size = 16


def train_model(time_limit=300):
    set_seed(42)

    # Load data
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    # Feature engineering
    agent = FeatureAgent()
    X_train = agent.engineer_features(train_df)
    X_val = agent.engineer_features(val_df)

    y_train = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    # Standardize features
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_train, X_val, feat_mean, feat_std = standardize_np(X_train, X_val)

    # Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Model / optimizer
    model = Net(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    criterion = RobustMultiTaskLoss()
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    start_time = time.time()
    epoch = 0
    best_state = None
    best_score = float('inf')
    last_improve_epoch = 0
    current_lr = lr

    n_train = len(X_train)
    if n_train <= 64:
        batch_size_eff = min(batch_size, max(4, n_train))
    else:
        batch_size_eff = batch_size

    while time.time() - start_time < time_limit - 5:
        model.train()

        # curriculum: later epochs oversample hard samples from previous residuals
        if epoch == 0 or not hasattr(train_model, "_sample_weights"):
            sample_weights = torch.ones(n_train, dtype=torch.float32)
        else:
            sample_weights = train_model._sample_weights.clone()

        sample_prob = sample_weights / sample_weights.sum()
        num_draw = n_train
        indices = torch.multinomial(sample_prob, num_samples=num_draw, replacement=True)

        epoch_ratio = min(1.0, epoch / 60.0)

        for i in range(0, len(indices), batch_size_eff):
            idx = indices[i:i + batch_size_eff]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            pred, gates = model(xb, return_gate=True)
            loss = criterion(pred, yb, gates=gates, epoch_ratio=epoch_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # refresh train residual weights occasionally
        if epoch % 8 == 0:
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train)
                train_abs_err = torch.abs(train_pred - y_train)

                # stronger focus on strength tasks, especially yield
                err_score = (
                    0.7 * train_abs_err[:, 0] +
                    1.1 * train_abs_err[:, 1] +
                    1.3 * train_abs_err[:, 2]
                )
                err_score = err_score / (err_score.mean() + 1e-6)
                sample_weights = 1.0 + 0.8 * torch.tanh(err_score)
                train_model._sample_weights = sample_weights.cpu()

        # validation and scheduler
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val)

                strain_mse_v = mse(pred_val[:, 0], y_val[:, 0]).item()
                tensile_mse_v = mse(pred_val[:, 1], y_val[:, 1]).item()
                yield_mse_v = mse(pred_val[:, 2], y_val[:, 2]).item()

                # validation score: slightly emphasize tensile/yield,
                # because current largest critical errors are strength-side
                overall_mse = mse(pred_val, y_val).item()
                score = 0.8 * strain_mse_v + 1.0 * tensile_mse_v + 1.15 * yield_mse_v

            if score < best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                last_improve_epoch = epoch
            elif epoch - last_improve_epoch >= 40:
                current_lr *= 0.65
                for g in optimizer.param_groups:
                    g['lr'] = max(current_lr, 1e-4)
                last_improve_epoch = epoch

        epoch += 1

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(X_val)

        overall_mse = mse(pred, y_val).item()
        strain_mse = mse(pred[:, 0], y_val[:, 0]).item()
        tensile_mse = mse(pred[:, 1], y_val[:, 1]).item()
        yield_mse = mse(pred[:, 2], y_val[:, 2]).item()

        strain_mae = mae(pred[:, 0], y_val[:, 0]).item()
        tensile_mae = mae(pred[:, 1], y_val[:, 1]).item()
        yield_mae = mae(pred[:, 2], y_val[:, 2]).item()

        strain_rel = (torch.abs(pred[:, 0] - y_val[:, 0]) / (y_val[:, 0].abs() + 1e-8)).mean().item()
        tensile_rel = (torch.abs(pred[:, 1] - y_val[:, 1]) / (y_val[:, 1].abs() + 1e-8)).mean().item()
        yield_rel = (torch.abs(pred[:, 2] - y_val[:, 2]) / (y_val[:, 2].abs() + 1e-8)).mean().item()

    print(f"Epochs: {epoch}")
    print(f"Val Loss: {overall_mse:.4f}")
    print(f"METRICS strain_mse={strain_mse:.4f} tensile_mse={tensile_mse:.4f} yield_mse={yield_mse:.4f}")
    print(f"METRICS strain_mae={strain_mae:.4f} tensile_mae={tensile_mae:.4f} yield_mae={yield_mae:.4f}")
    print(f"METRICS strain_rel={strain_rel:.4f} tensile_rel={tensile_rel:.4f} yield_rel={yield_rel:.4f}")

    # Per-sample validation residuals
    val_cols = set(val_df.columns.tolist())
    pred_np = pred.cpu().numpy()
    y_np = y_val.cpu().numpy()

    has_temp = 'temp' in val_cols
    has_time = 'time' in val_cols

    for i in range(len(val_df)):
        temp_v = float(val_df.iloc[i]['temp']) if has_temp else float('nan')
        time_v = float(val_df.iloc[i]['time']) if has_time else float('nan')

        strain_err = y_np[i, 0] - pred_np[i, 0]
        tensile_err = y_np[i, 1] - pred_np[i, 1]
        yield_err = y_np[i, 2] - pred_np[i, 2]

        print(
            f"VAL_PRED temp={temp_v} time={time_v} "
            f"strain_err={strain_err:.4f} tensile_err={tensile_err:.4f} yield_err={yield_err:.4f}"
        )

    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')