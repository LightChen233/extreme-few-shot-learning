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
# Model: shared trunk + MoE + multi-task heads
# -----------------------------
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.stem = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.SiLU(),
        )

        self.num_experts = 4
        self.experts = nn.ModuleList([Expert(64) for _ in range(self.num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, self.num_experts)
        )

        self.shared = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
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

    def forward(self, x, return_gate=False):
        h = self.stem(x)
        gate_logits = self.gate(h)
        gate_w = torch.softmax(gate_logits, dim=-1)

        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=1)  # [B, E, H]
        moe_h = torch.sum(expert_outs * gate_w.unsqueeze(-1), dim=1)  # [B, H]

        feat = self.shared(moe_h)

        strain = self.head_strain(feat)
        tensile = self.head_tensile(feat)
        yield_s = self.head_yield(feat)

        out = torch.cat([strain, tensile, yield_s], dim=1)
        if return_gate:
            return out, gate_w
        return out


# -----------------------------
# Loss
# -----------------------------
class RobustMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none', beta=1.0)

    def forward(self, pred, target, epoch_ratio=1.0):
        err = pred - target
        base_loss = self.huber(pred, target)  # [B, 3]

        with torch.no_grad():
            abs_err = torch.abs(err)
            scale = abs_err.mean(dim=0, keepdim=True) + 1e-6
            hard_weight = 1.0 + 0.5 * torch.tanh(abs_err / scale)

        # Dynamic task weighting by target scale + current difficulty
        target_scale = target.std(dim=0, unbiased=False) + 1e-6
        task_weight = 1.0 / target_scale
        task_weight = task_weight / task_weight.mean()

        # Put slightly more emphasis on strength tasks
        task_bias = torch.tensor([0.9, 1.05, 1.10], device=pred.device)
        task_weight = task_weight * task_bias

        weighted = base_loss * hard_weight * task_weight.view(1, 3)
        data_loss = weighted.mean()

        # Physical soft constraint: tensile strength should not be lower than yield strength
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # Gentle gate entropy regularization to avoid early collapse
        total_loss = data_loss + 0.08 * epoch_ratio * order_penalty
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = RobustMultiTaskLoss()
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    start_time = time.time()
    epoch = 0
    best_state = None
    best_score = float('inf')

    # Initial eval for scheduler-like manual control
    last_improve_epoch = 0
    current_lr = lr

    while time.time() - start_time < time_limit - 5:
        model.train()
        perm = torch.randperm(len(X_train))

        epoch_ratio = min(1.0, epoch / 50.0)

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb, epoch_ratio=epoch_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Periodic validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val)
                overall_mse = mse(pred_val, y_val).item()

            if overall_mse < best_score:
                best_score = overall_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                last_improve_epoch = epoch
            elif epoch - last_improve_epoch >= 40:
                current_lr *= 0.6
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