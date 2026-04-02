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
# Model: Piecewise-aware MoE + residual multi-task heads
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden, dropout=0.05):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        h = self.ln1(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.ln2(h)
        return F.silu(x + h)


class Expert(nn.Module):
    def __init__(self, dim, hidden=96, dropout=0.03):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class TaskHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        trunk_dim = 96

        self.stem = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.03),
            nn.Linear(128, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.SiLU(),
        )

        self.block1 = ResidualBlock(trunk_dim, 128, dropout=0.03)
        self.block2 = ResidualBlock(trunk_dim, 128, dropout=0.03)

        # More experts to better separate boundary process regions
        self.num_experts = 6
        self.experts = nn.ModuleList([Expert(trunk_dim, hidden=96, dropout=0.02) for _ in range(self.num_experts)])

        self.gate = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, self.num_experts)
        )

        self.post_moe = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.SiLU(),
            nn.Dropout(0.03),
        )

        # Shared + task-specific fusion
        self.shared = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.SiLU(),
        )

        self.head_strain = TaskHead(64)
        self.head_tensile = TaskHead(64)
        self.head_yield = TaskHead(64)

        # Learn tensile first, then positive gap to satisfy tensile >= yield softly/hardly
        self.yield_gap_head = TaskHead(64)

    def forward(self, x, return_gate=False):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)

        gate_logits = self.gate(h)
        gate_w = torch.softmax(gate_logits, dim=-1)

        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=1)  # [B, E, D]
        moe_h = torch.sum(expert_outs * gate_w.unsqueeze(-1), dim=1)
        feat = self.post_moe(moe_h + h)

        shared_feat = self.shared(feat)

        strain = self.head_strain(shared_feat)
        tensile = self.head_tensile(shared_feat)

        # Harder physical structure: tensile >= yield via positive gap
        gap = F.softplus(self.yield_gap_head(shared_feat))
        yield_s = tensile - gap

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
        self.huber = nn.SmoothL1Loss(reduction='none', beta=1.5)

    def forward(self, pred, target, gate_w=None, epoch_ratio=1.0):
        err = pred - target
        abs_err = torch.abs(err)

        base_loss = self.huber(pred, target)  # [B, 3]

        # Task balancing using target variability
        with torch.no_grad():
            target_scale = target.std(dim=0, unbiased=False) + 1e-6
            task_weight = 1.0 / target_scale
            task_weight = task_weight / task_weight.mean()

            # Keep stronger focus on strength while not sacrificing strain too much
            task_bias = torch.tensor([0.95, 1.10, 1.15], device=pred.device)
            task_weight = task_weight * task_bias

            # Hard example weighting, stronger on boundary/nonlinear points
            sample_err = (abs_err * task_weight.view(1, 3)).mean(dim=1, keepdim=True)
            sample_scale = sample_err.mean() + 1e-6
            hard_weight = 1.0 + 0.9 * torch.tanh(sample_err / sample_scale)

            # Additional per-task hard weighting
            per_task_scale = abs_err.mean(dim=0, keepdim=True) + 1e-6
            task_hard = 1.0 + 0.35 * torch.tanh(abs_err / per_task_scale)

        weighted = base_loss * task_weight.view(1, 3) * task_hard * hard_weight
        data_loss = weighted.mean()

        # Physical soft constraint: tensile should not be lower than yield
        order_penalty = F.relu(pred[:, 2] - pred[:, 1]).mean()

        # Encourage expert diversity / avoid total collapse, but very gently
        load_balance = torch.tensor(0.0, device=pred.device)
        entropy_reg = torch.tensor(0.0, device=pred.device)
        if gate_w is not None:
            mean_gate = gate_w.mean(dim=0)
            uniform = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
            load_balance = ((mean_gate - uniform) ** 2).mean()
            entropy = -(gate_w * torch.log(gate_w + 1e-8)).sum(dim=1).mean()
            entropy_reg = -entropy  # maximize entropy mildly

        total_loss = (
            data_loss
            + 0.10 * order_penalty
            + 0.02 * epoch_ratio * load_balance
            + 0.005 * epoch_ratio * entropy_reg
        )
        return total_loss


# -----------------------------
# Training
# -----------------------------
lr = 0.001
batch_size = 16


def train_model(time_limit=300):
    set_seed(42)

    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    agent = FeatureAgent()
    X_train = agent.engineer_features(train_df)
    X_val = agent.engineer_features(val_df)

    y_train = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_train, X_val, feat_mean, feat_std = standardize_np(X_train, X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    model = Net(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=8e-5)
    criterion = RobustMultiTaskLoss()
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    start_time = time.time()
    epoch = 0
    best_state = None
    best_score = float('inf')

    last_improve_epoch = 0
    current_lr = lr

    # Two-stage training:
    # early stage: smoother fitting
    # later stage: emphasize hard examples / nonlinear regions
    while time.time() - start_time < time_limit - 5:
        model.train()
        n = len(X_train)

        # Dynamic weighted sampling to focus on currently hard samples
        if epoch == 0 or epoch % 20 != 0:
            perm = torch.randperm(n)
        else:
            model.eval()
            with torch.no_grad():
                pred_train = model(X_train)
                sample_mse = ((pred_train - y_train) ** 2).mean(dim=1)
                weights = (sample_mse / (sample_mse.mean() + 1e-8)).clamp(min=0.2)
                weights = weights / weights.sum()
                sampled_idx = torch.multinomial(weights, num_samples=n, replacement=True)
            model.train()
            perm = sampled_idx

        epoch_ratio = min(1.0, epoch / 60.0)

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            pred, gate_w = model(xb, return_gate=True)
            loss = criterion(pred, yb, gate_w=gate_w, epoch_ratio=epoch_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val)
                overall_mse = mse(pred_val, y_val).item()

                # Slight preference to stronger balance across tasks
                strain_mse_v = mse(pred_val[:, 0], y_val[:, 0]).item()
                tensile_mse_v = mse(pred_val[:, 1], y_val[:, 1]).item()
                yield_mse_v = mse(pred_val[:, 2], y_val[:, 2]).item()
                monitor_score = (
                    0.20 * strain_mse_v +
                    0.40 * tensile_mse_v +
                    0.40 * yield_mse_v
                )

            if monitor_score < best_score:
                best_score = monitor_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                last_improve_epoch = epoch
            elif epoch - last_improve_epoch >= 30:
                current_lr *= 0.65
                for g in optimizer.param_groups:
                    g['lr'] = max(current_lr, 8e-5)
                last_improve_epoch = epoch

        epoch += 1

    if best_state is not None:
        model.load_state_dict(best_state)

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