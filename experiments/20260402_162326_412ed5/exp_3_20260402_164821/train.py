import time
import copy
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
# Model
# 分层 MoE：共享干路 + 路由 + 任务特异专家
# 针对 460/470℃, 12h 等潜在突变区增强分段拟合能力
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
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
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.03),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


class TaskHeadMoE(nn.Module):
    def __init__(self, feat_dim, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(feat_dim, feat_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, num_experts)
        )
        self.out = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, temp_signal=None):
        logits = self.gate(x)
        if temp_signal is not None:
            logits = logits + temp_signal
        w = torch.softmax(logits, dim=-1)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, H]
        feat = torch.sum(expert_outs * w.unsqueeze(-1), dim=1)
        y = self.out(feat)
        return y, w


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.stem = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
        )

        self.block1 = ResidualBlock(96, dropout=0.04)
        self.block2 = ResidualBlock(96, dropout=0.04)

        # 全局路由：捕获工艺分段
        self.num_global_experts = 4
        self.global_gate = nn.Sequential(
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, self.num_global_experts)
        )
        self.global_experts = nn.ModuleList([Expert(96, 96) for _ in range(self.num_global_experts)])

        self.shared_post = nn.Sequential(
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.03),
        )

        # 任务头：每个任务再做一次轻量 MoE，提升突变区域拟合能力
        self.head_strain = TaskHeadMoE(64, num_experts=3)
        self.head_tensile = TaskHeadMoE(64, num_experts=3)
        self.head_yield = TaskHeadMoE(64, num_experts=3)

        # 可学习基准偏置，帮助模型围绕原始样品水平学习残差
        self.base = nn.Parameter(torch.tensor([6.94, 145.83, 96.60], dtype=torch.float32))

    def forward(self, x, return_gate=False):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)

        # 全局 MoE
        global_logits = self.global_gate(h)
        global_w = torch.softmax(global_logits, dim=-1)
        global_expert_outs = torch.stack([e(h) for e in self.global_experts], dim=1)  # [B, E, H]
        h = torch.sum(global_expert_outs * global_w.unsqueeze(-1), dim=1)

        feat = self.shared_post(h)

        # 用全局 gate 前3维作为任务头的温时状态先验信号（不硬编码阈值）
        temp_signal = global_logits[:, :3]

        strain, gate_s = self.head_strain(feat, temp_signal=temp_signal)
        tensile, gate_t = self.head_tensile(feat, temp_signal=temp_signal)
        yield_s, gate_y = self.head_yield(feat, temp_signal=temp_signal)

        out = torch.cat([strain, tensile, yield_s], dim=1) + self.base.view(1, 3)

        if return_gate:
            gate_info = {
                'global_w': global_w,
                'strain_w': gate_s,
                'tensile_w': gate_t,
                'yield_w': gate_y,
            }
            return out, gate_info
        return out


# -----------------------------
# Loss
# 针对大误差样本做动态聚焦
# 加入通用物理约束：
# 1) tensile >= yield
# 2) strain > 0, strengths > 0
# -----------------------------
class RobustMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none', beta=1.0)

    def forward(self, pred, target, gate_info=None, epoch_ratio=1.0):
        err = pred - target
        abs_err = torch.abs(err)
        base = self.huber(pred, target)  # [B, 3]

        # 任务尺度归一，避免强度项主导或应变被忽略
        target_scale = target.std(dim=0, unbiased=False) + 1e-6
        task_weight = 1.0 / target_scale
        task_weight = task_weight / task_weight.mean()

        # 进一步提升应变与屈服权重，针对当前短板
        task_bias = torch.tensor([1.18, 1.00, 1.12], device=pred.device)
        task_weight = task_weight * task_bias

        # 样本级 hard-example 动态加权
        with torch.no_grad():
            scale = abs_err.mean(dim=0, keepdim=True) + 1e-6
            focal_like = (abs_err / scale).clamp(min=0.0)
            hard_weight = 1.0 + 0.35 * torch.tanh(focal_like) + 0.20 * torch.sqrt(focal_like + 1e-8)

        data_loss = (base * hard_weight * task_weight.view(1, 3)).mean()

        # 物理约束：抗拉 >= 屈服
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 正值约束（通用）
        positivity_penalty = (
            F.relu(-pred[:, 0]).mean() +
            F.relu(-pred[:, 1]).mean() +
            F.relu(-pred[:, 2]).mean()
        )

        # Gate 均衡正则：防止塌缩到单专家，提高局部分段稳健性
        balance_penalty = 0.0
        entropy_bonus = 0.0
        if gate_info is not None:
            for name in ['global_w', 'strain_w', 'tensile_w', 'yield_w']:
                w = gate_info[name]
                mean_w = w.mean(dim=0)
                uniform = torch.full_like(mean_w, 1.0 / mean_w.numel())
                balance_penalty = balance_penalty + F.mse_loss(mean_w, uniform)

                entropy = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()
                entropy_bonus = entropy_bonus + entropy

        total_loss = (
            data_loss
            + 0.10 * epoch_ratio * order_penalty
            + 0.02 * positivity_penalty
            + 0.02 * epoch_ratio * balance_penalty
            - 0.003 * epoch_ratio * entropy_bonus
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

    # learning rate control
    current_lr = lr
    last_improve_epoch = 0

    while time.time() - start_time < time_limit - 5:
        model.train()

        # 每轮根据当前残差构建采样权重，让高误差样本被更多看到
        if epoch == 0:
            sample_weights = torch.ones(len(X_train), dtype=torch.float32)
        else:
            model.eval()
            with torch.no_grad():
                pred_train = model(X_train)
                train_abs_err = torch.abs(y_train - pred_train)
                # 更关注屈服和应变，兼顾抗拉
                sample_score = (
                    1.10 * train_abs_err[:, 0] / (y_train[:, 0].std() + 1e-6) +
                    0.95 * train_abs_err[:, 1] / (y_train[:, 1].std() + 1e-6) +
                    1.15 * train_abs_err[:, 2] / (y_train[:, 2].std() + 1e-6)
                )
                sample_weights = (sample_score + 1.0).cpu()
            model.train()

        # Weighted sampling with replacement for hard examples
        num_batches = max(1, int(np.ceil(len(X_train) / batch_size)))
        sample_idx = torch.multinomial(sample_weights, num_samples=num_batches * batch_size, replacement=True)

        epoch_ratio = min(1.0, epoch / 60.0)

        for i in range(0, len(sample_idx), batch_size):
            idx = sample_idx[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            pred, gate_info = model(xb, return_gate=True)
            loss = criterion(pred, yb, gate_info=gate_info, epoch_ratio=epoch_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Validation
        if epoch % 8 == 0:
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val)
                overall_mse = mse(pred_val, y_val).item()

                # 用更贴合目标的综合评分选最佳：
                # 在 overall_mse 基础上，稍偏向 yield/strain，避免平均掩盖局部恶化
                strain_mse = mse(pred_val[:, 0], y_val[:, 0]).item()
                tensile_mse = mse(pred_val[:, 1], y_val[:, 1]).item()
                yield_mse = mse(pred_val[:, 2], y_val[:, 2]).item()
                score = 0.20 * strain_mse + 0.30 * tensile_mse + 0.50 * yield_mse

            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                last_improve_epoch = epoch
            elif epoch - last_improve_epoch >= 32:
                current_lr *= 0.65
                for g in optimizer.param_groups:
                    g['lr'] = max(current_lr, 8e-5)
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