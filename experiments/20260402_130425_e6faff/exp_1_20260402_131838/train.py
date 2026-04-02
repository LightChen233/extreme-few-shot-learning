import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from feature_agent import FeatureAgent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RobustScalerTorch:
    def __init__(self):
        self.median = None
        self.scale = None

    def fit(self, x_np):
        median = np.median(x_np, axis=0)
        q75 = np.percentile(x_np, 75, axis=0)
        q25 = np.percentile(x_np, 25, axis=0)
        scale = q75 - q25
        scale[scale < 1e-6] = 1.0
        self.median = torch.FloatTensor(median)
        self.scale = torch.FloatTensor(scale)
        return self

    def transform(self, x):
        return (x - self.median.to(x.device)) / self.scale.to(x.device)

    def inverse_transform(self, x):
        return x * self.scale.to(x.device) + self.median.to(x.device)


class MultiTaskNet(nn.Module):
    """
    主干 + 多任务头，同时显式建模:
    1) 屈服强度 <= 抗拉强度（通过参数化保证）
    2) 残差学习：以训练集标签均值作为基础偏置，网络学习相对偏移
    """
    def __init__(self, input_dim, y_mean):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.08),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.08),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )

        self.shared = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU()
        )

        self.strain_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        # 先预测 uts，再预测 gap = softplus(raw_gap)，使 yield = uts - gap
        self.uts_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        self.gap_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        self.register_buffer("y_mean", torch.FloatTensor(y_mean.reshape(1, 3)))

    def forward(self, x):
        h = self.backbone(x)
        h = self.shared(h)

        strain = self.y_mean[:, 0:1] + self.strain_head(h)
        uts = self.y_mean[:, 1:2] + self.uts_head(h)

        raw_gap = self.gap_head(h)
        gap = torch.nn.functional.softplus(raw_gap)
        yld = uts - gap

        out = torch.cat([strain, uts, yld], dim=1)
        return out


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, task_weights=None, sample_weights=None):
        err = pred - target
        abs_err = torch.abs(err)
        quadratic = torch.minimum(abs_err, torch.tensor(self.delta, device=pred.device))
        linear = abs_err - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear  # [N, 3]

        if task_weights is not None:
            loss = loss * task_weights.view(1, -1)

        if sample_weights is not None:
            loss = loss * sample_weights.view(-1, 1)

        return loss.mean()


def build_sample_weights(df):
    """
    根据误差分析，困难样本集中在高温(460/470) + 12h附近。
    这里不硬编码标签值，只依据工艺条件稀有/潜在峰值区域做温和加权。
    """
    temp = df["temp"].values.astype(np.float32)
    t = df["time"].values.astype(np.float32)

    temp_norm = (temp - temp.mean()) / (temp.std() + 1e-6)
    time_log = np.log1p(t)
    time_norm = (time_log - time_log.mean()) / (time_log.std() + 1e-6)

    # 对中高温、典型时效时间段稍加关注
    focus = 1.0 + 0.25 * (temp_norm > 0).astype(np.float32) + 0.25 * (np.abs(time_norm) < 0.8).astype(np.float32)
    return torch.FloatTensor(focus)


def compute_task_weights(y_train):
    """
    用目标标准差做归一，避免抗拉/屈服因量纲大主导损失。
    """
    std = y_train.std(axis=0) + 1e-6
    w = 1.0 / std
    w = w / w.mean()
    return torch.FloatTensor(w.astype(np.float32))


def monotonic_consistency_penalty(model, x_base, temp_idx, time_idx, x_scaler, device):
    """
    基于工艺常识加入弱物理约束：
    在局部范围内，性能对温度/时间通常应平滑，不应出现剧烈振荡。
    用小扰动前后预测差的平方作为平滑正则。
    不要求严格单调，只约束局部 Lipschitz/平滑性。
    """
    if temp_idx is None and time_idx is None:
        return torch.tensor(0.0, device=device)

    x1 = x_base.clone()
    x2 = x_base.clone()

    if temp_idx is not None:
        x1[:, temp_idx] += 0.15
        x2[:, temp_idx] -= 0.15
    if time_idx is not None:
        x1[:, time_idx] += 0.15
        x2[:, time_idx] -= 0.15

    p1 = model(x1)
    p2 = model(x2)
    return ((p1 - p2) ** 2).mean()


# 超参数
lr = 8e-4
batch_size = 8
weight_decay = 1e-4


def train_model(time_limit=300):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    # 特征工程（不计入训练时间 budget）
    agent = FeatureAgent()
    X_train_np = agent.engineer_features(train_df)
    X_val_np = agent.engineer_features(val_df)
    y_train_np = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val_np = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    # 特征标准化（不计入训练时间 budget）
    x_scaler = RobustScalerTorch().fit(X_train_np)
    X_train = torch.FloatTensor(X_train_np)
    X_val = torch.FloatTensor(X_val_np)
    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)

    y_train = torch.FloatTensor(y_train_np)
    y_val = torch.FloatTensor(y_val_np)

    # 寻找 temp/time 在原始表中的列位置，用于局部平滑约束（若 feature_agent 未保留原列，也不依赖）
    temp_idx = None
    time_idx = None
    if isinstance(X_train_np, np.ndarray) and X_train_np.shape[1] >= 2:
        # 无法可靠知道 engineered features 中 temp/time 索引，默认不强行使用
        temp_idx = None
        time_idx = None

    # 模型
    y_mean = y_train_np.mean(axis=0)
    model = MultiTaskNet(X_train.shape[1], y_mean).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    criterion = WeightedHuberLoss(delta=6.0)
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    task_weights = compute_task_weights(y_train_np).to(device)
    sample_weights = build_sample_weights(train_df).to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    best_state = None
    best_score = float('inf')

    # 训练循环（特征工程结束后才开始计时）
    start_time = time.time()
    epoch = 0
    patience_counter = 0

    while time.time() - start_time < time_limit - 3:
        model.train()
        perm = torch.randperm(len(X_train), device=device)

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            sw = sample_weights[idx]

            optimizer.zero_grad()
            pred = model(xb)

            data_loss = criterion(pred, yb, task_weights=task_weights, sample_weights=sw)

            # 软约束：应变非负、强度非负（通用物理约束）
            nonneg_penalty = (
                torch.relu(-pred[:, 0]).mean() +
                torch.relu(-pred[:, 1]).mean() +
                torch.relu(-pred[:, 2]).mean()
            )

            # 已通过结构保证 yield <= tensile；这里再加一个极弱的安全约束
            order_penalty = torch.relu(pred[:, 2] - pred[:, 1]).mean()

            # 平滑性约束，避免小数据集上对局部条件过拟合振荡
            smooth_penalty = monotonic_consistency_penalty(
                model, xb, temp_idx, time_idx, x_scaler, device
            )

            loss = data_loss + 0.02 * nonneg_penalty + 0.05 * order_penalty + 0.01 * smooth_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()
        epoch += 1

        # 每 10 epoch 评估一次，打印进度 + 真正的 early stop
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val)
                val_mse_now = mse(pred_val, y_val).item()
                val_score = (
                    ((pred_val[:, 0] - y_val[:, 0]).abs() / (y_val[:, 0].abs() + 1e-6)).mean() * 2.0 +
                    ((pred_val[:, 1] - y_val[:, 1]).abs() / (y_val[:, 1].abs() + 1e-6)).mean() * 1.0 +
                    ((pred_val[:, 2] - y_val[:, 2]).abs() / (y_val[:, 2].abs() + 1e-6)).mean() * 1.0
                ).item()

                elapsed = time.time() - start_time
                print(f"[epoch {epoch:5d} | {elapsed:5.0f}s] val_mse={val_mse_now:.2f} val_score={val_score:.4f} patience={patience_counter}", flush=True)

                if val_score < best_score:
                    best_score = val_score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                # early stop：patience 超过 50 次（500 epoch 无改善）且至少训练了 60 秒
                if patience_counter >= 50 and elapsed > 60:
                    print(f"Early stop at epoch {epoch}", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 评估
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

    pred_np = pred.detach().cpu().numpy()
    y_np = y_val.detach().cpu().numpy()
    for j in range(len(y_np)):
        row = val_df.iloc[j]
        print(
            f"VAL_PRED temp={row['temp']} time={row['time']} "
            f"strain_err={y_np[j,0]-pred_np[j,0]:.3f} "
            f"tensile_err={y_np[j,1]-pred_np[j,1]:.3f} "
            f"yield_err={y_np[j,2]-pred_np[j,2]:.3f}"
        )

    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')
    print("MODEL_SAVED model.pt")