import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_agent import FeatureAgent


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class StandardScalerNP:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class TargetScalerNP:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float32)
        self.mean = y.mean(axis=0, keepdims=True)
        self.std = y.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, y):
        y = np.asarray(y, dtype=np.float32)
        return (y - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        y_scaled = np.asarray(y_scaled, dtype=np.float32)
        return y_scaled * self.std + self.mean


class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 3)
        )

    def forward(self, x):
        return self.net(x)


# 主模型定义 - 必须命名为 Net
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 混合专家：适合处理高温/长时下可能存在的突变、过时效、相变区域
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

        self.expert1 = MLPExpert(128, hidden1=128, hidden2=64, dropout=0.08)
        self.expert2 = MLPExpert(128, hidden1=128, hidden2=64, dropout=0.08)
        self.expert3 = MLPExpert(128, hidden1=128, hidden2=64, dropout=0.08)

        # 额外残差头，增强细节拟合
        self.residual_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        h = self.shared(x)
        gate_logits = self.gate(h)
        gate_w = torch.softmax(gate_logits, dim=-1)

        e1 = self.expert1(h)
        e2 = self.expert2(h)
        e3 = self.expert3(h)

        experts = torch.stack([e1, e2, e3], dim=1)  # [B, 3, 3]
        moe_out = (gate_w.unsqueeze(-1) * experts).sum(dim=1)
        out = moe_out + 0.3 * self.residual_head(h)
        return out


def infer_temp_time_columns(df):
    temp_candidates = ['temp', 'temperature', 'Temperature', 'TEMP']
    time_candidates = ['time', 'holding_time', 'Time', 'TIME']

    temp_col = None
    time_col = None

    for c in temp_candidates:
        if c in df.columns:
            temp_col = c
            break

    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break

    # 回退：模糊匹配
    if temp_col is None:
        for c in df.columns:
            cl = c.lower()
            if 'temp' in cl:
                temp_col = c
                break
    if time_col is None:
        for c in df.columns:
            cl = c.lower()
            if 'time' in cl:
                time_col = c
                break

    return temp_col, time_col


def compute_sample_weights(df, y):
    n = len(df)
    weights = np.ones(n, dtype=np.float32)

    temp_col, time_col = infer_temp_time_columns(df)

    if temp_col is not None:
        temp = pd.to_numeric(df[temp_col], errors='coerce').fillna(df[temp_col].median()).values.astype(np.float32)
        t_norm = (temp - temp.min()) / (temp.max() - temp.min() + 1e-6)
        weights *= (1.0 + 0.35 * t_norm)

    if time_col is not None:
        hold = pd.to_numeric(df[time_col], errors='coerce').fillna(df[time_col].median()).values.astype(np.float32)
        h_norm = (hold - hold.min()) / (hold.max() - hold.min() + 1e-6)
        weights *= (1.0 + 0.35 * h_norm)

    # 对目标空间边缘样本适度增权，帮助学习突变区域
    y = np.asarray(y, dtype=np.float32)
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True) + 1e-6
    z = np.abs((y - y_mean) / y_std).mean(axis=1)
    weights *= (1.0 + 0.25 * np.clip(z, 0, 3))

    weights = weights / (weights.mean() + 1e-8)
    return weights.astype(np.float32)


def physics_constraint_loss(pred_scaled, y_scaled_ref=None):
    """
    通用物理约束：
    对大多数金属材料，抗拉强度通常不低于屈服强度。
    这里在标准化空间中加入软约束，避免不合理预测。
    """
    tensile_pred = pred_scaled[:, 1]
    yield_pred = pred_scaled[:, 2]
    order_penalty = F.relu(yield_pred - tensile_pred).pow(2).mean()

    # 轻微范围平滑正则，防止输出发散
    reg = pred_scaled.pow(2).mean()
    return order_penalty + 0.001 * reg


def weighted_huber_loss(pred, target, sample_weight, delta=1.0, task_weight=None):
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear

    if task_weight is not None:
        loss = loss * task_weight.view(1, -1)

    loss = loss.mean(dim=1) * sample_weight
    return loss.mean()


def train_model(time_limit=300):
    set_seed(42)

    # 加载数据
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    # 特征工程
    agent = FeatureAgent()
    X_train = agent.engineer_features(train_df)
    X_val = agent.engineer_features(val_df)

    y_train = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    # 特征/目标标准化
    x_scaler = StandardScalerNP()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)

    y_scaler = TargetScalerNP()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    # 样本权重：强调高温/长时/边缘样本
    train_weights = compute_sample_weights(train_df, y_train)

    # 转 tensor
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train_t = torch.FloatTensor(y_train)
    y_val_t = torch.FloatTensor(y_val)
    y_train_scaled_t = torch.FloatTensor(y_train_scaled)
    y_val_scaled_t = torch.FloatTensor(y_val_scaled)
    train_weights_t = torch.FloatTensor(train_weights)

    # 模型
    model = Net(X_train.shape[1])

    # 多任务权重：按目标标准差归一化后已较均衡，这里略加强调强度预测
    task_weight = torch.FloatTensor([1.1, 1.2, 1.2])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=3e-5)

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    batch_size = min(32, max(8, len(X_train) // 4))
    start_time = time.time()
    epoch = 0

    best_state = None
    best_metric = float('inf')
    patience = 80
    stale = 0

    while time.time() - start_time < time_limit - 5:
        model.train()
        perm = torch.randperm(len(X_train))

        # 动态难例加权：基于当前残差提升困难样本权重
        with torch.no_grad():
            pred_all_scaled = model(X_train)
            residual = (pred_all_scaled - y_train_scaled_t).abs().mean(dim=1)
            hard_factor = 1.0 + 0.8 * (residual / (residual.mean() + 1e-6))
            hard_factor = torch.clamp(hard_factor, 1.0, 3.0)
            epoch_weights = train_weights_t * hard_factor
            epoch_weights = epoch_weights / (epoch_weights.mean() + 1e-8)

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb_scaled = y_train_scaled_t[idx]
            wb = epoch_weights[idx]

            optimizer.zero_grad()
            pred_scaled = model(xb)

            loss_main = weighted_huber_loss(
                pred_scaled, yb_scaled, wb, delta=1.0, task_weight=task_weight
            )
            loss_phy = physics_constraint_loss(pred_scaled, yb_scaled)
            loss = loss_main + 0.08 * loss_phy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()
        epoch += 1

        # 早停依据：原始尺度 overall mse
        model.eval()
        with torch.no_grad():
            val_pred_scaled = model(X_val).cpu().numpy()
            val_pred = y_scaler.inverse_transform(val_pred_scaled)
            val_pred_t = torch.FloatTensor(val_pred)
            current_metric = mse(val_pred_t, y_val_t).item()

        if current_metric < best_metric:
            best_metric = current_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if stale >= patience and epoch > 100:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 评估 - 多维度
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_val).cpu().numpy()
        pred = y_scaler.inverse_transform(pred_scaled)
        pred_t = torch.FloatTensor(pred)

        overall_mse = mse(pred_t, y_val_t).item()
        strain_mse = mse(pred_t[:, 0], y_val_t[:, 0]).item()
        tensile_mse = mse(pred_t[:, 1], y_val_t[:, 1]).item()
        yield_mse = mse(pred_t[:, 2], y_val_t[:, 2]).item()

        strain_mae = mae(pred_t[:, 0], y_val_t[:, 0]).item()
        tensile_mae = mae(pred_t[:, 1], y_val_t[:, 1]).item()
        yield_mae = mae(pred_t[:, 2], y_val_t[:, 2]).item()

        strain_rel = (torch.abs(pred_t[:, 0] - y_val_t[:, 0]) / (y_val_t[:, 0].abs() + 1e-8)).mean().item()
        tensile_rel = (torch.abs(pred_t[:, 1] - y_val_t[:, 1]) / (y_val_t[:, 1].abs() + 1e-8)).mean().item()
        yield_rel = (torch.abs(pred_t[:, 2] - y_val_t[:, 2]) / (y_val_t[:, 2].abs() + 1e-8)).mean().item()

    print(f"Epochs: {epoch}")
    print(f"Val Loss: {overall_mse:.4f}")
    print(f"METRICS strain_mse={strain_mse:.4f} tensile_mse={tensile_mse:.4f} yield_mse={yield_mse:.4f}")
    print(f"METRICS strain_mae={strain_mae:.4f} tensile_mae={tensile_mae:.4f} yield_mae={yield_mae:.4f}")
    print(f"METRICS strain_rel={strain_rel:.4f} tensile_rel={tensile_rel:.4f} yield_rel={yield_rel:.4f}")

    # 逐样本误差打印：按总误差降序
    temp_col, time_col = infer_temp_time_columns(val_df)
    temp_vals = val_df[temp_col].values if temp_col is not None else np.arange(len(val_df))
    time_vals = val_df[time_col].values if time_col is not None else np.zeros(len(val_df))

    errs = y_val - pred  # true - pred，正值=低估
    total_abs_err = np.abs(errs).sum(axis=1)
    order = np.argsort(-total_abs_err)

    for idx in order:
        print(
            f"VAL_PRED temp={temp_vals[idx]} time={time_vals[idx]} "
            f"strain_err={errs[idx, 0]:.4f} tensile_err={errs[idx, 1]:.4f} yield_err={errs[idx, 2]:.4f}"
        )

    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')