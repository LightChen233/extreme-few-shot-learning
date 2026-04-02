import time
import math
import copy
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

    def fit(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        self.median = torch.tensor(np.median(x_np, axis=0), dtype=torch.float32)
        q75 = np.percentile(x_np, 75, axis=0)
        q25 = np.percentile(x_np, 25, axis=0)
        scale = q75 - q25
        scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        return self

    def transform(self, x):
        return (x - self.median.to(x.device)) / self.scale.to(x.device)

    def inverse_transform(self, x):
        return x * self.scale.to(x.device) + self.median.to(x.device)


class StandardScalerTorch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        self.mean = torch.tensor(np.mean(x_np, axis=0), dtype=torch.float32)
        std = np.std(x_np, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.std = torch.tensor(std, dtype=torch.float32)
        return self

    def transform(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.08):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResidualNet(nn.Module):
    """
    残差学习：
    - 输入中最后3维为 baseline 的 [strain_base, tensile_base, yield_base]
    - 网络学习 residual，并通过可学习 gate 控制偏移幅度
    - 屈服强度 = 抗拉强度 - softplus(gap)，保证 yield <= tensile
    """
    def __init__(self, input_dim, hidden=160, dropout=0.08):
        super().__init__()
        feat_dim = input_dim - 3
        self.feat_dim = feat_dim

        self.feat_stem = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.base_stem = nn.Sequential(
            nn.Linear(3, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.shared = nn.Sequential(
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
        )

        self.res_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )

        self.gate_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid(),
        )

        self.softplus = nn.Softplus()

    def forward(self, x_scaled, baseline_scaled):
        feat = x_scaled[:, :self.feat_dim]
        h_feat = self.feat_stem(feat)
        h_base = self.base_stem(baseline_scaled)
        h = self.fuse(torch.cat([h_feat, h_base], dim=1))
        h = self.shared(h)

        residual = self.res_head(h)
        gate = self.gate_head(h)

        raw_out = baseline_scaled + gate * residual

        strain = raw_out[:, 0:1]
        tensile = raw_out[:, 1:2]

        # 使用 baseline gap 初始化，更稳定
        base_gap = torch.relu(baseline_scaled[:, 1:2] - baseline_scaled[:, 2:3])
        pred_gap = self.softplus(raw_out[:, 2:3] + base_gap)
        yld = tensile - pred_gap

        return torch.cat([strain, tensile, yld], dim=1)


lr = 0.0025
batch_size = 16
weight_decay = 8e-5


def infer_temp_time_cols(df):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    temp_candidates = ['temp', 'temperature', 't']
    time_candidates = ['time', 'holding_time', 'duration']
    temp_col = None
    time_col = None
    for c in temp_candidates:
        if c in lower_map:
            temp_col = lower_map[c]
            break
    for c in time_candidates:
        if c in lower_map:
            time_col = lower_map[c]
            break

    if temp_col is None:
        for c in cols:
            if 'temp' in c.lower():
                temp_col = c
                break
    if time_col is None:
        for c in cols:
            lc = c.lower()
            if 'time' in lc or 'hour' in lc or 'min' in lc:
                time_col = c
                break
    return temp_col, time_col


def build_group_stats(train_df):
    temp_col, time_col = infer_temp_time_cols(train_df)
    if temp_col is None or time_col is None:
        return None, None, None

    grp_mean = train_df.groupby([temp_col, time_col])[['strain', 'tensile_strength', 'yield_strength']].mean()
    grp_count = train_df.groupby([temp_col, time_col]).size()
    global_mean = train_df[['strain', 'tensile_strength', 'yield_strength']].mean().values.astype(np.float32)
    return grp_mean, grp_count, global_mean


def build_physics_baseline(train_df, val_df, targets):
    temp_col, time_col = infer_temp_time_cols(train_df)
    if temp_col is None or time_col is None:
        train_base = np.tile(targets.mean(axis=0, keepdims=True), (len(train_df), 1))
        val_base = np.tile(targets.mean(axis=0, keepdims=True), (len(val_df), 1))
        train_w = np.ones(len(train_df), dtype=np.float32)
        val_w = np.ones(len(val_df), dtype=np.float32)
        return train_base.astype(np.float32), val_base.astype(np.float32), train_w, val_w

    grp_mean = train_df.groupby([temp_col, time_col])[['strain', 'tensile_strength', 'yield_strength']].mean().reset_index()
    grp_count = train_df.groupby([temp_col, time_col]).size().reset_index(name='count')

    merged_train = train_df[[temp_col, time_col]].merge(grp_mean, on=[temp_col, time_col], how='left')
    merged_train = merged_train.merge(grp_count, on=[temp_col, time_col], how='left')

    merged_val = val_df[[temp_col, time_col]].merge(grp_mean, on=[temp_col, time_col], how='left')
    merged_val = merged_val.merge(grp_count, on=[temp_col, time_col], how='left')

    global_mean = train_df[['strain', 'tensile_strength', 'yield_strength']].mean().values.astype(np.float32)

    train_base = merged_train[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    val_base = merged_val[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    for arr in [train_base, val_base]:
        mask = np.isnan(arr)
        if mask.any():
            arr[mask] = np.take(global_mean, np.where(mask)[1])

    # 稀有/难点工艺加权：count 越少，权重越高；seen group 略高于 unseen group
    train_count = merged_train['count'].fillna(1).values.astype(np.float32)
    val_count = merged_val['count'].fillna(0).values.astype(np.float32)

    train_w = 1.0 + 0.6 / np.sqrt(np.maximum(train_count, 1.0))
    val_w = 1.0 + 0.6 / np.sqrt(np.maximum(val_count, 1.0))
    val_w[val_count <= 0] = 1.6

    return train_base, val_base, train_w.astype(np.float32), val_w.astype(np.float32)


def make_features(agent, train_df, val_df):
    X_train = agent.engineer_features(train_df)
    X_val = agent.engineer_features(val_df)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)

    y_train_np = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val_np = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    train_base, val_base, train_w, val_w = build_physics_baseline(train_df, val_df, y_train_np)

    X_train = np.concatenate([X_train, train_base], axis=1)
    X_val = np.concatenate([X_val, val_base], axis=1)

    return X_train, y_train_np, X_val, y_val_np, train_base, val_base, train_w, val_w


def weighted_huber(pred, target, weight, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    if weight.dim() == 1:
        weight = weight.unsqueeze(1)
    return (loss * weight).mean()


def compute_loss(pred_scaled, target_scaled, pred_raw, target_raw, baseline_raw, sample_weight, epoch):
    # 前期更关注稳定拟合，后期增加高误差点/残差学习约束
    alpha = min(1.0, epoch / 120.0)

    w = sample_weight
    huber_beta = 0.9

    strain_loss = weighted_huber(pred_scaled[:, 0], target_scaled[:, 0], w, beta=huber_beta)
    tensile_loss = weighted_huber(pred_scaled[:, 1], target_scaled[:, 1], w, beta=huber_beta)
    yield_loss = weighted_huber(pred_scaled[:, 2], target_scaled[:, 2], w, beta=huber_beta)

    # 略提高 yield 权重，修复上一轮屈服性能退化
    data_loss = 1.10 * strain_loss + 1.00 * tensile_loss + 1.18 * yield_loss

    # 排序物理约束：yield <= tensile
    order_penalty = (torch.relu(pred_raw[:, 2] - pred_raw[:, 1]).pow(2) * w).mean()

    # gap 对齐：比直接拟合 yield 更稳定
    true_gap = (target_raw[:, 1] - target_raw[:, 2]).detach()
    pred_gap = pred_raw[:, 1] - pred_raw[:, 2]
    gap_loss = weighted_huber(pred_gap, true_gap, w, beta=10.0)

    # 残差学习约束：鼓励预测不要无端偏离 baseline，但不强压制
    true_res = target_raw - baseline_raw
    pred_res = pred_raw - baseline_raw
    residual_loss = weighted_huber(pred_res, true_res, w, beta=8.0)

    # 多任务相关性约束：UTS 和 YS 通常正相关，方向一致
    true_center = target_raw - target_raw.mean(dim=0, keepdim=True)
    pred_center = pred_raw - pred_raw.mean(dim=0, keepdim=True)

    true_ty = (true_center[:, 1] * true_center[:, 2]).mean().detach()
    pred_ty = (pred_center[:, 1] * pred_center[:, 2]).mean()
    corr_loss = (pred_ty - true_ty).pow(2) / (true_ty.abs() + 1.0)

    # 小比例MSE稳定训练
    mse_aux = ((pred_scaled - target_scaled).pow(2) * w.unsqueeze(1)).mean()

    total = (
        data_loss
        + (0.22 + 0.10 * alpha) * gap_loss
        + 0.60 * order_penalty
        + (0.12 + 0.10 * alpha) * residual_loss
        + 0.03 * corr_loss
        + 0.04 * mse_aux
    )
    return total


def evaluate_and_print(model, X_val, y_val, y_scaler, val_df, baseline_val_scaled):
    model.eval()
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    with torch.no_grad():
        pred_scaled = model(X_val, baseline_val_scaled)
        pred = y_scaler.inverse_transform(pred_scaled)

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

    print(f"Val Loss: {overall_mse:.4f}")
    print(f"METRICS strain_mse={strain_mse:.4f} tensile_mse={tensile_mse:.4f} yield_mse={yield_mse:.4f}")
    print(f"METRICS strain_mae={strain_mae:.4f} tensile_mae={tensile_mae:.4f} yield_mae={yield_mae:.4f}")
    print(f"METRICS strain_rel={strain_rel:.4f} tensile_rel={tensile_rel:.4f} yield_rel={yield_rel:.4f}")

    temp_col, time_col = infer_temp_time_cols(val_df)
    pred_np = pred.detach().cpu().numpy()
    y_np = y_val.detach().cpu().numpy()

    err_total = np.sum(np.abs(y_np - pred_np), axis=1)
    order = np.argsort(-err_total)

    for idx in order:
        temp_val = val_df.iloc[idx][temp_col] if temp_col is not None else -1
        time_val = val_df.iloc[idx][time_col] if time_col is not None else -1
        strain_err = y_np[idx, 0] - pred_np[idx, 0]
        tensile_err = y_np[idx, 1] - pred_np[idx, 1]
        yield_err = y_np[idx, 2] - pred_np[idx, 2]
        print(
            f"VAL_PRED temp={temp_val} time={time_val} "
            f"strain_err={strain_err:.4f} tensile_err={tensile_err:.4f} yield_err={yield_err:.4f}"
        )

    return overall_mse


def train_model(time_limit=300):
    set_seed(42)

    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    agent = FeatureAgent()
    X_train_np, y_train_np, X_val_np, y_val_np, train_base_np, val_base_np, train_w_np, val_w_np = make_features(agent, train_df, val_df)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)

    train_base = torch.tensor(train_base_np, dtype=torch.float32)
    val_base = torch.tensor(val_base_np, dtype=torch.float32)
    train_w = torch.tensor(train_w_np, dtype=torch.float32)
    val_w = torch.tensor(val_w_np, dtype=torch.float32)

    x_scaler = RobustScalerTorch().fit(X_train)
    y_scaler = StandardScalerTorch().fit(y_train)
    base_scaler = StandardScalerTorch().fit(train_base)

    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)

    y_train_scaled = y_scaler.transform(y_train)
    base_train_scaled = base_scaler.transform(train_base)
    base_val_scaled = base_scaler.transform(val_base)

    model = ResidualNet(X_train.shape[1], hidden=160, dropout=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=400,
        pct_start=0.18,
        anneal_strategy='cos',
        div_factor=12.0,
        final_div_factor=80.0,
    )

    best_state = None
    best_val = float('inf')

    start_time = time.time()
    epoch = 0
    step_count = 0

    while time.time() - start_time < time_limit - 8:
        model.train()
        perm = torch.randperm(len(X_train))

        # 前期稍强增强，后期减弱
        feat_noise = max(0.0, 0.02 * (1.0 - epoch / 220.0))
        base_mix = max(0.0, 0.08 * (1.0 - epoch / 180.0))

        for i in range(0, len(X_train), batch_size):
            if time.time() - start_time >= time_limit - 8:
                break

            idx = perm[i:i + batch_size]
            xb = X_train[idx].clone()
            yb_scaled = y_train_scaled[idx]
            yb_raw = y_train[idx]
            bb_raw = train_base[idx]
            bb_scaled = base_train_scaled[idx]
            wb = train_w[idx]

            if feat_noise > 0:
                xb[:, :-3] = xb[:, :-3] + feat_noise * torch.randn_like(xb[:, :-3])

            if base_mix > 0 and len(idx) > 1:
                shuffle_idx = idx[torch.randperm(len(idx))]
                mixed_raw = (1.0 - base_mix) * bb_raw + base_mix * train_base[shuffle_idx]
                bb_raw = mixed_raw
                bb_scaled = base_scaler.transform(bb_raw)

            optimizer.zero_grad()
            pred_scaled = model(xb, bb_scaled)
            pred_raw = y_scaler.inverse_transform(pred_scaled)

            loss = compute_loss(pred_scaled, yb_scaled, pred_raw, yb_raw, bb_raw, wb, epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if step_count < 400:
                scheduler.step()
            step_count += 1

        epoch += 1

        if epoch % 8 == 0:
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_val, base_val_scaled)
                pred_raw = y_scaler.inverse_transform(pred_scaled)

                # 验证时也轻微关注稀有组
                mse_per_sample = ((pred_raw - y_val) ** 2).mean(dim=1)
                val_mse = (mse_per_sample * val_w).mean().item()

            if val_mse < best_val:
                best_val = val_mse
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Epochs: {epoch}")
    overall_mse = evaluate_and_print(model, X_val, y_val, y_scaler, val_df, base_val_scaled)
    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')