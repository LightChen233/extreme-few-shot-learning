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
    def __init__(self, dim, dropout=0.06):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.ln2(h)
        return self.act(x + h)


class MultiTaskResidualNet(nn.Module):
    """
    设计目标：
    1. 利用 physics baseline 做残差学习
    2. 共享主干 + 目标专属头，提升 tensile / yield 的协同建模
    3. 用硬约束保证 tensile >= yield
    4. 对高误差区域使用门控修正，增强局部非线性表达
    """
    def __init__(self, input_dim, hidden=192, depth=4, dropout=0.06):
        super().__init__()
        self.input_dim = input_dim

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.shared = nn.Sequential(*[ResidualBlock(hidden, dropout=dropout) for _ in range(depth)])

        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid()
        )

        self.strain_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 1),
        )

        self.tensile_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 1),
        )

        self.gap_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 1),
        )

        self.softplus = nn.Softplus()

    def forward(self, x):
        base = x[:, -3:]  # scaled baseline
        h0 = self.stem(x)
        h = self.shared(h0)
        g = self.gate(h)

        strain_res = self.strain_head(h) * (0.6 + g[:, 0:1])
        tensile_res = self.tensile_head(h) * (0.6 + g[:, 1:2])
        gap_res = self.gap_head(h) * (0.6 + g[:, 2:3])

        strain = base[:, 0:1] + strain_res
        tensile = base[:, 1:2] + tensile_res

        base_gap = torch.relu(base[:, 1:2] - base[:, 2:3])
        gap = self.softplus(gap_res) + 0.10 * base_gap
        yld = tensile - gap

        return torch.cat([strain, tensile, yld], dim=1)


lr = 0.0016
batch_size = 16
weight_decay = 3e-5


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


def build_physics_baseline(train_df, val_df, targets):
    temp_col, time_col = infer_temp_time_cols(train_df)
    if temp_col is None or time_col is None:
        train_base = np.tile(targets.mean(axis=0, keepdims=True), (len(train_df), 1))
        val_base = np.tile(targets.mean(axis=0, keepdims=True), (len(val_df), 1))
        return train_base.astype(np.float32), val_base.astype(np.float32)

    grp = train_df.groupby([temp_col, time_col])[['strain', 'tensile_strength', 'yield_strength']].mean().reset_index()

    merged_train = train_df[[temp_col, time_col]].merge(grp, on=[temp_col, time_col], how='left')
    merged_val = val_df[[temp_col, time_col]].merge(grp, on=[temp_col, time_col], how='left')

    global_mean = train_df[['strain', 'tensile_strength', 'yield_strength']].mean().values.astype(np.float32)

    train_base = merged_train[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    val_base = merged_val[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    for arr in [train_base, val_base]:
        mask = np.isnan(arr)
        if mask.any():
            arr[mask] = np.take(global_mean, np.where(mask)[1])

    return train_base, val_base


def make_features(agent, train_df, val_df):
    X_train = agent.engineer_features(train_df)
    X_val = agent.engineer_features(val_df)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)

    y_train_np = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val_np = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    train_base, val_base = build_physics_baseline(train_df, val_df, y_train_np)

    X_train = np.concatenate([X_train, train_base], axis=1)
    X_val = np.concatenate([X_val, val_base], axis=1)

    return X_train, y_train_np, X_val, y_val_np, train_base.astype(np.float32), val_base.astype(np.float32)


def build_sample_weights(train_df):
    temp_col, time_col = infer_temp_time_cols(train_df)
    n = len(train_df)
    if temp_col is None or time_col is None:
        return np.ones(n, dtype=np.float32)

    grp_count = train_df.groupby([temp_col, time_col]).size().reset_index(name='cnt')
    merged = train_df[[temp_col, time_col]].merge(grp_count, on=[temp_col, time_col], how='left')
    cnt = merged['cnt'].values.astype(np.float32)

    w = 1.0 / np.sqrt(np.maximum(cnt, 1.0))
    w = w / np.mean(w)

    temp_vals = train_df[temp_col].values.astype(np.float32)
    time_vals = train_df[time_col].values.astype(np.float32)

    temp_rank = (temp_vals - temp_vals.min()) / (temp_vals.max() - temp_vals.min() + 1e-6)
    time_rank = (np.log1p(time_vals) - np.log1p(time_vals).min()) / (
        np.log1p(time_vals).max() - np.log1p(time_vals).min() + 1e-6
    )

    high_temp_long_time = 1.0 + 0.20 * temp_rank * time_rank
    mid_time_focus = 1.0 + 0.12 * np.exp(-((time_rank - 0.65) ** 2) / 0.06)

    w = w * high_temp_long_time * mid_time_focus
    w = w / np.mean(w)
    return w.astype(np.float32)


def weighted_huber(pred, target, weight, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return (loss * weight).mean()


def weighted_mse(pred, target, weight):
    return (((pred - target) ** 2) * weight).mean()


def compute_loss(pred_scaled, target_scaled, pred_raw, target_raw, base_raw, sample_weight, epoch_ratio):
    sample_weight = sample_weight.view(-1, 1)

    strain_w = 1.05
    tensile_w = 1.28
    yield_w = 1.25

    strain_loss = weighted_huber(pred_scaled[:, 0:1], target_scaled[:, 0:1], sample_weight, beta=0.75)
    tensile_loss = weighted_huber(pred_scaled[:, 1:2], target_scaled[:, 1:2], sample_weight, beta=0.70)
    yield_loss = weighted_huber(pred_scaled[:, 2:3], target_scaled[:, 2:3], sample_weight, beta=0.70)

    data_loss = strain_w * strain_loss + tensile_w * tensile_loss + yield_w * yield_loss

    mse_aux = weighted_mse(pred_scaled, target_scaled, sample_weight)

    order_penalty = weighted_mse(
        torch.relu(pred_raw[:, 2:3] - pred_raw[:, 1:2]),
        torch.zeros_like(pred_raw[:, 2:3]),
        sample_weight
    )

    true_gap = (target_raw[:, 1:2] - target_raw[:, 2:3]).detach()
    pred_gap = pred_raw[:, 1:2] - pred_raw[:, 2:3]
    gap_loss = weighted_huber(pred_gap, true_gap, sample_weight, beta=2.5)

    residual = pred_raw - base_raw
    residual_reg = (sample_weight * residual.pow(2)).mean()

    base_err = (target_raw - base_raw).detach()
    align_loss = weighted_huber(
        pred_raw - base_raw,
        base_err,
        sample_weight,
        beta=4.0
    )

    under_tensile = torch.relu(target_raw[:, 1:2] - pred_raw[:, 1:2])
    under_yield = torch.relu(target_raw[:, 2:3] - pred_raw[:, 2:3])
    asym_penalty = (sample_weight * (under_tensile.pow(2) + under_yield.pow(2))).mean()

    strain_positive_penalty = weighted_mse(
        torch.relu(-pred_raw[:, 0:1]),
        torch.zeros_like(pred_raw[:, 0:1]),
        sample_weight
    )

    total = (
        data_loss
        + 0.05 * mse_aux
        + 0.20 * gap_loss
        + 0.55 * order_penalty
        + (0.08 * (1.0 - 0.7 * epoch_ratio)) * residual_reg
        + 0.10 * align_loss
        + 0.05 * asym_penalty
        + 0.08 * strain_positive_penalty
    )
    return total


def evaluate_and_print(model, X_val, y_val, y_scaler, val_df):
    model.eval()
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    with torch.no_grad():
        pred_scaled = model(X_val)
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


def train_single_model(X_train, y_train, X_val, y_val, train_base_raw, sample_w, seed, time_limit_inner):
    set_seed(seed)

    x_scaler = RobustScalerTorch().fit(X_train)
    y_scaler = StandardScalerTorch().fit(y_train)

    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    y_train_s = y_scaler.transform(y_train)

    model = MultiTaskResidualNet(X_train_s.shape[1], hidden=192, depth=4, dropout=0.06)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=360,
        pct_start=0.18,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=60.0
    )

    best_state = None
    best_val = float('inf')
    best_epoch = -1

    start_time = time.time()
    epoch = 0
    max_epochs_hint = 360

    while time.time() - start_time < time_limit_inner - 5 and epoch < max_epochs_hint:
        model.train()

        perm = torch.randperm(len(X_train_s))
        epoch_ratio = min(1.0, epoch / max(1, max_epochs_hint - 1))
        current_noise = max(0.0, 0.012 * (1.0 - epoch_ratio))

        for i in range(0, len(X_train_s), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_s[idx]
            yb_scaled = y_train_s[idx]
            yb_raw = y_train[idx]
            base_raw_b = train_base_raw[idx]
            wb = sample_w[idx]

            if current_noise > 0:
                noise = current_noise * torch.randn_like(xb)
                noise[:, -3:] = 0.0
                xb = xb + noise

            optimizer.zero_grad()
            pred_scaled = model(xb)
            pred_raw = y_scaler.inverse_transform(pred_scaled)

            loss = compute_loss(
                pred_scaled=pred_scaled,
                target_scaled=yb_scaled,
                pred_raw=pred_raw,
                target_raw=yb_raw,
                base_raw=base_raw_b,
                sample_weight=wb,
                epoch_ratio=epoch_ratio,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            scheduler.step()

        epoch += 1

        if epoch % 6 == 0:
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_val_s)
                pred_raw = y_scaler.inverse_transform(pred_scaled)
                val_mse = nn.MSELoss()(pred_raw, y_val).item()

            if val_mse < best_val:
                best_val = val_mse
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        if epoch > 110 and epoch - best_epoch > 72:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, x_scaler, y_scaler, best_val


class EnsembleWrapper(nn.Module):
    def __init__(self, models, x_scalers, y_scalers):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.x_scalers = x_scalers
        self.y_scalers = y_scalers

    def forward(self, x_raw):
        preds = []
        for model, x_scaler, y_scaler in zip(self.models, self.x_scalers, self.y_scalers):
            x_scaled = x_scaler.transform(x_raw)
            pred_scaled = model(x_scaled)
            pred_raw = y_scaler.inverse_transform(pred_scaled)
            preds.append(pred_raw)
        return torch.stack(preds, dim=0).mean(dim=0)


def train_model(time_limit=300):
    set_seed(42)

    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    agent = FeatureAgent()
    X_train_np, y_train_np, X_val_np, y_val_np, train_base_np, val_base_np = make_features(agent, train_df, val_df)

    sample_w_np = build_sample_weights(train_df)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    train_base_raw = torch.tensor(train_base_np, dtype=torch.float32)
    sample_w = torch.tensor(sample_w_np, dtype=torch.float32)

    start_all = time.time()

    seeds = [42, 123]
    models = []
    x_scalers = []
    y_scalers = []
    best_vals = []

    remaining_total = time_limit
    per_model_budget = max(110, int((remaining_total - 20) / len(seeds)))

    for i, seed in enumerate(seeds):
        elapsed = time.time() - start_all
        remain = time_limit - elapsed
        if remain < 35:
            break

        inner_budget = min(per_model_budget, int(remain - 10))
        model_i, x_scaler_i, y_scaler_i, best_val_i = train_single_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            train_base_raw=train_base_raw,
            sample_w=sample_w,
            seed=seed,
            time_limit_inner=inner_budget
        )
        models.append(model_i)
        x_scalers.append(x_scaler_i)
        y_scalers.append(y_scaler_i)
        best_vals.append(best_val_i)

    if len(models) == 0:
        model_i, x_scaler_i, y_scaler_i, best_val_i = train_single_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            train_base_raw=train_base_raw,
            sample_w=sample_w,
            seed=42,
            time_limit_inner=max(60, time_limit - 10)
        )
        models = [model_i]
        x_scalers = [x_scaler_i]
        y_scalers = [y_scaler_i]

    ensemble_model = EnsembleWrapper(models, x_scalers, y_scalers)

    print(f"Epochs: ensemble_{len(models)}")
    overall_mse = evaluate_and_print(ensemble_model, X_val, y_val, StandardScalerTorch().fit(y_train), val_df)
    return overall_mse, ensemble_model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')