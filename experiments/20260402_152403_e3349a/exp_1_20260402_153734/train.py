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
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class Net(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.shared = nn.Sequential(
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
        )
        self.strain_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.tensile_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.shared(self.stem(x))
        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        delta = self.softplus(self.delta_head(h))
        yld = tensile - delta
        return torch.cat([strain, tensile, yld], dim=1)


lr = 0.003
batch_size = 16
weight_decay = 1e-4


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

    return X_train, y_train_np, X_val, y_val_np


def compute_loss(pred_scaled, target_scaled, pred_raw, target_raw):
    huber = nn.SmoothL1Loss(beta=0.8)
    mse = nn.MSELoss()

    strain_loss = huber(pred_scaled[:, 0], target_scaled[:, 0])
    tensile_loss = huber(pred_scaled[:, 1], target_scaled[:, 1])
    yield_loss = huber(pred_scaled[:, 2], target_scaled[:, 2])

    data_loss = 1.2 * strain_loss + 1.0 * tensile_loss + 1.0 * yield_loss

    # 通用物理约束：抗拉强度通常不低于屈服强度
    order_penalty = torch.relu(pred_raw[:, 2] - pred_raw[:, 1]).pow(2).mean()

    # 与真实差值尺度对齐的软约束，避免 tensile 与 yield 次序异常
    true_gap = (target_raw[:, 1] - target_raw[:, 2]).detach()
    pred_gap = pred_raw[:, 1] - pred_raw[:, 2]
    gap_loss = huber(pred_gap, true_gap)

    total = data_loss + 0.2 * gap_loss + 0.5 * order_penalty + 0.05 * mse(pred_scaled, target_scaled)
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


def train_model(time_limit=300):
    set_seed(42)

    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    agent = FeatureAgent()
    X_train_np, y_train_np, X_val_np, y_val_np = make_features(agent, train_df, val_df)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)

    x_scaler = RobustScalerTorch().fit(X_train)
    y_scaler = StandardScalerTorch().fit(y_train)

    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)
    y_train_scaled = y_scaler.transform(y_train)

    model = Net(X_train.shape[1], hidden=128, dropout=0.12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-5)

    best_state = None
    best_val = float('inf')

    start_time = time.time()
    epoch = 0

    while time.time() - start_time < time_limit - 5:
        model.train()
        perm = torch.randperm(len(X_train))

        current_noise = max(0.0, 0.03 * (1.0 - epoch / 300.0))

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb_scaled = y_train_scaled[idx]
            yb_raw = y_train[idx]

            if current_noise > 0:
                xb = xb + current_noise * torch.randn_like(xb)

            optimizer.zero_grad()
            pred_scaled = model(xb)
            pred_raw = y_scaler.inverse_transform(pred_scaled)
            loss = compute_loss(pred_scaled, yb_scaled, pred_raw, yb_raw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()
        epoch += 1

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_val)
                pred_raw = y_scaler.inverse_transform(pred_scaled)
                val_mse = nn.MSELoss()(pred_raw, y_val).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Epochs: {epoch}")
    overall_mse = evaluate_and_print(model, X_val, y_val, y_scaler, val_df)
    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')