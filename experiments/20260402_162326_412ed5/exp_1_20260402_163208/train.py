import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_agent import FeatureAgent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RobustScalerTorch:
    def __init__(self):
        self.center = None
        self.scale = None

    def fit(self, x):
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
        self.center = np.median(arr, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        scale = q75 - q25
        scale[scale < 1e-6] = 1.0
        self.scale = scale
        return self

    def transform(self, x):
        if isinstance(x, torch.Tensor):
            center = torch.tensor(self.center, dtype=x.dtype, device=x.device)
            scale = torch.tensor(self.scale, dtype=x.dtype, device=x.device)
            return (x - center) / scale
        return (x - self.center) / self.scale

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 128
        trunk_hidden = 96
        expert_hidden = 64
        self.num_experts = 3

        self.input_bn = nn.BatchNorm1d(input_dim)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, trunk_hidden),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, 3)
            )
            for _ in range(self.num_experts)
        ])

        self.shared_head = nn.Sequential(
            nn.Linear(trunk_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x_bn = self.input_bn(x)
        feat = self.trunk(x_bn)
        gate_logits = self.gate(x_bn)
        gate_w = torch.softmax(gate_logits, dim=-1)

        expert_outs = torch.stack([expert(feat) for expert in self.experts], dim=1)
        moe_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)
        shared_out = self.shared_head(feat)
        out = shared_out + moe_out
        return out, gate_w


lr = 0.003
batch_size = 16
weight_decay = 1e-4


def compute_sample_weights(y_true):
    y_np = y_true.cpu().numpy()
    med = np.median(y_np, axis=0, keepdims=True)
    mad = np.median(np.abs(y_np - med), axis=0, keepdims=True) + 1e-6
    z = np.abs((y_np - med) / mad)
    hard = z.mean(axis=1)
    weights = 1.0 + 0.15 * np.clip(hard, 0, 8)
    return torch.tensor(weights, dtype=torch.float32)


def weighted_huber_multitask(pred, target, sample_weights, task_weights, delta=1.0):
    err = pred - target
    abs_err = torch.abs(err)
    huber = torch.where(abs_err < delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
    weighted = huber * task_weights.view(1, -1)
    weighted = weighted.mean(dim=1) * sample_weights
    return weighted.mean()


def physical_constraint_loss(pred):
    strain = pred[:, 0]
    tensile = pred[:, 1]
    yld = pred[:, 2]

    order_penalty = F.relu(yld - tensile).mean()

    positive_penalty = (
        F.relu(-strain).mean() +
        F.relu(-tensile).mean() +
        F.relu(-yld).mean()
    )
    return order_penalty + 0.1 * positive_penalty


def gate_entropy_loss(gate_w):
    entropy = -(gate_w * torch.log(gate_w + 1e-8)).sum(dim=1).mean()
    return -entropy


def evaluate_and_print(model, X_val, y_val, scaler_y, val_df):
    model.eval()
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    with torch.no_grad():
        pred_scaled, gate_w = model(X_val)
        pred = scaler_y.inverse_transform(pred_scaled)
        true = scaler_y.inverse_transform(y_val)

        overall_mse = mse(pred, true).item()
        strain_mse = mse(pred[:, 0], true[:, 0]).item()
        tensile_mse = mse(pred[:, 1], true[:, 1]).item()
        yield_mse = mse(pred[:, 2], true[:, 2]).item()

        strain_mae = mae(pred[:, 0], true[:, 0]).item()
        tensile_mae = mae(pred[:, 1], true[:, 1]).item()
        yield_mae = mae(pred[:, 2], true[:, 2]).item()

        strain_rel = (torch.abs(pred[:, 0] - true[:, 0]) / (true[:, 0].abs() + 1e-8)).mean().item()
        tensile_rel = (torch.abs(pred[:, 1] - true[:, 1]) / (true[:, 1].abs() + 1e-8)).mean().item()
        yield_rel = (torch.abs(pred[:, 2] - true[:, 2]) / (true[:, 2].abs() + 1e-8)).mean().item()

        print(f"Val Loss: {overall_mse:.4f}")
        print(f"METRICS strain_mse={strain_mse:.4f} tensile_mse={tensile_mse:.4f} yield_mse={yield_mse:.4f}")
        print(f"METRICS strain_mae={strain_mae:.4f} tensile_mae={tensile_mae:.4f} yield_mae={yield_mae:.4f}")
        print(f"METRICS strain_rel={strain_rel:.4f} tensile_rel={tensile_rel:.4f} yield_rel={yield_rel:.4f}")

        pred_np = pred.cpu().numpy()
        true_np = true.cpu().numpy()

        temps = val_df['temp'].values if 'temp' in val_df.columns else np.zeros(len(val_df))
        times = val_df['time'].values if 'time' in val_df.columns else np.arange(len(val_df))

        for i in range(len(val_df)):
            strain_err = true_np[i, 0] - pred_np[i, 0]
            tensile_err = true_np[i, 1] - pred_np[i, 1]
            yield_err = true_np[i, 2] - pred_np[i, 2]
            print(
                f"VAL_PRED temp={temps[i]} time={times[i]} "
                f"strain_err={strain_err:.4f} tensile_err={tensile_err:.4f} yield_err={yield_err:.4f}"
            )

    return overall_mse


def train_model(time_limit=300):
    set_seed(42)

    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    agent = FeatureAgent()
    X_train_np = agent.engineer_features(train_df)
    X_val_np = agent.engineer_features(val_df)

    y_train_np = train_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)
    y_val_np = val_df[['strain', 'tensile_strength', 'yield_strength']].values.astype(np.float32)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_train_raw = torch.tensor(y_train_np, dtype=torch.float32)
    y_val_raw = torch.tensor(y_val_np, dtype=torch.float32)

    scaler_x = RobustScalerTorch()
    scaler_y = RobustScalerTorch()
    X_train = scaler_x.fit_transform(X_train)
    X_val = scaler_x.transform(X_val)
    y_train = scaler_y.fit_transform(y_train_raw)
    y_val = scaler_y.transform(y_val_raw)

    model = Net(X_train.shape[1])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=5e-5)

    sample_weights = compute_sample_weights(y_train_raw)

    y_scale = torch.tensor(scaler_y.scale, dtype=torch.float32)
    task_weights = 1.0 / (y_scale ** 2 + 1e-6)
    task_weights = task_weights / task_weights.mean()

    best_state = None
    best_metric = float('inf')

    start_time = time.time()
    epoch = 0

    while time.time() - start_time < time_limit - 5:
        model.train()
        perm = torch.randperm(len(X_train))

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            sw = sample_weights[idx]

            optimizer.zero_grad()
            pred, gate_w = model(xb)

            base_loss = weighted_huber_multitask(pred, yb, sw, task_weights, delta=1.0)

            pred_raw = scaler_y.inverse_transform(pred)
            phys_loss = physical_constraint_loss(pred_raw)
            ent_loss = gate_entropy_loss(gate_w)

            loss = base_loss + 0.05 * phys_loss + 0.002 * ent_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()
        epoch += 1

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred_scaled, _ = model(X_val)
                val_pred = scaler_y.inverse_transform(val_pred_scaled)
                val_true = scaler_y.inverse_transform(y_val)
                val_metric = ((val_pred - val_true) ** 2).mean().item()
                if val_metric < best_metric:
                    best_metric = val_metric
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Epochs: {epoch}")
    overall_mse = evaluate_and_print(model, X_val, y_val, scaler_y, val_df)
    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')