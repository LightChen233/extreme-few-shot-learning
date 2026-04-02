import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 保守小模型：仅在当前基线上加入极轻量残差，增强稳定拟合能力
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.SiLU(),
            nn.Linear(20, 10),
            nn.SiLU(),
        )

        self.backbone_res = nn.Linear(input_dim, 10)

        self.strain_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

        self.yield_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

        self.delta_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        h = self.backbone(x) + 0.15 * self.backbone_res(x)

        strain = self.strain_head(h)
        yld = self.yield_head(h)
        delta = F.softplus(self.delta_head(h))
        tensile = yld + delta

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=4.0e-4,
        weight_decay=1.2e-4
    )


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 小幅强调强度任务，但避免破坏当前较优基线
    task_weights = torch.tensor(
        [1.0, 1.10, 1.08],
        device=X_train.device,
        dtype=X_train.dtype
    )

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)

        # 温和难例重加权：仅在 batch 内、基于 detach 残差、并严格截断
        with torch.no_grad():
            sample_err = per_elem.mean(dim=1)
            sample_w = 1.0 + 0.6 * sample_err / (sample_err.mean() + 1e-6)
            sample_w = sample_w.clamp(1.0, 1.8)

        base_loss = ((per_elem * task_weights).mean(dim=1) * sample_w).mean()

        strain_pred = pred[:, 0]
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 冗余顺序约束
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量局部平滑：防止插值点波动过大
        if len(xb) >= 2:
            pair_idx = torch.roll(torch.arange(len(xb), device=xb.device), shifts=1)
            x_dist = (xb - xb[pair_idx]).pow(2).mean(dim=1)

            s_diff = (strain_pred - strain_pred[pair_idx]).pow(2)
            t_diff = (tensile_pred - tensile_pred[pair_idx]).pow(2)
            y_diff = (yield_pred - yield_pred[pair_idx]).pow(2)

            smooth_weight = torch.exp(-0.5 * x_dist).detach()
            smooth_penalty = (
                smooth_weight * (0.20 * s_diff + 0.45 * t_diff + 0.35 * y_diff)
            ).mean()
        else:
            smooth_penalty = torch.zeros((), device=xb.device, dtype=xb.dtype)

        # 强度间弱耦合：抗拉与屈服应共享部分趋势，但不强行绑定残差
        strength_gap = tensile_pred - yield_pred
        gap_reg = F.relu(0.0 - strength_gap).mean()

        loss = base_loss + 0.02 * order_penalty + 0.004 * smooth_penalty + 0.01 * gap_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()