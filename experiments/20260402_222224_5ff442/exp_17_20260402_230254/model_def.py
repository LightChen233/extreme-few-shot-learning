import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 保守小模型：维持低容量，避免 29 条样本下过拟合
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
        )

        # 应变单独头
        self.strain_head = nn.Sequential(
            nn.Linear(8, 6),
            nn.SiLU(),
            nn.Linear(6, 1)
        )

        # 强度共享一层后再拆分，增强 tensile/yield 耦合学习
        self.strength_shared = nn.Sequential(
            nn.Linear(8, 6),
            nn.SiLU(),
        )

        self.yield_head = nn.Linear(6, 1)
        self.delta_head = nn.Linear(6, 1)

    def forward(self, x):
        h = self.backbone(x)

        strain = self.strain_head(h)

        hs = self.strength_shared(h)
        yld = self.yield_head(hs)
        delta = F.softplus(self.delta_head(hs))
        tensile = yld + delta  # 保证 tensile >= yield

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 极小幅调整：更稳一点的 lr，轻微正则
    return torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=2.0e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 针对当前主要问题：略提高屈服权重，但保持克制，避免再次明显 trade-off
    task_weights = torch.tensor([1.0, 1.10, 1.18], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        per_elem = criterion(pred, yb)

        # 温和 hard-example 重加权：仅按 batch 内残差放大，且上限严格限制
        with torch.no_grad():
            abs_res = (pred - yb).abs()
            sample_score = (
                0.8 * abs_res[:, 0] +
                1.0 * abs_res[:, 1] +
                1.1 * abs_res[:, 2]
            )
            norm_score = sample_score / (sample_score.mean() + 1e-6)
            sample_weights = torch.clamp(0.75 + 0.5 * norm_score, 0.75, 1.6)

        weighted_loss = (per_elem * task_weights.unsqueeze(0)).mean(dim=1)
        base_loss = (weighted_loss * sample_weights).mean()

        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 冗余顺序约束
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量局部平滑：减少插值点(如 460,12)附近波动，但不强压远距离样本
        if len(xb) >= 2:
            pair_idx = torch.roll(torch.arange(len(xb), device=xb.device), shifts=1)
            x_dist = (xb - xb[pair_idx]).pow(2).mean(dim=1)

            t_diff = (tensile_pred - tensile_pred[pair_idx]).pow(2)
            y_diff = (yield_pred - yield_pred[pair_idx]).pow(2)

            smooth_weight = torch.exp(-0.6 * x_dist).detach()
            smooth_penalty = (smooth_weight * (0.45 * t_diff + 0.55 * y_diff)).mean()
        else:
            smooth_penalty = torch.zeros((), device=xb.device, dtype=xb.dtype)

        loss = base_loss + 0.02 * order_penalty + 0.004 * smooth_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()