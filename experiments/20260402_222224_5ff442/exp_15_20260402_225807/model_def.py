import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 小样本下保持低容量；相对基线仅做轻微增强：
        # 1) backbone 稍加 LayerNorm 提升不同特征尺度下的稳定性
        # 2) 保留多任务头
        # 3) tensile = yield + softplus(delta) 维持物理顺序
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.SiLU(),
            nn.LayerNorm(24),
            nn.Linear(24, 12),
            nn.SiLU(),
        )

        self.strain_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

        self.yield_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

        self.delta_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

        # 极轻量任务不确定性加权参数，训练时自动平衡三任务
        # 初始化为 0，等价于初始同权重；后续会被轻微学习
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        h = self.backbone(x)

        strain = self.strain_head(h)
        yld = self.yield_head(h)
        delta = F.softplus(self.delta_head(h))
        tensile = yld + delta

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 延续保守策略：仅小幅微调 lr / wd
    return torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=2e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)
    device = X_train.device
    dtype = X_train.dtype

    perm = torch.randperm(n, device=device)

    # Huber 更稳，尤其对少数大误差点
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 轻微偏向强度任务，但比固定大权重更保守
    static_task_weights = torch.tensor([1.0, 1.10, 1.12], device=device, dtype=dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)  # [B, 3]

        # -------- 多任务不确定性加权（带 clamp，防止数值爆炸） --------
        log_vars = torch.clamp(model.log_vars, min=-2.5, max=2.5)
        inv_vars = torch.exp(-log_vars).clamp(0.08, 12.0)

        task_loss = per_elem.mean(dim=0) * static_task_weights
        weighted_task_loss = (inv_vars * task_loss + 0.5 * log_vars).sum()

        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 冗余顺序约束（虽然结构已满足，但保留极轻约束）
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # -------- 轻量局部平滑正则 --------
        # 减少插值点附近波动，但避免过强拉平外推斜率
        if len(xb) >= 2:
            pair_idx = torch.roll(torch.arange(len(xb), device=device), shifts=1)
            x_dist = (xb - xb[pair_idx]).pow(2).mean(dim=1)

            t_diff = (tensile_pred - tensile_pred[pair_idx]).pow(2)
            y_diff = (yield_pred - yield_pred[pair_idx]).pow(2)
            s_diff = (pred[:, 0] - pred[pair_idx, 0]).pow(2)

            smooth_weight = torch.exp(-0.6 * x_dist).detach()
            smooth_penalty = (
                smooth_weight * (0.45 * s_diff + 0.75 * t_diff + 0.85 * y_diff)
            ).mean()
        else:
            smooth_penalty = torch.zeros((), device=device, dtype=dtype)

        # -------- 轻量 hard-example 重加权（有上界，且基于 detach） --------
        # 只做 batch 内温和聚焦，避免像历史版本那样过度放大小样本噪声
        abs_err = (pred.detach() - yb).abs()
        norm_err = abs_err / (abs_err.mean(dim=0, keepdim=True) + 1e-6)
        sample_weight = 1.0 + 0.35 * norm_err.mean(dim=1)
        sample_weight = sample_weight.clamp(1.0, 1.8)

        focal_loss = ((per_elem.mean(dim=1) * sample_weight).mean())

        # 主损失 = 不确定性加权任务损失 + 少量难例聚焦 + 很轻正则
        loss = (
            0.82 * weighted_task_loss
            + 0.18 * focal_loss
            + 0.02 * order_penalty
            + 0.004 * smooth_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()