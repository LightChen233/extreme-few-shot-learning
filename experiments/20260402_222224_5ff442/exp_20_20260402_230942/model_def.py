import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 保守改动：仅在现有小模型基础上加入极轻量 skip，
        # 提高小样本下的稳定性与边界点拟合能力，避免大改架构。
        self.in_proj = nn.Linear(input_dim, 10)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.SiLU(),
            nn.Linear(20, 10),
            nn.SiLU(),
        )

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
        # 轻量残差特征融合：降低过平滑风险，帮助插值点/边界点保留输入信息
        h = self.backbone(x) + 0.25 * self.in_proj(x)
        h = F.silu(h)

        strain = self.strain_head(h)
        yld = self.yield_head(h)
        delta = F.softplus(self.delta_head(h))  # 保证 tensile >= yield
        tensile = yld + delta

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 历史多次大改退化，因此只做极小幅超参调整
    return torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=1.8e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 延续“强度略高权重”的稳健思路，但不激进
    task_weights = torch.tensor([1.0, 1.10, 1.08], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)

        # 保守 hard-example 重加权：仅轻微放大当前 batch 中较难样本，
        # 使用 detach 且权重有上界，避免小样本训练发散。
        sample_err = per_elem.mean(dim=1).detach()
        if len(sample_err) > 1:
            norm_err = sample_err / (sample_err.mean() + 1e-6)
            sample_w = torch.clamp(0.85 + 0.35 * norm_err, 0.85, 1.35)
        else:
            sample_w = torch.ones_like(sample_err)

        base_loss = ((per_elem * task_weights).mean(dim=1) * sample_w).mean()

        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 冗余顺序约束（结构上已满足，但保留极轻权重）
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量局部平滑约束：仅约束相近输入的强度输出不要剧烈跳变
        if len(xb) >= 2:
            pair_idx = torch.roll(torch.arange(len(xb), device=xb.device), shifts=1)
            x_dist = (xb - xb[pair_idx]).pow(2).mean(dim=1)
            t_diff = (tensile_pred - tensile_pred[pair_idx]).pow(2)
            y_diff = (yield_pred - yield_pred[pair_idx]).pow(2)

            smooth_weight = torch.exp(-0.5 * x_dist).detach()
            smooth_penalty = (smooth_weight * (0.55 * t_diff + 0.45 * y_diff)).mean()
        else:
            smooth_penalty = torch.zeros((), device=xb.device, dtype=xb.dtype)

        # 轻量多任务关联约束：抗拉-屈服差值不宜塌缩到过小，
        # 但不硬编码具体数值，仅抑制 delta->0 的退化。
        delta_margin_penalty = F.softplus(1.0 - (tensile_pred - yield_pred)).mean()

        loss = (
            base_loss
            + 0.02 * order_penalty
            + 0.004 * smooth_penalty
            + 0.003 * delta_margin_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()