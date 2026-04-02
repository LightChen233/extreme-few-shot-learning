"""
保守改进版 model_def.py

设计原则：
1) 延续当前较优基线：小型共享主干 + 多任务头 + tensile >= yield 的硬约束参数化
2) 只做小步改动，不重写架构
3) 针对当前误差特征做两点轻量修正：
   - 外推点与局部突变点并存，但历史上激进 hard mining 会崩，因此采用“有上界的温和动态重加权”
   - 460/12（训练覆盖点）仍误差较大，说明局部规律与温度-时间交互仍需加强；加入极轻量残差连接提升低样本拟合稳定性
4) 保持数值稳定：不对输入求梯度，不使用高阶梯度，保留梯度裁剪

相对当前代码的核心变化：
- backbone 改为更稳的“小残差 MLP”，容量几乎不变
- Huber beta 按任务尺度区分：应变更敏感，强度更平滑
- 使用基于 detach 残差的 capped hard-example reweighting，权重限制在 [1, 1.8]
- 保留 tensile >= yield 硬约束
- 保留轻量平滑正则，但进一步减弱，避免过度抹平 470/12 等可能的非线性变化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        return self.norm(x + 0.5 * h)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = 16

        self.input_proj = nn.Linear(input_dim, hidden)
        self.block1 = ResidualBlock(hidden)
        self.mid = nn.Linear(hidden, 10)
        self.mid_act = nn.SiLU()

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
        h = self.input_proj(x)
        h = F.silu(h)
        h = self.block1(h)
        h = self.mid(h)
        h = self.mid_act(h)

        strain = self.strain_head(h)
        yld = self.yield_head(h)
        delta = F.softplus(self.delta_head(h))
        tensile = yld + delta

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=1.8e-4)


def _smooth_l1_per_task(pred, target):
    """
    对不同任务使用不同 beta 的 Huber，兼顾应变与强度尺度差异。
    返回 [B, 3]
    """
    err = pred - target
    abs_err = err.abs()

    betas = torch.tensor([0.6, 6.0, 5.0], device=pred.device, dtype=pred.dtype).view(1, 3)
    quad = 0.5 * err.pow(2) / betas
    lin = abs_err - 0.5 * betas
    return torch.where(abs_err < betas, quad, lin)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    # 强度略高权重，但保持克制
    task_weights = torch.tensor([1.0, 1.10, 1.08], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        per_elem = _smooth_l1_per_task(pred, yb)  # [B,3]

        # 温和 hard-example 重加权：仅轻微关注难例，避免历史版本那种失稳
        with torch.no_grad():
            norm_resid = (pred.detach() - yb).abs()
            scale = torch.tensor([1.5, 18.0, 15.0], device=xb.device, dtype=xb.dtype).view(1, 3)
            norm_resid = (norm_resid / scale).mean(dim=1)  # [B]
            sample_w = 1.0 + 0.8 * torch.tanh(norm_resid)   # in [1, ~1.8)
            sample_w = sample_w.clamp(1.0, 1.8)

        weighted = (per_elem * task_weights) * sample_w.unsqueeze(1)
        base_loss = weighted.mean()

        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 虽然结构已保证，但保留极轻量冗余项
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量局部平滑：仅弱约束，避免把可能的峰值/突变过度抹平
        if len(xb) >= 2:
            pair_idx = torch.roll(torch.arange(len(xb), device=xb.device), shifts=1)
            x_dist = (xb - xb[pair_idx]).pow(2).mean(dim=1)
            t_diff = (tensile_pred - tensile_pred[pair_idx]).pow(2)
            y_diff = (yield_pred - yield_pred[pair_idx]).pow(2)
            smooth_weight = torch.exp(-0.7 * x_dist).detach()
            smooth_penalty = (smooth_weight * (0.55 * t_diff + 0.45 * y_diff)).mean()
        else:
            smooth_penalty = torch.zeros((), device=xb.device, dtype=xb.dtype)

        # 多任务一致性：strain 不直接强绑强度，只做极弱能量约束避免头部发散
        strain_pred = pred[:, 0]
        centered_strength = (tensile_pred + yield_pred) * 0.5
        consistency_penalty = 1e-4 * (
            strain_pred.pow(2).mean() + centered_strength.pow(2).mean()
        )

        loss = base_loss + 0.01 * order_penalty + 0.003 * smooth_penalty + consistency_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()