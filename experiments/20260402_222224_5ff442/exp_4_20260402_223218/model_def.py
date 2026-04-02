"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本次改动遵循“保守改进”原则：
1) 保持小模型与多任务头，不重写整体架构
2) 在共享表示上加入 very-light residual/gating，增强对局部非线性/分段规律的表达
3) 保留 SmoothL1 稳健损失与 hard-example 重加权，但降低激进程度，避免过度追逐离群点
4) 引入可学习多任务不确定性权重（做 clamp 保证稳定），缓解三目标 trade-off
5) 保留 tensile >= yield 的物理约束，并加入轻量输出平滑正则以稳定小样本训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 共享主干：容量仍然较小，适合 29 条样本
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # 轻量 gating：让模型能对不同工艺区域形成不同表征，但复杂度远低于 MoE
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.Sigmoid(),
        )

        # 任务头
        self.strain_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.tensile_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.yield_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        # 多任务不确定性权重（训练时使用），初值为 0 => 权重约为 1
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        h = self.backbone(x)
        g = self.gate(x)
        h = h * (0.7 + 0.3 * g)  # 温和调制，避免 gating 过强导致不稳定

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 比当前基线再略小一点 lr，配合可学习任务权重更稳
    return torch.optim.Adam(model.parameters(), lr=6e-4, weight_decay=8e-5)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 固定先验权重：仍稍偏向强度任务，但比之前更温和，减少 trade-off
    fixed_task_weights = torch.tensor(
        [1.0, 1.25, 1.15], device=X_train.device, dtype=X_train.dtype
    )

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)  # [B, 3]

        # 多任务不确定性加权（数值稳定版）
        log_vars = model.log_vars.clamp(-2.0, 2.0)
        precision = torch.exp(-log_vars).clamp(0.135, 7.39)  # exp(-2)~exp(2)

        task_loss = per_elem.mean(dim=0)  # [3]
        uncertainty_loss = (fixed_task_weights * precision * task_loss + 0.5 * log_vars).sum()

        # 轻量 hard-example 重加权：比上一版更保守，避免被个别外推点牵着走
        with torch.no_grad():
            sample_err = (pred - yb).abs().mean(dim=1)
            weights = 1.0 + 0.35 * sample_err / (sample_err.mean() + 1e-6)
            weights = weights.clamp(1.0, 1.8)

        weighted_loss = ((per_elem * fixed_task_weights).mean(dim=1) * weights).mean()

        # 物理约束：抗拉强度通常不应低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量输出幅值正则，抑制小样本下极端预测
        pred_reg = (pred ** 2).mean()

        loss = (
            0.6 * uncertainty_loss +
            0.35 * weighted_loss +
            0.08 * order_penalty +
            1e-4 * pred_reg
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()