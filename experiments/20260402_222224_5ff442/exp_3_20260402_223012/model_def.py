"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本次改动遵循保守原则，在当前较优基线上做小幅增强：
1) 保留小模型 + 多任务头，避免小样本过拟合
2) 在共享干路上加入轻量残差块与少量 dropout，提高外推稳健性
3) 保留 Huber 损失、hard-example 重加权、梯度裁剪
4) 增加温和的输出间物理约束：
   - 抗拉强度 >= 屈服强度
5) 增加“边界外推不确定性”感知：
   - 通过输入与训练集中心的标准化距离，对远离训练中心的样本略增权
   - 重点缓解 440/24、470/12 这类外推点误差
6) 使用更稳健的 AdamW + 轻量 scheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.08):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return self.act(x + h)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.shared_res = ResidualBlock(16, dropout=0.08)

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

        # 用于在 train_step 中缓存训练集统计量，做轻量“外推距离”重加权
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))
        self._stats_initialized = False

    def forward(self, x):
        h = self.stem(x)
        h = self.shared_res(h)
        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    return optimizer


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    # 缓存训练集统计量，用于外推距离感知重加权
    if (not getattr(model, "_stats_initialized", False)) or model.x_mean.shape[0] != X_train.shape[1]:
        with torch.no_grad():
            model.x_mean.copy_(X_train.mean(dim=0))
            model.x_std.copy_(X_train.std(dim=0).clamp_min(1e-6))
            model._stats_initialized = True

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 当前 tensile / yield 仍是主要误差来源，保持温和再平衡
    task_weights = torch.tensor([1.0, 1.35, 1.2], device=X_train.device, dtype=X_train.dtype)

    # 轻量调度：每个 epoch 调一次即可
    if not hasattr(model, "_scheduler"):
        model._scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.92)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)  # [B, 3]

        # 基础多任务损失
        base_per_sample = (per_elem * task_weights).mean(dim=1)
        base_loss = base_per_sample.mean()

        with torch.no_grad():
            # hard-example 重加权
            sample_err = (pred - yb).abs().mean(dim=1)
            hard_w = 1.0 + 0.45 * sample_err / (sample_err.mean() + 1e-6)
            hard_w = hard_w.clamp(1.0, 2.0)

            # 外推/边界距离重加权：距离训练中心越远，略增权
            z = (xb - model.x_mean) / model.x_std
            dist = torch.sqrt((z ** 2).mean(dim=1) + 1e-8)
            dist_w = 1.0 + 0.20 * dist / (dist.mean() + 1e-6)
            dist_w = dist_w.clamp(1.0, 1.5)

            sample_w = (hard_w * dist_w).clamp(1.0, 2.2)

        weighted_loss = (base_per_sample * sample_w).mean()

        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 物理约束：抗拉强度通常不低于屈服强度
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        loss = 0.45 * base_loss + 0.55 * weighted_loss + 0.08 * order_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

    model._scheduler.step()