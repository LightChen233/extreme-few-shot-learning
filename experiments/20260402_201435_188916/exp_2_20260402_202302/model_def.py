"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。
train.py 是固定 runner，不要修改它。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 小样本场景：采用低容量共享干路 + 多任务头，避免过拟合
        hidden = 32
        trunk_out = 16

        self.input_norm = nn.LayerNorm(input_dim)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, trunk_out),
            nn.Tanh(),
        )

        # 三任务头：应变、抗拉、屈服
        self.head_strain = nn.Sequential(
            nn.Linear(trunk_out, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        self.head_tensile = nn.Sequential(
            nn.Linear(trunk_out, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        self.head_yield = nn.Sequential(
            nn.Linear(trunk_out, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

        # 可学习任务不确定性权重（多任务稳健加权）
        self.log_vars = nn.Parameter(torch.zeros(3))

        # 用于 train_step 中缓存训练统计量
        self._stats_ready = False
        self.register_buffer("_x_mean", torch.zeros(input_dim))
        self.register_buffer("_x_std", torch.ones(input_dim))
        self.register_buffer("_y_mean", torch.zeros(3))
        self.register_buffer("_y_std", torch.ones(3))

    def set_stats(self, X, y):
        with torch.no_grad():
            self._x_mean.copy_(X.mean(dim=0))
            self._x_std.copy_(X.std(dim=0).clamp_min(1e-6))
            self._y_mean.copy_(y.mean(dim=0))
            self._y_std.copy_(y.std(dim=0).clamp_min(1e-6))
            self._stats_ready = True

    def forward(self, x):
        x = self.input_norm(x)
        h = self.trunk(x)
        strain = self.head_strain(h)
        tensile = self.head_tensile(h)
        yld = self.head_yield(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """
    单 epoch 训练逻辑：
    1) 小样本下全量/近全量训练更稳定
    2) 使用标准化后的多任务不确定性加权损失
    3) 加入物理软约束：抗拉强度 >= 屈服强度
    4) 对困难样本做动态重加权，聚焦误差大的外推/突变点
    5) 使用输入梯度平滑正则，提升外推稳定性
    """
    device = X_train.device
    model.train()

    if not getattr(model, "_stats_ready", False):
        model.set_stats(X_train, y_train)

    # 小数据集：使用较大的 batch，减小梯度噪声
    batch_size = min(max(batch_size, 16), len(X_train))

    # 全局标准化统计
    x_mean = model._x_mean
    x_std = model._x_std
    y_mean = model._y_mean
    y_std = model._y_std

    # 简单学习率调度：训练早期快一些，后期更稳
    if not hasattr(model, "_step_count"):
        model._step_count = 0
    model._step_count += 1
    base_lr = 3e-3
    if model._step_count > 200:
        lr = 8e-4
    elif model._step_count > 120:
        lr = 1.5e-3
    else:
        lr = base_lr
    for g in optimizer.param_groups:
        g["lr"] = lr

    perm = torch.randperm(len(X_train), device=device)

    for i in range(0, len(X_train), batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx].detach().clone().requires_grad_(True)
        yb = y_train[idx]

        pred = model(xb)

        # 输出标准化，避免 tensile/yield 尺度主导损失
        pred_n = (pred - y_mean) / y_std
        yb_n = (yb - y_mean) / y_std

        # 样本级误差（3任务平均）
        per_sample_mse = ((pred_n - yb_n) ** 2).mean(dim=1).detach()

        # 难例重加权：强调大残差样本，但限制上界避免训练失稳
        weights = 1.0 + 1.5 * (per_sample_mse / (per_sample_mse.mean() + 1e-8))
        weights = weights.clamp(1.0, 3.0)

        # 多任务不确定性加权损失
        sq_err = (pred_n - yb_n) ** 2
        weighted_sq_err = weights.unsqueeze(1) * sq_err

        task_losses = weighted_sq_err.mean(dim=0)
        data_loss = 0.0
        for t in range(3):
            inv_var = torch.exp(-model.log_vars[t])
            data_loss = data_loss + inv_var * task_losses[t] + model.log_vars[t]

        # 物理软约束：抗拉强度 >= 屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 平滑正则：约束模型对输入的小扰动不过分敏感，改善外推
        pred_sum = pred_n.sum()
        grads = torch.autograd.grad(
            pred_sum, xb, create_graph=True, retain_graph=True
        )[0]
        smooth_penalty = (grads ** 2).mean()

        # 温和 L2 正则
        l2_penalty = torch.tensor(0.0, device=device)
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() > 1:
                l2_penalty = l2_penalty + (p ** 2).mean()

        loss = (
            data_loss
            + 0.20 * order_penalty
            + 0.01 * smooth_penalty
            + 1e-4 * l2_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()