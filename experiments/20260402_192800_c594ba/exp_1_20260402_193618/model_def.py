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
        self.input_dim = input_dim

        # 小样本场景：使用低容量共享骨干 + 多任务头，避免过拟合
        hidden = 24

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # 两个轻量专家：用于拟合不同工艺区间（如常规时效 / 过时效或突变区）
        self.expert1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.expert2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # 轻量 gating，避免复杂 MoE 在 29 条数据上失控
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )

        # 多任务头
        self.head_strain = nn.Linear(hidden, 1)
        self.head_tensile = nn.Linear(hidden, 1)
        self.head_yield = nn.Linear(hidden, 1)

        # 为极小样本提供一个可学习的全局基线
        self.base = nn.Parameter(torch.zeros(1, 3))

    def forward(self, x):
        h = self.backbone(x)

        e1 = self.expert1(h)
        e2 = self.expert2(h)

        gate_logits = self.gate(x)
        gate_w = torch.softmax(gate_logits, dim=-1)

        mixed = gate_w[:, :1] * e1 + gate_w[:, 1:2] * e2

        strain = self.head_strain(mixed)
        tensile = self.head_tensile(mixed)
        yld = self.head_yield(mixed)

        out = torch.cat([strain, tensile, yld], dim=-1) + self.base
        return out


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """
    单 epoch 训练逻辑：
    1) 标准化多任务残差，避免抗拉强度量纲主导训练
    2) 使用 hard-example 重加权，重点关注 440/1、470/12 一类突变难例
    3) 加入物理软约束：抗拉强度 >= 屈服强度
    4) 轻量 L2 输出约束，提升小样本稳定性
    """
    model.train()
    n = len(X_train)
    if n == 0:
        return

    batch_size = min(batch_size, n)

    # 基于训练集标签做任务尺度归一化，避免高量纲目标支配 MSE
    with torch.no_grad():
        y_std = y_train.std(dim=0, unbiased=False).clamp_min(1.0)
        y_mean = y_train.mean(dim=0)

    perm = torch.randperm(n, device=X_train.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 任务归一化残差
        err = (pred - yb) / y_std

        # 基础多任务损失：SmoothL1 比纯 MSE 更稳健
        per_sample_task = F.smooth_l1_loss(
            pred / y_std, yb / y_std, reduction="none"
        ).mean(dim=1)

        # Hard example mining:
        # 用当前残差大小自适应加权，让模型更关注突变样本而不是仅拟合平均趋势
        with torch.no_grad():
            hard_score = err.abs().mean(dim=1)
            weights = 1.0 + 1.5 * hard_score
            weights = weights / weights.mean().clamp_min(1e-6)

        data_loss = (per_sample_task * weights).mean()

        # 物理约束：抗拉强度应不小于屈服强度
        phys_loss = F.relu(pred[:, 2] - pred[:, 1]).mean()

        # 轻微输出正则：抑制小样本下异常大偏移
        reg_loss = ((pred - y_mean) / y_std).pow(2).mean()

        loss = data_loss + 0.2 * phys_loss + 0.01 * reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()