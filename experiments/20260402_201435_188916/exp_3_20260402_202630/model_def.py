"""
模型定义 — 可修改文件
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

设计思路（针对 29 条小样本 + 外推点误差大）：
1) 降低容量，避免深宽 MLP 过拟合；
2) 使用共享干路 + 多任务头，显式建模三目标相关性；
3) 使用软混合专家（MoE），应对 460/470℃、12/24h 一类可能存在突变/过时效分段规律；
4) 训练时加入：
   - 多任务稳健损失（SmoothL1）
   - 基于残差的 hard example reweight
   - 物理关系约束：UTS >= YS
   - 输出平滑/局部 Lipschitz 正则：减小边界外推振荡
5) 使用 AdamW + 小权重衰减，提升小样本泛化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_dim, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = 3
        hidden = 24

        # 轻量 gating：适合小样本；用温度/时间及其组合诱导分段
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, self.num_experts)
        )

        self.experts = nn.ModuleList([Expert(input_dim, hidden=hidden) for _ in range(self.num_experts)])

        # 共享表示后接多任务头
        self.shared = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.Tanh(),
        )

        self.head_strain = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.head_tensile = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.head_yield = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        # 可学习任务权重（homoscedastic uncertainty 风格）
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [B, E]

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(x).unsqueeze(1))  # [B,1,H]
        expert_outs = torch.cat(expert_outs, dim=1)  # [B,E,H]

        mixed = torch.sum(gate_w.unsqueeze(-1) * expert_outs, dim=1)  # [B,H]
        h = self.shared(mixed)

        strain = self.head_strain(h)
        tensile = self.head_tensile(h)
        yield_s = self.head_yield(h)

        return torch.cat([strain, tensile, yield_s], dim=1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)


def _robust_multitask_loss(model, pred, target):
    """
    多任务稳健损失：
    - 逐目标 SmoothL1，降低异常点对整体训练的破坏
    - 任务不确定性加权，防止 tensile 主导
    - hard example reweight，聚焦突变/外推难点
    """
    # [B,3]
    per_elem = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)

    # hard example 权重：对样本整体残差大的点提高权重，但限制幅度避免不稳定
    with torch.no_grad():
        sample_err = per_elem.mean(dim=1)  # [B]
        norm = sample_err / (sample_err.mean() + 1e-6)
        hard_w = torch.clamp(0.7 + norm, 0.7, 2.5)  # [B]

    weighted = per_elem * hard_w.unsqueeze(1)

    # 任务不确定性加权
    task_losses = weighted.mean(dim=0)  # [3]
    loss = 0.0
    for k in range(3):
        precision = torch.exp(-model.log_vars[k])
        loss = loss + precision * task_losses[k] + model.log_vars[k]
    return loss, task_losses


def _physics_constraint_loss(pred):
    """
    通用物理软约束：
    抗拉强度 >= 屈服强度
    """
    tensile = pred[:, 1]
    yield_s = pred[:, 2]
    order_penalty = F.relu(yield_s - tensile).mean()
    return order_penalty


def _smoothness_loss(model, x):
    """
    局部平滑正则：
    对输入加微小扰动，约束输出变化不过度，缓解边界外推振荡。
    """
    if x.size(0) == 0:
        return torch.tensor(0.0, device=x.device)

    noise = 0.01 * torch.randn_like(x)
    x_aug = x + noise
    pred1 = model(x)
    pred2 = model(x_aug)
    return ((pred1 - pred2) ** 2).mean()


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)

    # 小样本下略小 batch，让难例重加权更有效
    batch_size = min(batch_size, 8)
    perm = torch.randperm(n, device=X_train.device if X_train.is_cuda else None)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad()
        pred = model(xb)

        data_loss, _ = _robust_multitask_loss(model, pred, yb)
        phys_loss = _physics_constraint_loss(pred)
        smooth_loss = _smoothness_loss(model, xb)

        # 权重控制：以数据拟合为主，物理与平滑为辅
        loss = data_loss + 2.0 * phys_loss + 0.2 * smooth_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()