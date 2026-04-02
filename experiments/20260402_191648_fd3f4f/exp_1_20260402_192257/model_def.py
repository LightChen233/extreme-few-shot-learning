"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。
train.py 是固定 runner，不要修改它。

设计思路（针对 29 条小样本 + 局部突变误差大）：
1) 小样本下控制模型容量，避免深宽 MLP 过拟合。
2) 使用共享干网络 + 多任务头，兼顾三输出相关性与差异性。
3) 使用轻量 MoE（混合专家）：让模型能对不同温度/时间区间进行软分段，
   以拟合 460/470°C、12h 等可能的相变/过时效突变区域。
4) 训练采用：
   - 相对尺度归一化后的加权 Huber 损失（比 MSE 对异常点更稳）
   - hard example 动态重加权（突出大残差样本）
   - 输出关系软约束：抗拉强度 >= 屈服强度
   - 温和 L2 正则
5) 为兼容固定 runner，所有训练状态保存在 model 对象内部。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # 小样本场景下，先做一层较窄的共享表征
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
        )

        # 轻量 gating 网络：对不同工艺区间进行软路由
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )

        # 3 个小专家，容量受控，避免过拟合
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(16, 12),
                nn.Tanh(),
                nn.Linear(12, 8),
                nn.Tanh()
            )
            for _ in range(3)
        ])

        # 多任务头：共享专家融合后分别预测
        self.head_strain = nn.Sequential(
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.head_tensile = nn.Sequential(
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.head_yield = nn.Sequential(
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        # 训练态缓存
        self._ema_abs_err = None
        self._target_scale = None
        self._epoch = 0

    def forward(self, x):
        h = self.trunk(x)                            # [B, 16]
        g_logits = self.gate(x)                     # [B, 3]
        g = torch.softmax(g_logits, dim=-1)         # [B, 3]

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(h).unsqueeze(1))   # [B,1,8]
        expert_outs = torch.cat(expert_outs, dim=1)      # [B,3,8]

        fused = torch.sum(expert_outs * g.unsqueeze(-1), dim=1)  # [B,8]

        strain = self.head_strain(fused)
        tensile = self.head_tensile(fused)
        yld = self.head_yield(fused)

        return torch.cat([strain, tensile, yld], dim=1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """
    单 epoch 训练：
    - 初始阶段估计各任务尺度
    - 采用尺度归一化 Huber + 动态难例加权
    - 加入 tensile >= yield 软约束
    """
    model.train()
    device = X_train.device

    # 目标尺度：防止三个任务量纲差异导致 tensile 主导训练
    if model._target_scale is None:
        scale = y_train.std(dim=0, unbiased=False).detach()
        scale = torch.clamp(scale, min=1.0)
        model._target_scale = scale.to(device)

    n = len(X_train)
    if n == 0:
        return

    # 用当前模型全量前向，估计每个样本的难度，做动态采样/重加权
    with torch.no_grad():
        pred_all = model(X_train)
        abs_err = (pred_all - y_train).abs() / model._target_scale.unsqueeze(0)  # [N,3]
        sample_err = abs_err.mean(dim=1)  # [N]

        if model._ema_abs_err is None:
            model._ema_abs_err = sample_err.clone()
        else:
            model._ema_abs_err = 0.7 * model._ema_abs_err + 0.3 * sample_err

        # 难例权重：不过分极端，避免小样本被个别噪声样本主导
        hard_w = 1.0 + 1.5 * (model._ema_abs_err / (model._ema_abs_err.mean() + 1e-8))
        hard_w = torch.clamp(hard_w, 0.8, 3.0)

        # 按权重采样，提升突变区样本曝光率
        probs = hard_w / hard_w.sum()
        num_draw = max(n, batch_size * ((n + batch_size - 1) // batch_size))
        sampled_idx = torch.multinomial(probs, num_samples=num_draw, replacement=True)

    # 每个 epoch 打乱采样结果
    perm = torch.randperm(len(sampled_idx), device=device)
    sampled_idx = sampled_idx[perm]

    # 轻微学习率衰减
    model._epoch += 1
    base_lr = 2e-3
    lr = base_lr * (0.97 ** max(model._epoch - 1, 0))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    for i in range(0, len(sampled_idx), batch_size):
        idx = sampled_idx[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 尺度归一化残差
        diff = (pred - yb) / model._target_scale.unsqueeze(0)

        # Huber（smooth_l1）比 MSE 更稳健，同时保留对大误差样本的梯度
        per_elem = F.smooth_l1_loss(
            diff, torch.zeros_like(diff), reduction="none", beta=1.0
        )  # [B,3]

        # 任务权重：略提升 tensile/yield 的学习力度，但不过强
        task_w = torch.tensor([1.1, 1.25, 1.2], device=device).view(1, 3)
        per_sample = (per_elem * task_w).mean(dim=1)

        # batch 内 hard example 权重
        with torch.no_grad():
            raw_abs = diff.abs().mean(dim=1)
            batch_w = 1.0 + 1.2 * raw_abs
            batch_w = torch.clamp(batch_w, 1.0, 3.0)

        data_loss = (per_sample * batch_w).mean()

        # 物理软约束：抗拉强度应不低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        phys_loss = F.relu(yield_pred - tensile_pred).mean()

        # gating 熵正则：避免过早塌缩到单一专家，但系数很小
        gate_logits = model.gate(xb)
        gate_prob = torch.softmax(gate_logits, dim=-1)
        gate_entropy = -(gate_prob * torch.log(gate_prob + 1e-8)).sum(dim=1).mean()

        loss = data_loss + 0.15 * phys_loss - 0.005 * gate_entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()