"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本次改动遵循“保守改进原则”：
1) 保持小型共享干路 + 多任务头，不重写整体范式
2) 鉴于历史多次大改均退化，本次只做极小幅度调整：
   - 去掉上版对局部区间可能过敏的输入门控，减少小样本下的方差
   - 保留少量共享层归一化，提升不同输出尺度下训练稳定性
   - 对 hard example 仅做“有上界”的轻量重加权，重点关注 470/12、440/24 这类大误差点，但避免被异常点绑架
3) 保留通用物理约束：抗拉强度 >= 屈服强度
4) 保留抗拉/屈服残差一致性的弱耦合
5) 使用 AdamW + 梯度裁剪，保证数值稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 更保守的小模型：略窄，但加入 LayerNorm 稳定不同尺度特征
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.LayerNorm(20),
            nn.Linear(20, 10),
            nn.ReLU(),
        )

        self.strain_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.tensile_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.yield_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        h = self.backbone(x)
        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 极小幅微调：略降学习率，减小小样本振荡
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.5)
    task_weights = torch.tensor([1.0, 1.15, 1.10], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        per_elem = criterion(pred, yb)  # [B, 3]

        # 轻量 hard-example 重加权：
        # 基于样本当前平均残差，强调大误差样本，但上界严格限制，避免训练发散
        with torch.no_grad():
            abs_res = (pred - yb).abs().mean(dim=1, keepdim=True)  # [B,1]
            sample_weight = 1.0 + 0.25 * abs_res / (abs_res.mean() + 1e-6)
            sample_weight = sample_weight.clamp(1.0, 1.8)

        weighted_loss = per_elem * task_weights.unsqueeze(0) * sample_weight
        base_loss = weighted_loss.mean()

        # 通用物理约束：抗拉强度通常不应低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 抗拉/屈服误差弱耦合，避免两者偏差方向完全脱钩
        tensile_res = pred[:, 1] - yb[:, 1]
        yield_res = pred[:, 2] - yb[:, 2]
        coupling_penalty = F.smooth_l1_loss(
            yield_res, tensile_res, reduction="mean", beta=10.0
        )

        loss = base_loss + 0.08 * order_penalty + 0.02 * coupling_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()