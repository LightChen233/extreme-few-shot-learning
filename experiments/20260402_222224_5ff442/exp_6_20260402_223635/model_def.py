"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本次改动遵循“保守改进原则”：
1) 不重写整体范式，仍使用小型共享干路 + 多任务头
2) 去掉上次容易放大噪声的 hard-example 重加权，避免小样本下被个别异常点牵着走
3) 加入极轻量的温度/时间分段门控（piecewise gating），用于缓解 470℃/12h、440℃/24h 这类局部工艺区间误差
4) 保留通用物理约束：抗拉强度 >= 屈服强度
5) 使用更平滑的 Huber 损失 + 更温和任务权重，降低 trade-off 风险
6) 使用 AdamW + 梯度裁剪，保证数值稳定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 主干保持小模型，防止 29 条样本过拟合
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
        )

        # 极轻量分段门控：
        # 用一个很小的 gating 网络对共享表示做缩放，
        # 让模型能在不同工艺区间学到略有差异的局部映射，
        # 但不引入完整 MoE 的高方差训练风险。
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.Sigmoid()
        )

        self.strain_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.tensile_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.yield_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        h = self.backbone(x)
        g = self.gate(x)
        # 残差式门控，避免 gate 过强抹掉主干信息
        h = h * (0.75 + 0.5 * g)

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 相比之前进一步偏保守：略小 lr，较轻 weight decay
    return torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=8e-5)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    # Huber 对小样本中的局部突变更稳健
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.5)

    # 保守任务权重：仍略偏向强度，但避免像之前那样过度拉偏
    task_weights = torch.tensor([1.0, 1.20, 1.10], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        per_elem = criterion(pred, yb)  # [B, 3]
        base_loss = (per_elem * task_weights).mean()

        # 通用物理约束：抗拉强度通常不应低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量输出耦合：抗拉和屈服的误差方向通常具有一定相关性，
        # 用一个很小的残差一致性约束，避免两者分叉过大
        tensile_res = pred[:, 1] - yb[:, 1]
        yield_res = pred[:, 2] - yb[:, 2]
        coupling_penalty = F.smooth_l1_loss(yield_res, tensile_res, reduction="mean", beta=10.0)

        loss = base_loss + 0.08 * order_penalty + 0.03 * coupling_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()