"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。
train.py 是固定 runner，不要修改它。

本版采取保守改进：
1) 小样本下缩小模型容量，降低过拟合
2) 使用共享干路 + 多任务头，缓解三目标 trade-off
3) 加入轻量输出关系约束：抗拉强度 >= 屈服强度
4) 对强度任务做温和加权，针对当前 tensile / yield 误差偏大
5) 使用 Huber 风格平滑损失提升对离群点/突变点的稳健性
6) 加入梯度裁剪，保证数值稳定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 小样本下采用更小网络；共享表示后接任务头，减少三任务互相干扰
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

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

    def forward(self, x):
        h = self.backbone(x)
        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 相比基线略减小 lr，并加入轻微 weight_decay，更适合 29 条小样本
    return torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n)

    # Huber 比 MSE 对少量突变点更稳健
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 当前强度误差远大于应变，给予温和再平衡，避免过度牺牲应变
    task_weights = torch.tensor([1.0, 1.35, 1.2], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 基础多任务损失
        per_elem = criterion(pred, yb)  # [B, 3]
        base_loss = (per_elem * task_weights).mean()

        # 轻量 hard-example 重加权：聚焦大残差样本，但限制上界防止不稳定
        with torch.no_grad():
            sample_err = (pred - yb).abs().mean(dim=1)
            weights = 1.0 + 0.5 * sample_err / (sample_err.mean() + 1e-6)
            weights = weights.clamp(1.0, 2.0)

        weighted_loss = ((per_elem * task_weights).mean(dim=1) * weights).mean()

        # 物理约束：抗拉强度通常不应低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        loss = 0.5 * base_loss + 0.5 * weighted_loss + 0.1 * order_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()