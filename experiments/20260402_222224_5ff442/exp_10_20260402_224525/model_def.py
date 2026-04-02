"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本版遵循“极小幅度保守改进”：
1) 保持小型共享干路 + 多任务头，不重写整体范式
2) 保留轻量 gating，但进一步减小其自由度，降低小样本过拟合风险
3) 仅加入“有上界”的难例重加权，重点关注 470/12、440/24 一类突变/外推难点
4) 保留通用物理约束：抗拉强度 >= 屈服强度
5) 维持平滑 Huber 损失与梯度裁剪，确保训练稳定
6) 只做轻微超参微调，避免历史上大改动导致回退
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 小样本下保持低容量主干
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
        )

        # 更保守的轻量门控：容量比上一版略小
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 12),
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

        # 残差式弱门控，幅度收窄，减少被局部噪声带偏
        h = h * (0.85 + 0.30 * g)

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 仅做轻微微调：略小 lr，略强一点点正则，保持稳定
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.5)

    # 保守任务权重：略偏向两项强度，但不明显拉偏
    task_weights = torch.tensor([1.0, 1.15, 1.10], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)  # [B, 3]

        # 动态难例重加权：基于 detach 残差，且严格限制上界，避免训练爆炸
        with torch.no_grad():
            abs_res = (pred - yb).abs()  # [B, 3]
            # 先按任务尺度求每个样本的相对难度
            sample_hardness = (abs_res * task_weights).mean(dim=1, keepdim=True)
            sample_mean = sample_hardness.mean()
            sample_std = sample_hardness.std(unbiased=False)

            if torch.isfinite(sample_std) and sample_std > 1e-8:
                z = (sample_hardness - sample_mean) / (sample_std + 1e-6)
                sample_weight = 1.0 + 0.35 * torch.clamp(z, min=0.0, max=2.0)
            else:
                sample_weight = torch.ones_like(sample_hardness)

            sample_weight = sample_weight.clamp(1.0, 1.7)

        weighted_elem = per_elem * task_weights
        base_loss = (weighted_elem * sample_weight).mean()

        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 通用物理约束：抗拉强度通常不应小于屈服强度
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量耦合：两种强度残差不宜完全背离
        tensile_res = pred[:, 1] - yb[:, 1]
        yield_res = pred[:, 2] - yb[:, 2]
        coupling_penalty = F.smooth_l1_loss(
            yield_res, tensile_res, reduction="mean", beta=10.0
        )

        # 很弱的输出平滑正则，抑制头部过大幅值，减少外推抖动
        pred_scale_penalty = (pred[:, 1:].pow(2).mean()) * 1e-5

        loss = (
            base_loss
            + 0.08 * order_penalty
            + 0.02 * coupling_penalty
            + pred_scale_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()