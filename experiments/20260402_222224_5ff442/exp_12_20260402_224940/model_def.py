"""
模型定义 — LLM 唯一可修改的文件。
遵循保守改进原则：

基于历史结果，当前基线已相对稳定，避免大改架构。
本次仅做极小幅度、可解释的调整：
1) 保留小型共享干路 + 轻量 gating，多任务头不变
2) 去掉上次容易引发 trade-off 的“输出残差耦合惩罚”，避免把屈服误差错误绑定到抗拉误差
3) 维持通用物理约束：抗拉强度 >= 屈服强度
4) 采用更稳健的 Huber，并对强度任务做非常轻微的再平衡
5) 对难例做“有上界”的温和重加权（detach 后计算，clamp 到 [1, 2]），
   重点关注 470/12、440/24 这类突变/外推点，但避免像之前那样被极端点主导
6) 学习率再略降一点，提升小样本稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 小样本下保持低容量
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
        )

        # 轻量分段门控，保留对局部工艺区间的适应性
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

        # 残差式门控，抑制 gate 过强带来的不稳定
        h = h * (0.8 + 0.4 * g)

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 比当前基线更保守一点：略降 lr，保持轻正则
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.5)

    # 轻微偏向强度任务，但避免过度牺牲应变
    task_weights = torch.tensor(
        [1.0, 1.15, 1.10],
        device=X_train.device,
        dtype=X_train.dtype
    )

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        per_elem = criterion(pred, yb)  # [B, 3]

        # 温和 hard-example reweighting：
        # 基于当前 batch 的样本平均残差构造权重，使用 detach 防止不稳定，
        # 且严格限制上界，避免被个别异常点牵着走。
        with torch.no_grad():
            sample_err = (pred - yb).abs().mean(dim=1)  # [B]
            norm_err = sample_err / (sample_err.mean() + 1e-6)
            sample_w = torch.clamp(0.75 + 0.5 * norm_err, 1.0, 2.0)  # [B]

        weighted_loss = per_elem * task_weights.unsqueeze(0) * sample_w.unsqueeze(1)
        base_loss = weighted_loss.mean()

        # 通用物理约束：抗拉强度通常不应低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        loss = base_loss + 0.10 * order_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()