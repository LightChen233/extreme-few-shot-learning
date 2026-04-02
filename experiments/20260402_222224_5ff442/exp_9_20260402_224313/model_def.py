"""
模型定义 — LLM 唯一可修改的文件。
本版遵循“保守改进原则”：

基于历史结果，当前基线已相对较优（401 MSE），因此只做小幅、定向调整：
1) 保留小型共享干路 + 轻量门控，不重写范式
2) 将门控从“逐通道缩放”微调为“残差式局部修正”，减小对主干的扰动
3) 轻度加入二次特征投影分支，帮助学习 440/24、470/12 这类非线性/峰值型响应
4) 保留通用物理约束：抗拉强度 >= 屈服强度
5) 用更稳健的 Huber + 极轻量 hard-example 重加权（有上界，基于 detach）
6) AdamW + 梯度裁剪，保证小样本训练稳定

注意：
- 数据仅 29 条，严控模型容量
- 不使用高阶梯度、不对输入求导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 主干：保持小模型，避免过拟合
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
        )

        # 轻量局部分支：对输入做一次额外投影，帮助拟合局部非线性区间
        # 容量很小，避免引入高方差
        self.local_branch = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 12),
        )

        # 轻量门控：只控制 local_branch 的注入强度
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
        local = self.local_branch(x)
        g = self.gate(x)

        # 残差式局部修正，比直接缩放主干更稳健
        h = h + 0.35 * g * local

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 在当前较优基线附近仅做轻微微调
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.2)

    # 保持轻度偏向强度，但进一步压低 trade-off 风险
    task_weights = torch.tensor([1.0, 1.15, 1.10], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        per_elem = criterion(pred, yb)  # [B, 3]

        # 极轻量难例重加权：只让大残差样本多一点关注，防止被极端点主导
        with torch.no_grad():
            abs_res = torch.abs(pred.detach() - yb)  # [B, 3]
            sample_score = abs_res.mean(dim=1, keepdim=True)  # [B, 1]
            norm_score = sample_score / (sample_score.mean() + 1e-6)
            sample_weights = torch.clamp(0.85 + 0.35 * norm_score, 0.85, 1.35)

        weighted_elem = per_elem * task_weights * sample_weights
        base_loss = weighted_elem.mean()

        # 通用物理约束：抗拉强度通常不应低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量输出耦合：抗拉/屈服残差不宜完全背离
        tensile_res = pred[:, 1] - yb[:, 1]
        yield_res = pred[:, 2] - yb[:, 2]
        coupling_penalty = F.smooth_l1_loss(
            yield_res, tensile_res, reduction="mean", beta=8.0
        )

        loss = base_loss + 0.08 * order_penalty + 0.02 * coupling_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()