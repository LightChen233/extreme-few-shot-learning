"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本次改动遵循“极小幅度、可回退、以稳为主”的原则：
1) 保留小型共享主干 + 多任务头，不重写整体范式
2) 去掉上版额外 gating，避免 29 条样本下局部门控带来的高方差
3) 保留并轻微强化通用物理约束：抗拉强度 >= 屈服强度
4) 增加一个非常轻量的“输出增量参数化”：
   令 tensile = yield + softplus(delta)，从结构上保证顺序关系，减少强度头互相打架
5) 使用更稳的 Huber 损失 + 极温和的难例重加权（有上界，且基于 detach）
   仅让模型略微关注 470/12、440/24 等大误差点，避免像历史版本那样被异常点牵着走
6) 继续使用 AdamW + 梯度裁剪，确保数值稳定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 小样本下尽量控制容量，使用平滑激活降低训练抖动
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.SiLU(),
            nn.Linear(20, 10),
            nn.SiLU(),
        )

        self.strain_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

        # 强度采用“屈服 + 增量”的参数化，硬性保证 tensile >= yield
        self.yield_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )
        self.delta_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        h = self.backbone(x)

        strain = self.strain_head(h)
        yld = self.yield_head(h)
        delta = F.softplus(self.delta_head(h))  # >= 0
        tensile = yld + delta

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 在当前已较优基线附近做极小调整：略降 lr，略增正则
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.2e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    # 稍小 beta，让中等误差样本更快进入近似 L1 区域，提升对外推点鲁棒性
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.2)

    # 保守任务权重：继续略偏向两种强度，但避免明显 trade-off
    task_weights = torch.tensor([1.0, 1.15, 1.10], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        per_elem = criterion(pred, yb)  # [B, 3]

        # 极温和 hard-example 重加权：只按样本级平均残差微调，且严格上界
        # 历史上激进重加权会退化，这里仅做 1.0~1.6 的轻微关注
        with torch.no_grad():
            abs_res = (pred - yb).abs().mean(dim=1, keepdim=True)  # [B,1]
            scale = abs_res / (abs_res.mean() + 1e-6)
            sample_weights = torch.clamp(1.0 + 0.25 * scale, min=1.0, max=1.6)

        weighted = per_elem * task_weights * sample_weights
        base_loss = weighted.mean()

        # 冗余但稳妥的顺序约束（虽结构上已满足，仍保留极小权重，防数值边界问题）
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 强度残差一致性：两者通常共同受时效状态影响，但权重要很小，避免绑死
        tensile_res = pred[:, 1] - yb[:, 1]
        yield_res = pred[:, 2] - yb[:, 2]
        coupling_penalty = F.smooth_l1_loss(
            yield_res, tensile_res, reduction="mean", beta=12.0
        )

        loss = base_loss + 0.02 * order_penalty + 0.015 * coupling_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()