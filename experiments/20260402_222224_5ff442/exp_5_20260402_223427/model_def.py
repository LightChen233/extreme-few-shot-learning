"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本次改动遵循“保守改进”原则：
1) 保持小模型与多任务头，不重写整体架构
2) 在共享表示上加入极轻量 gate（按样本自适应缩放表示），提升对局部工艺区间差异的表达
3) 保留 SmoothL1 + hard example reweight，但降低激进程度，避免过拟合到少数异常点
4) 加入温和的输出不确定性加权（可学习 log_vars，并做 clamp），缓解任务间 trade-off
5) 保留 tensile >= yield 的物理约束
6) 继续使用梯度裁剪，保证稳定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 小样本场景下保持低容量
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # 轻量 gating：不引入显著复杂度，但允许不同工艺区间有不同表征强度
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Sigmoid()
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

        # 多任务不确定性权重参数；训练时会 clamp，避免数值不稳定
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        h = self.backbone(x)
        g = self.gate(x)
        # 残差式门控，避免 gate 过强抑制特征
        h = h * (0.7 + 0.6 * g)

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 在当前较优基线附近做小幅微调：略小 lr、轻微正则
    return torch.optim.Adam(model.parameters(), lr=7e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 保持对强度任务的轻度关注，但比上一版更保守
    static_task_weights = torch.tensor(
        [1.0, 1.25, 1.15],
        device=X_train.device,
        dtype=X_train.dtype
    )

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)  # [B, 3]

        # 可学习不确定性加权，严格 clamp 防止爆炸
        log_vars = torch.clamp(model.log_vars, min=-2.0, max=2.0)
        inv_vars = torch.exp(-log_vars).clamp(0.135, 7.39)  # 对应 exp([-2,2])

        # 静态权重 × 不确定性权重
        task_scale = static_task_weights * inv_vars
        base_per_sample = (per_elem * task_scale).mean(dim=1)
        unc_reg = 0.5 * log_vars.mean()

        # 更温和的 hard-example 重加权，避免过度追逐外推异常点
        with torch.no_grad():
            sample_err = (pred - yb).abs().mean(dim=1)
            norm_err = sample_err / (sample_err.mean() + 1e-6)
            weights = 1.0 + 0.35 * norm_err
            weights = weights.clamp(1.0, 1.6)

        weighted_loss = (base_per_sample * weights).mean()
        base_loss = base_per_sample.mean()

        # 物理约束：抗拉强度通常不应低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        loss = 0.6 * base_loss + 0.4 * weighted_loss + 0.05 * order_penalty + unc_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()