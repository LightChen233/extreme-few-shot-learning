"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。
train.py 是固定 runner，不要修改它。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 小样本（29条）下采用轻量多任务共享干路 + 独立输出头
        # 相比原始单头 MLP，参数量仍较小，但更利于三个目标的差异化建模
        hidden = 32
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.head_strain = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
        self.head_tensile = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
        self.head_yield = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

        # 多任务不确定性加权参数，训练中会做 clamp 防止数值不稳定
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        h = self.backbone(x)
        strain = self.head_strain(h)
        tensile = self.head_tensile(h)
        yld = self.head_yield(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练：稳健多任务损失 + 难例重加权 + 物理关系软约束"""
    model.train()
    n = len(X_train)
    if n == 0:
        return

    # 小样本下适当减小 batch，提高梯度更新频率
    batch_size = min(6, n)

    # 预先计算各任务尺度，避免高量纲目标主导训练
    with torch.no_grad():
        y_scale = y_train.std(dim=0).clamp_min(1.0)

    perm = torch.randperm(n)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 标准化残差，缓解三任务量纲差异
        err = (pred - yb) / y_scale

        # 使用 smooth_l1 增强对异常点鲁棒性
        per_elem = F.smooth_l1_loss(err, torch.zeros_like(err), reduction="none")

        # 难例动态重加权：基于 detach 的残差，且有上界，避免爆炸
        with torch.no_grad():
            abs_err = err.abs()
            sample_hard = abs_err.mean(dim=1, keepdim=True)
            weights = 1.0 + 1.5 * torch.tanh(sample_hard)
            weights = weights.clamp(1.0, 2.5)

        weighted_task_loss = (per_elem * weights).mean(dim=0)

        # 多任务不确定性加权（带 clamp）
        log_vars = model.log_vars.clamp(-2.0, 2.0)
        precisions = torch.exp(-log_vars).clamp(0.135, 7.39)
        data_loss = (precisions * weighted_task_loss + log_vars).sum()

        # 物理软约束：通常抗拉强度 >= 屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_loss = F.relu(yield_pred - tensile_pred).mean()

        # 物理软约束：应变、强度不应为明显负值
        nonneg_loss = (
            F.relu(-pred[:, 0]).mean() +
            F.relu(-pred[:, 1]).mean() +
            F.relu(-pred[:, 2]).mean()
        )

        loss = data_loss + 0.08 * order_loss + 0.02 * nonneg_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()