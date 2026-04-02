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
        # 小样本场景：保持容量克制，但比基线略强
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # 多任务头：共享主干 + 分任务输出，适合三目标相关但不完全一致
        self.head_strain = nn.Linear(32, 1)
        self.head_tensile = nn.Linear(32, 1)
        self.head_yield_ = nn.Linear(32, 1)

        # 可学习多任务权重（不确定性加权），训练时做 clamp 保证稳定
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        h = self.backbone(x)
        strain = self.head_strain(h)
        tensile = self.head_tensile(h)
        yield_strength = self.head_yield_(h)
        return torch.cat([strain, tensile, yield_strength], dim=1)


def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练：稳健多任务损失 + 轻度难例重加权 + 输出物理约束"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 基础逐任务 MSE
        per_dim_mse = (pred - yb) ** 2  # [B, 3]

        # 难例重加权：基于样本整体残差，但做 detach 且限制上界，避免不稳定
        with torch.no_grad():
            sample_err = (pred.detach() - yb).abs().mean(dim=1, keepdim=True)  # [B,1]
            norm_err = sample_err / (sample_err.mean() + 1e-6)
            sample_w = torch.clamp(0.75 + 0.75 * norm_err, 0.75, 2.0)

        weighted_mse = (per_dim_mse * sample_w).mean(dim=0)  # [3]

        # 多任务不确定性加权，带 clamp 保证数值稳定
        log_vars = torch.clamp(model.log_vars, min=-2.0, max=2.0)
        inv_vars = torch.exp(-log_vars)
        task_loss = (inv_vars * weighted_mse + log_vars).sum()

        # 物理软约束：通常抗拉强度 >= 屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).pow(2).mean()

        # 轻度输出范围正则：应变/强度不应出现明显负值
        nonneg_penalty = F.relu(-pred).pow(2).mean()

        loss = task_loss + 0.2 * order_penalty + 0.05 * nonneg_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()