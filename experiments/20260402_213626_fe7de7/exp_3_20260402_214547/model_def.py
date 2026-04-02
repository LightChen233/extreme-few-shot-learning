"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本版在已有较好基线（MSE≈513）上做“保守改进”：
1) 保持小模型与多任务头，避免小样本过拟合
2) 在共享干路上加入轻量残差式表示，提升温度-时间交互拟合能力
3) 用可学习的不确定性多任务加权，但做数值 clamp，保证稳定
4) 保留 hard-example 重加权，并略微降低其激进程度
5) 保留通用物理约束：屈服强度 <= 抗拉强度
6) 增加温和的一致性正则（同一 batch 内预测方差不过度塌缩/爆炸并不直接约束输入梯度）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.06):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        return F.relu(x + h)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = 32

        # 小样本下保持轻量；先投影，再做轻量残差提取交互
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )
        self.block1 = ResidualBlock(hidden, dropout=0.06)
        self.block2 = ResidualBlock(hidden, dropout=0.06)

        # 共享后再做轻量任务专属变换
        self.shared_dropout = nn.Dropout(0.05)

        self.strain_head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.tensile_head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.yield_head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # 多任务不确定性参数；train_step 中会 clamp，避免数值爆炸
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        h = self.input_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.shared_dropout(h)

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 相比上一版仅做温和微调：略降 lr，略减 wd，适合当前已较优基线
    return torch.optim.Adam(model.parameters(), lr=6e-4, weight_decay=8e-5)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练逻辑"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_task_sq = (pred - yb) ** 2  # [B, 3]

        # 1) 温和 hard-example 重加权：继续关注外推/突变点，但避免过强放大
        with torch.no_grad():
            abs_res = (pred.detach() - yb).abs().mean(dim=1)  # [B]
            scale = abs_res.mean().clamp_min(1e-6)
            sample_w = 1.0 + 0.6 * (abs_res / scale)
            sample_w = sample_w.clamp(1.0, 2.2)

        weighted_task_mse = (per_task_sq * sample_w.unsqueeze(1)).mean(dim=0)  # [3]

        # 2) 多任务不确定性加权（数值稳定版）
        log_vars = torch.clamp(model.log_vars, min=-2.5, max=2.5)
        inv_vars = torch.exp(-log_vars).clamp(min=0.08, max=12.0)
        multi_task_loss = (inv_vars * weighted_task_mse + log_vars).sum()

        # 3) 物理约束：屈服强度不应高于抗拉强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        phys_loss = F.relu(yield_pred - tensile_pred).mean()

        # 4) 轻度任务平衡：仍略强调强度任务，但比固定大权重更保守
        task_bias = torch.tensor([1.0, 1.05, 1.08], device=pred.device, dtype=pred.dtype)
        balanced_loss = (per_task_sq.mean(dim=0) * task_bias).mean()

        # 5) 极轻量输出范围稳定项：防止 batch 内预测完全塌缩，尤其是外推点过度平滑
        # 不使用输入梯度，仅约束输出统计量，稳定安全
        pred_std = pred.std(dim=0, unbiased=False)
        target_std = yb.std(dim=0, unbiased=False).detach()
        spread_loss = F.smooth_l1_loss(pred_std, target_std)

        loss = (
            0.55 * multi_task_loss
            + 0.30 * balanced_loss
            + 0.08 * phys_loss
            + 0.04 * spread_loss
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()