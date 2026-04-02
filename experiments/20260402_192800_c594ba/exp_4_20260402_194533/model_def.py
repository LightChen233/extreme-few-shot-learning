"""
模型定义 — 可修改文件
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

设计思路（针对小样本 29 条）：
1. 采用轻量级共享干路 + 多任务头，避免大模型过拟合；
2. 使用软门控（gating）形成简化的分段/混合专家结构，帮助拟合 460/470℃ 长时效附近的性能突变；
3. 输出采用“有序强度参数化”：
   - pred[:,1] = tensile
   - pred[:,2] = tensile - softplus(gap)  => 保证 yield <= tensile
   这是通用物理约束，不硬编码具体数值；
4. 训练时使用：
   - 加权 Huber：降低离群点梯度爆炸，同时对难例自适应加权；
   - 输出维度重加权：更关注当前误差最大的抗拉/屈服强度；
   - 排序约束辅助项：进一步稳定 tensile >= yield；
5. 小样本下加入较强正则：LayerNorm + Dropout + AdamW。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # 轻量 gating，根据输入自适应路由到不同专家
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )

        # 两个小专家，容量受控，适合小样本
        self.expert1 = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.LayerNorm(24),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(24, 16),
            nn.SiLU(),
        )
        self.expert2 = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.LayerNorm(24),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(24, 16),
            nn.SiLU(),
        )

        # 共享后处理
        self.trunk = nn.Sequential(
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Dropout(0.05),
        )

        # 多任务头：
        # strain 直接预测
        self.head_strain = nn.Linear(16, 1)

        # tensile 直接预测；yield 通过 tensile - softplus(gap) 得到
        self.head_strength = nn.Linear(16, 2)  # [tensile_raw, gap_raw]

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [N, 2]

        h1 = self.expert1(x)
        h2 = self.expert2(x)
        h = gate_w[:, :1] * h1 + gate_w[:, 1:] * h2
        h = self.trunk(h)

        strain = self.head_strain(h)
        tensile_raw, gap_raw = self.head_strength(h).chunk(2, dim=-1)

        # 物理约束：yield <= tensile
        gap = F.softplus(gap_raw)
        yield_strength = tensile_raw - gap

        out = torch.cat([strain, tensile_raw, yield_strength], dim=-1)
        return out


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=1e-3
    )


def _weighted_huber(pred, target, delta=8.0, dim_weights=None):
    """
    pred/target: [N, 3]
    dim_weights: [3]
    难例重加权：基于样本当前残差大小，但做截断，避免个别点主导训练。
    """
    err = pred - target
    abs_err = err.abs()

    # Huber
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=abs_err.device, dtype=abs_err.dtype))
    linear = abs_err - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear  # [N,3]

    if dim_weights is not None:
        huber = huber * dim_weights.view(1, -1)

    # 样本难度权重：按每个样本三任务平均绝对误差决定
    sample_err = abs_err.mean(dim=1, keepdim=True)
    norm = sample_err.detach().mean().clamp_min(1e-6)
    sample_w = 1.0 + 0.8 * torch.clamp(sample_err / norm - 1.0, min=0.0, max=2.0)

    return (huber * sample_w).mean()


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """
    单 epoch 训练
    - 小 batch + shuffle
    - 加权 Huber
    - 强度任务更高权重
    - 排序约束辅助损失
    """
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device if hasattr(X_train, "device") else None)

    # 根据当前验证表现趋势，适度强调 tensile / yield
    # strain 已相对更容易被优化，因此降低其相对权重
    dim_weights = torch.tensor([0.8, 1.35, 1.15], dtype=X_train.dtype, device=X_train.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad()
        pred = model(xb)

        # 主损失
        loss_main = _weighted_huber(pred, yb, delta=8.0, dim_weights=dim_weights)

        # 软排序约束（虽然模型结构已基本满足，这里再加一层数值稳定）
        order_penalty = F.relu(pred[:, 2] - pred[:, 1]).mean()

        # 轻微输出尺度正则，避免小样本下局部发散
        l2_pred = (pred ** 2).mean()

        loss = loss_main + 2.0 * order_penalty + 1e-5 * l2_pred
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()