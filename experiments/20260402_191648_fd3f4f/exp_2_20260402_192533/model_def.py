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
        self.input_dim = input_dim

        # 小样本场景：采用低容量 + 分段/门控思想
        # 共享干路
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
        )

        # 门控网络：用于处理 460/470°C、12h 等可能存在性能突变的区域
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 2)
        )

        # 两个轻量专家
        self.expert1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )
        self.expert2 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )

        # 全局线性残差支路，提升低样本稳定性
        self.linear_skip = nn.Linear(input_dim, 3)

        # 记录标准化信息，首次训练时自适应估计
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))
        self.register_buffer("y_mean", torch.zeros(3))
        self.register_buffer("y_std", torch.ones(3))
        self.stats_initialized = False

    def _normalize_x(self, x):
        return (x - self.x_mean) / self.x_std.clamp_min(1e-6)

    def _denormalize_y(self, y):
        return y * self.y_std + self.y_mean

    def forward(self, x):
        x_norm = self._normalize_x(x)
        h = self.backbone(x_norm)

        gate_logits = self.gate(x_norm)
        gate_w = torch.softmax(gate_logits, dim=-1)

        out1 = self.expert1(h)
        out2 = self.expert2(h)
        mixed = gate_w[:, 0:1] * out1 + gate_w[:, 1:2] * out2

        y_norm = mixed + 0.35 * self.linear_skip(x_norm)
        y = self._denormalize_y(y_norm)
        return y


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练：标准化 + 难例重加权 + 物理约束 + 小样本稳定训练"""
    device = X_train.device
    model.train()

    # 首次进入时，依据训练集估计标准化统计量
    if not getattr(model, "stats_initialized", False):
        with torch.no_grad():
            x_mean = X_train.mean(dim=0)
            x_std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)
            y_mean = y_train.mean(dim=0)
            y_std = y_train.std(dim=0, unbiased=False).clamp_min(1e-6)

            model.x_mean.copy_(x_mean)
            model.x_std.copy_(x_std)
            model.y_mean.copy_(y_mean)
            model.y_std.copy_(y_std)
            model.stats_initialized = True

    n = len(X_train)
    if n <= 16:
        batch_size = n
    else:
        batch_size = min(batch_size, n)

    # 基于“高温长时”区域构造轻度过采样权重
    with torch.no_grad():
        x = X_train
        temp = x[:, 0]
        time = x[:, 1] if x.shape[1] > 1 else torch.zeros_like(temp)

        temp_thr = temp.median()
        time_thr = time.median()

        hard_region = ((temp >= temp_thr) & (time >= time_thr)).float()
        sample_w = 1.0 + 1.5 * hard_region
        sample_w = sample_w / sample_w.sum()

    # 使用带放回采样，提升难区出现频率
    num_draw = max(n, batch_size * ((n + batch_size - 1) // batch_size))
    indices = torch.multinomial(sample_w, num_samples=num_draw, replacement=True)

    for i in range(0, num_draw, batch_size):
        idx = indices[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 基础损失：按输出尺度归一化，避免抗拉强度主导
        err_norm = (pred - yb) / model.y_std.clamp_min(1e-6)

        # 动态难例加权：聚焦突变点和当前残差大的样本
        with torch.no_grad():
            per_sample = err_norm.abs().mean(dim=1)
            hard_w = 1.0 + 1.5 * torch.tanh(per_sample)

        base_loss = (hard_w * (err_norm ** 2).mean(dim=1)).mean()

        # 稳健项：L1 提高对小样本异常点的鲁棒性
        l1_loss = (hard_w * err_norm.abs().mean(dim=1)).mean()

        # 物理约束：抗拉强度 >= 屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 门控熵正则：避免极端塌缩，同时保留分段能力
        x_norm = model._normalize_x(xb)
        gate_logits = model.gate(x_norm)
        gate_prob = torch.softmax(gate_logits, dim=-1)
        gate_entropy = -(gate_prob * torch.log(gate_prob.clamp_min(1e-8))).sum(dim=1).mean()

        loss = base_loss + 0.25 * l1_loss + 0.2 * order_penalty - 0.01 * gate_entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()