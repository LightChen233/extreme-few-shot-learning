import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # 小样本场景：采用轻量 MoE + 残差结构，避免过深过宽
        hidden = 48
        trunk = 32
        expert_hidden = 32
        n_experts = 3

        # 输入标准化参数（在训练中自适应更新）
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))
        self.register_buffer("y_mean", torch.zeros(3))
        self.register_buffer("y_std", torch.ones(3))
        self.register_buffer("fitted_stats", torch.tensor(0, dtype=torch.long))

        # 共享主干
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.05),
            nn.Linear(hidden, trunk),
            nn.SiLU(),
        )

        # gating：处理不同温度/时间区域的突变
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, n_experts)
        )

        # 多专家，每个专家输出3维残差
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk, expert_hidden),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(expert_hidden, 3)
            ) for _ in range(n_experts)
        ])

        # 直接线性支路，增强外推稳定性
        self.linear_head = nn.Linear(input_dim, 3)

        # 多任务校正头
        self.refine = nn.Sequential(
            nn.Linear(trunk + 3, 24),
            nn.SiLU(),
            nn.Linear(24, 3)
        )

        # 原始样品先验均值，作为弱基线
        self.register_buffer("base_prior", torch.tensor([6.94, 145.83, 96.60], dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.9)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def adapt_stats(self, X, y=None):
        if X.ndim == 2:
            self.x_mean.copy_(X.mean(dim=0))
            self.x_std.copy_(X.std(dim=0).clamp_min(1e-3))
        if y is not None and y.ndim == 2:
            self.y_mean.copy_(y.mean(dim=0))
            self.y_std.copy_(y.std(dim=0).clamp_min(1e-3))
        self.fitted_stats.fill_(1)

    def _normalize_x(self, x):
        if self.fitted_stats.item() == 1:
            return (x - self.x_mean) / self.x_std
        return x

    def forward(self, x):
        x_in = x
        x = self._normalize_x(x)

        shared = self.shared(x)

        gate_logits = self.gate(x)
        gate_w = F.softmax(gate_logits, dim=-1)

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(shared).unsqueeze(1))
        expert_outs = torch.cat(expert_outs, dim=1)  # [B, E, 3]
        moe_out = (expert_outs * gate_w.unsqueeze(-1)).sum(dim=1)

        lin_out = self.linear_head(x)
        rough = 0.65 * moe_out + 0.35 * lin_out

        refined = rough + 0.5 * self.refine(torch.cat([shared, rough], dim=-1))

        # 弱物理先验：围绕整体均值学习偏移，提升极小样本稳定性
        out = refined + self.base_prior

        # 软约束：屈服强度通常不应高于抗拉强度
        tensile = out[:, 1:2]
        yld = out[:, 2:3]
        yld = torch.minimum(yld, tensile - 1e-3)

        # 应变非负
        strain = F.softplus(out[:, 0:1])

        return torch.cat([strain, tensile, yld], dim=1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=2e-4)


def train_step(model, optimizer, X_train, y_train):
    device = next(model.parameters()).device
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    if model.fitted_stats.item() == 0:
        model.adapt_stats(X_train, y_train)

    model.train()

    # 小数据集：每个 epoch 做多次带噪声重采样，提升鲁棒性
    n = X_train.size(0)
    batch_size = min(12, n)
    inner_steps = 6

    # 训练步计数，用于调学习率与难例加权
    if not hasattr(model, "_step_count"):
        model._step_count = 0

    # 简单余弦衰减
    model._step_count += 1
    base_lr = 8e-3
    min_lr = 8e-4
    period = 250
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * (model._step_count % period) / period))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    for _ in range(inner_steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        xb = X_train[idx]
        yb = y_train[idx]

        # 特征轻微扰动，缓解重复工艺点带来的过拟合
        noise_scale = 0.01
        xb_aug = xb + torch.randn_like(xb) * noise_scale

        pred = model(xb_aug)

        # 基础逐任务标准化误差
        y_scale = model.y_std.to(device).clamp_min(1.0)
        err = (pred - yb) / y_scale

        # Huber 比纯 MSE 更稳健，适合小样本+局部突变
        huber = F.smooth_l1_loss(err, torch.zeros_like(err), reduction='none')

        # 动态难例加权：突出突变区/大残差样本
        with torch.no_grad():
            sample_res = err.abs().mean(dim=1)
            hard_w = 1.0 + 1.5 * torch.tanh(sample_res)

            # 对高温长时区域给轻微额外关注，不硬编码具体值，按输入极端性自适应
            xz = model._normalize_x(xb)
            extremeness = xz[:, : min(4, xz.size(1))].abs().mean(dim=1)
            hard_w = hard_w * (1.0 + 0.15 * torch.tanh(extremeness))

        # 任务权重：评估更重视应变相对误差，但强度误差绝对量大，也需平衡
        task_w = torch.tensor([1.8, 1.0, 1.0], device=device).view(1, 3)
        data_loss = ((huber * task_w).mean(dim=1) * hard_w).mean()

        # 物理约束1：屈服 <= 抗拉
        rel_penalty = F.relu(pred[:, 2] - pred[:, 1]).pow(2).mean()

        # 物理约束2：抗拉、屈服不应为明显负值
        nonneg_strength = (F.relu(-pred[:, 1]).pow(2) + F.relu(-pred[:, 2]).pow(2)).mean()

        # gating 熵正则：避免所有样本塌缩到单专家，但不过强
        x_norm = model._normalize_x(xb)
        gate_logits = model.gate(x_norm)
        gate_prob = F.softmax(gate_logits, dim=-1)
        gate_entropy = -(gate_prob * (gate_prob.clamp_min(1e-8).log())).sum(dim=1).mean()

        # L2 输出平滑，避免小样本时出现离谱值
        pred_reg = ((pred - model.base_prior.to(device)) / model.y_std.to(device).clamp_min(1.0)).pow(2).mean()

        loss = (
            data_loss
            + 0.6 * rel_penalty
            + 0.05 * nonneg_strength
            - 0.01 * gate_entropy
            + 0.003 * pred_reg
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()