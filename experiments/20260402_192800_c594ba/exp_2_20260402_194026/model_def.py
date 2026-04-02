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

        # 小样本(29条)下使用轻量 MoE：
        # - 共享干路提取平滑趋势
        # - gating 学习不同工艺区间（如 440/1h, 460-470/12h, 440/24h）的分段行为
        # - 3个专家足够表达“常规区 / 突变区 / 过时效区”而不过度复杂
        hidden = 24
        trunk_dim = 16
        num_experts = 3

        self.input_norm = nn.LayerNorm(input_dim)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, trunk_dim),
            nn.Tanh(),
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, num_experts)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_dim, 16),
                nn.Tanh(),
                nn.Linear(16, 3)
            ) for _ in range(num_experts)
        ])

        # 全局线性支路，增强低样本稳定性
        self.linear_head = nn.Linear(input_dim, 3)

        # 可学习输出缩放，便于多任务平衡
        self.out_scale = nn.Parameter(torch.ones(3))

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        feat = self.trunk(x)

        gate_logits = self.gate(x)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [B, E]

        expert_outs = torch.stack([expert(feat) for expert in self.experts], dim=1)  # [B, E, 3]
        moe_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)  # [B, 3]

        linear_out = self.linear_head(x)

        out = linear_out + moe_out
        out = out * self.out_scale
        return out


def build_optimizer(model):
    # 小样本+多任务，较小学习率与轻微权重衰减更稳
    return torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练：鲁棒多任务损失 + 难例重加权 + 物理约束"""
    model.train()
    n = len(X_train)

    # 首先在全体样本上前向一次，计算当前残差用于难例采样/加权
    with torch.no_grad():
        pred_all = model(X_train)
        abs_err = (pred_all - y_train).abs()  # [N,3]

        # 按目标尺度归一化，避免强度指标数值主导一切
        target_scale = y_train.std(dim=0, unbiased=False).clamp_min(1.0)
        norm_abs_err = abs_err / target_scale.unsqueeze(0)

        # 样本难度：三任务平均
        sample_hardness = norm_abs_err.mean(dim=1)

        # 采样概率：难例更高，但保留所有样本机会，避免过拟合个别点
        sample_prob = sample_hardness + 0.2
        sample_prob = sample_prob / sample_prob.sum()

    # batch size 对 29 条样本取较小值，增加随机性
    batch_size = min(batch_size, max(4, n))

    # 每个 epoch 做略多于 1 次遍历，提升难例学习强度
    steps = max(4, (n + batch_size - 1) // batch_size + 1)

    # 任务权重：重点压低抗拉/屈服误差，兼顾应变
    task_weights = torch.tensor([1.2, 1.6, 1.4], dtype=X_train.dtype, device=X_train.device)

    for _ in range(steps):
        idx = torch.multinomial(sample_prob, batch_size, replacement=True)

        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        diff = pred - yb
        abs_diff = diff.abs()

        # 按任务标准差归一化后的鲁棒损失（SmoothL1）
        target_scale = y_train.std(dim=0, unbiased=False).clamp_min(1.0)
        norm_diff = diff / target_scale.unsqueeze(0)

        base_loss_per_target = F.smooth_l1_loss(
            norm_diff, torch.zeros_like(norm_diff), reduction="none", beta=0.8
        )  # [B,3]

        # 难例动态重加权：当前 batch 中残差大的样本权重更高
        with torch.no_grad():
            sample_err = norm_diff.abs().mean(dim=1)
            hard_w = 1.0 + 1.5 * torch.clamp(sample_err, 0.0, 3.0) / 3.0  # [B]

        weighted_task_loss = (base_loss_per_target * task_weights.unsqueeze(0)).mean(dim=1)
        data_loss = (weighted_task_loss * hard_w).mean()

        # 物理软约束1：通常抗拉强度 >= 屈服强度
        phys_order_loss = F.relu(pred[:, 2] - pred[:, 1]).mean()

        # 物理软约束2：输出不要远离训练分布过多，限制极端外推
        y_mean = y_train.mean(dim=0)
        y_std = y_train.std(dim=0, unbiased=False).clamp_min(1.0)
        z = (pred - y_mean.unsqueeze(0)) / y_std.unsqueeze(0)
        range_reg = F.relu(z.abs() - 3.0).mean()

        # gating 熵正则：避免单专家塌缩，但不强制平均
        if hasattr(model, "gate"):
            gate_logits = model.gate(model.input_norm(xb))
            gate_prob = torch.softmax(gate_logits, dim=-1)
            gate_entropy = -(gate_prob * torch.log(gate_prob.clamp_min(1e-8))).sum(dim=1).mean()
            gate_reg = -0.01 * gate_entropy
        else:
            gate_reg = torch.tensor(0.0, device=X_train.device, dtype=X_train.dtype)

        loss = data_loss + 0.15 * phys_order_loss + 0.03 * range_reg + gate_reg

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()