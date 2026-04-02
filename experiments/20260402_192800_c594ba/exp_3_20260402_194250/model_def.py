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

        # 小样本场景：使用低容量 + 混合专家，专门处理 440/1h、460/12h、470/12h 等潜在“组织演化突变区”
        hidden = 24
        expert_hidden = 20
        n_experts = 3

        # 共享底座
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # gating：根据输入自适应选择专家
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, n_experts)
        )

        # 3 个轻量专家，避免参数过多
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, expert_hidden),
                nn.SiLU(),
                nn.Linear(expert_hidden, 3)
            ) for _ in range(n_experts)
        ])

        # 全局线性捷径：增强对整体平滑趋势的拟合，减少仅靠非线性网络造成的断层
        self.linear_head = nn.Linear(input_dim, 3)

        # 可学习输出缩放，便于平衡三任务梯度
        self.out_scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        h = self.backbone(x)                           # [B, H]
        gate_logits = self.gate(x)                    # [B, E]
        gate_w = torch.softmax(gate_logits, dim=-1)   # [B, E]

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(h).unsqueeze(1))  # [B,1,3]
        expert_outs = torch.cat(expert_outs, dim=1)     # [B,E,3]

        moe_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)  # [B,3]
        linear_out = self.linear_head(x)

        out = linear_out + moe_out * self.out_scale
        return out


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """
    单 epoch 训练：
    1) 按目标标准差归一化多任务损失，避免抗拉强度主导训练
    2) 对 hard example 做动态加权，聚焦 440/1h、460/12h、470/12h 等高误差区
    3) 加入物理约束：抗拉强度 >= 屈服强度
    4) 轻量 gating 熵正则，防止专家塌缩
    """
    model.train()
    n = len(X_train)
    device = X_train.device

    # 目标尺度归一化：平衡 strain / tensile / yield
    with torch.no_grad():
        y_std = y_train.std(dim=0, unbiased=False).clamp_min(1.0)

    # 先用当前模型估计每个样本难度，进行 hard-example 重加权采样
    with torch.no_grad():
        pred_all = model(X_train)
        abs_err = ((pred_all - y_train).abs() / y_std).mean(dim=1)  # [N]
        sample_w = 1.0 + 1.5 * torch.clamp(abs_err, 0.0, 3.0)
        sample_w = sample_w / sample_w.sum()

    num_steps = max(1, (n + batch_size - 1) // batch_size)

    for _ in range(num_steps):
        idx = torch.multinomial(sample_w, batch_size, replacement=True)
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 归一化残差
        diff = (pred - yb) / y_std

        # 主损失：SmoothL1 比纯 MSE 更稳，且保留对大误差样本的敏感性
        per_elem = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none', beta=1.0)

        # batch 内 hard example 动态权重
        with torch.no_grad():
            hard = diff.abs().mean(dim=1)
            hard_w = 1.0 + 1.5 * torch.clamp(hard, 0.0, 3.0)  # [B]

        # 任务权重：适度提高强度任务，避免仅改善应变
        task_w = torch.tensor([1.0, 1.35, 1.25], device=device).view(1, 3)
        data_loss = (per_elem * task_w).mean(dim=1)
        data_loss = (data_loss * hard_w).mean()

        # 物理约束：UTS >= YS
        phys_loss = F.relu(pred[:, 2] - pred[:, 1]).mean()

        # gating 熵正则：避免长期只用一个专家，也避免完全平均
        gate_logits = model.gate(xb)
        gate_prob = torch.softmax(gate_logits, dim=-1)
        entropy = -(gate_prob * torch.log(gate_prob.clamp_min(1e-8))).sum(dim=1).mean()

        # 温和正则
        l2_reg = torch.tensor(0.0, device=device)
        for p in model.parameters():
            l2_reg = l2_reg + p.pow(2).sum()

        loss = data_loss + 0.12 * phys_loss + 0.005 * entropy + 1e-6 * l2_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()