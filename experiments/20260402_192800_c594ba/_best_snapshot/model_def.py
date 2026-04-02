import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # 小样本下采用轻量 MoE：
        # - 一个 gate 学习工艺区域切分
        # - 两个小专家分别拟合平滑区 / 突变区
        # 复杂度受控，避免过拟合
        hidden = 24

        self.input_norm = nn.LayerNorm(input_dim)

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 2)
        )

        self.expert1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3)
        )

        self.expert2 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3)
        )

        # 全局线性支路，增强小样本稳健性
        self.linear_head = nn.Linear(input_dim, 3)

        # 可学习输出缩放，便于不同目标共同训练
        self.out_scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        x = self.input_norm(x)

        gate_logits = self.gate(x)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [B, 2]

        y1 = self.expert1(x)
        y2 = self.expert2(x)
        y_mix = gate_w[:, 0:1] * y1 + gate_w[:, 1:2] * y2

        y_lin = self.linear_head(x)

        # 非线性 + 线性残差组合，小数据更稳
        y = y_mix + 0.35 * y_lin
        y = y * self.out_scale

        return y


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=1e-3
    )


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)
    device = X_train.device

    # ------- 基于温度/时间识别“可能突变区”并加权 -------
    # 假设前两个输入是 temp, time（该任务设定下通常如此）
    # 若输入维度不足，则退化为均匀权重
    if X_train.shape[1] >= 2:
        temp = X_train[:, 0]
        time = X_train[:, 1]

        # 从训练集自适应找高温/长时区域，不硬编码绝对数值
        temp_hi = torch.quantile(temp, 0.65)
        time_hi = torch.quantile(time, 0.65)

        hard_region = ((temp >= temp_hi) & (time >= time_hi)).float()

        # 额外关注中高温且时间接近中上分位附近的样本
        time_mid = torch.quantile(time, 0.50)
        temp_mid_hi = torch.quantile(temp, 0.50)
        transition_region = (
            (temp >= temp_mid_hi) &
            (torch.abs(time - time_mid) <= (time.std(unbiased=False) + 1e-6))
        ).float()

        sample_weight = 1.0 + 1.2 * hard_region + 0.5 * transition_region
    else:
        sample_weight = torch.ones(n, device=device)

    # ------- 目标尺度归一化权重 -------
    # 避免抗拉/屈服因数值大而主导，也避免单纯追求应变
    y_std = y_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    base_task_w = 1.0 / (y_std ** 2)
    base_task_w = base_task_w / base_task_w.mean()

    # 略加强度任务，回应当前主要误差来源
    task_boost = torch.tensor([0.9, 1.15, 1.15], device=device)
    task_w = base_task_w * task_boost
    task_w = task_w / task_w.mean()

    # ------- 两阶段：先均匀采样，再对高残差样本进行轻度 hard mining -------
    with torch.no_grad():
        pred_all = model(X_train)
        abs_res = (pred_all - y_train).abs().mean(dim=1)
        err_scale = abs_res.median().clamp_min(1e-6)
        hard_factor = 1.0 + 0.8 * torch.tanh(abs_res / err_scale)

        probs = sample_weight * hard_factor
        probs = probs / probs.sum()

    # 小数据下每个 epoch 多看几次难例
    steps = max(4, math.ceil(n / batch_size) + 2)

    for step in range(steps):
        if step < max(1, steps // 3):
            idx = torch.randperm(n, device=device)[:batch_size]
        else:
            idx = torch.multinomial(probs, num_samples=min(batch_size, n), replacement=True)

        xb = X_train[idx]
        yb = y_train[idx]
        sw = sample_weight[idx]

        optimizer.zero_grad()
        pred = model(xb)

        # 加权多任务 MSE
        sq_err = (pred - yb) ** 2
        loss_main = (sq_err * task_w.view(1, -1)).mean(dim=1)
        loss_main = (loss_main * sw).mean()

        # Huber 项：提升对异常/突变点的稳健性
        huber = F.smooth_l1_loss(pred, yb, reduction="none", beta=1.0)
        loss_huber = (huber * task_w.view(1, -1)).mean(dim=1)
        loss_huber = (loss_huber * sw).mean()

        # 物理软约束：通常抗拉强度 >= 屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        loss_order = F.relu(yield_pred - tensile_pred).mean()

        # gate 熵正则：避免塌缩到单一专家，但强度要小
        x_norm = model.input_norm(xb)
        gate_logits = model.gate(x_norm)
        gate_prob = torch.softmax(gate_logits, dim=-1)
        gate_entropy = -(gate_prob * torch.log(gate_prob + 1e-8)).sum(dim=1).mean()

        loss = (
            0.7 * loss_main +
            0.3 * loss_huber +
            0.08 * loss_order -
            0.01 * gate_entropy
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()