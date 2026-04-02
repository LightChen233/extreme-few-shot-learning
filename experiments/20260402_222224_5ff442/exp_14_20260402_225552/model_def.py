"""
模型定义 — LLM 唯一可修改的文件。
本版遵循“小步、稳健、可回退”原则：

相对当前基线，仅做两类保守改动：
1) 保留小型共享主干 + 多任务头 + tensile = yield + softplus(delta) 的稳定参数化
2) 去掉上版“样本级难例重加权”与“强度残差耦合惩罚”——历史表现提示主要问题在少数外推点，
   但训练集内做 residual-based hard mining 容易放大小样本方差，导致局部拟合不稳
3) 增加极轻量、通用的平滑先验：对同一 batch 内样本，两强度输出不应对输入扰动过于敏感；
   用预测差分的温和二次惩罚实现，但不对输入求梯度，数值稳定
4) 损失改为更稳的 Huber，并继续对强度任务略加权；学习率做极小幅下调

核心判断：
- 当前 MSE 已较低（<500），不适合大改架构
- 大误差主要来自外推点，激进机制（门控/MoE/重加权）在 29 条样本上风险高
- 先提高局部平滑性和训练稳定性，争取减少 460/12 插值点波动，并避免外推斜率继续失真
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 小样本下控制容量，保持与当前有效版本接近
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.SiLU(),
            nn.Linear(20, 10),
            nn.SiLU(),
        )

        self.strain_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

        self.yield_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

        self.delta_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        h = self.backbone(x)

        strain = self.strain_head(h)
        yld = self.yield_head(h)
        delta = F.softplus(self.delta_head(h))  # >= 0
        tensile = yld + delta                   # 保证 tensile >= yield

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 极小幅调整：略降学习率，保持 AdamW
    return torch.optim.AdamW(model.parameters(), lr=4.5e-4, weight_decay=1.5e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    # 略偏向强度，但避免过强 trade-off
    task_weights = torch.tensor([1.0, 1.12, 1.08], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)
        base_loss = (per_elem * task_weights).mean()

        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 冗余顺序约束（结构上已满足，但保留极轻权重更稳妥）
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 轻量“输出平滑”约束：对 batch 内随机配对样本，输入相近时强度预测不应剧烈跳变
        # 不对输入求梯度，只基于样本间距离做温和局部 Lipschitz 正则
        if len(xb) >= 2:
            pair_idx = torch.roll(torch.arange(len(xb), device=xb.device), shifts=1)
            x_dist = (xb - xb[pair_idx]).pow(2).mean(dim=1)  # [B]
            t_diff = (tensile_pred - tensile_pred[pair_idx]).pow(2)
            y_diff = (yield_pred - yield_pred[pair_idx]).pow(2)

            # 仅惩罚“相近输入却预测差很多”的情况；远距离样本基本不约束
            smooth_weight = torch.exp(-0.5 * x_dist).detach()
            smooth_penalty = (smooth_weight * (0.6 * t_diff + 0.4 * y_diff)).mean()
        else:
            smooth_penalty = torch.zeros((), device=xb.device, dtype=xb.dtype)

        loss = base_loss + 0.02 * order_penalty + 0.006 * smooth_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()