"""
模型定义 — LLM 唯一可修改的文件。
本版在当前较优基线(MSE≈401)上做“保守改进”：

核心判断：
1) 大误差主要集中在外推点 470@12、440@24，且具有系统性偏差，不是纯随机噪声
2) 当前轻量门控方向是对的，但 coupling_penalty 可能过强地绑死抗拉/屈服残差，
   对 460@12 这类“屈服与抗拉不同步”的样本不够友好
3) 数据仅 29 条，因此不能上重型 MoE；只做极轻量“软分段 + 温和难例加权”

本次改动：
- 保持小型共享主干 + 多任务头，不大改范式
- 将 gate 输入扩展为 [x, x^2]，提升对温度/时间局部区间的软分段能力，但参数量很小
- 轻微缩小主干宽度，降低小样本过拟合
- 将强度头改为“共享强度子表征 + 两个输出头”，增强 tensile/yield 的相关学习，但不强绑残差
- 移除偏硬的 residual coupling penalty，改为更温和的表征共享
- 引入有上界的动态难例加权（基于 detach 残差，clamp 到 [1, 2.2]），聚焦突变区但防止爆炸
- 保留 tensile >= yield 的通用物理约束
- 使用更保守 lr / weight_decay / 梯度裁剪
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # 小型主干：进一步偏保守，减少过拟合风险
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
        )

        # 轻量软分段门控：使用 x 与 x^2 以增强对局部区间/弯曲趋势的感知
        gate_in_dim = input_dim * 2
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
            nn.Sigmoid()
        )

        # 应变头单独建模
        self.strain_head = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        # 强度共享子表征，替代“残差一致性强约束”
        self.strength_tower = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
        )
        self.tensile_head = nn.Linear(8, 1)
        self.yield_head = nn.Linear(8, 1)

    def forward(self, x):
        h = self.backbone(x)

        x2 = x * x
        g_in = torch.cat([x, x2], dim=-1)
        g = self.gate(g_in)

        # 残差式温和门控
        h = h * (0.80 + 0.40 * g)

        strain = self.strain_head(h)

        hs = self.strength_tower(h)
        tensile = self.tensile_head(hs)
        yld = self.yield_head(hs)

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 小样本下继续保守微调
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.2e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    # Huber 稳健损失
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.2)

    # 继续略偏重强度，但更温和
    task_weights = torch.tensor([1.0, 1.18, 1.10], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        per_elem = criterion(pred, yb)  # [B, 3]

        # 动态难例加权：仅温和强调当前batch内较难样本，避免被极端点牵着走
        with torch.no_grad():
            abs_err = (pred.detach() - yb).abs()  # [B, 3]
            sample_err = (abs_err * task_weights).mean(dim=1)  # [B]
            norm = sample_err.mean().clamp_min(1e-6)
            hard_w = (1.0 + 0.35 * sample_err / norm).clamp(1.0, 2.2)  # [B]

        weighted_elem = per_elem * task_weights.unsqueeze(0)
        base_loss = (weighted_elem.mean(dim=1) * hard_w).mean()

        # 通用物理约束：抗拉强度通常不低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 温和输出幅值正则，抑制外推点异常漂移；不直接压预测值，只压 batch 内方差过大倾向
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        smooth_penalty = (pred_centered.pow(2).mean())

        loss = base_loss + 0.10 * order_penalty + 0.002 * smooth_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
        optimizer.step()