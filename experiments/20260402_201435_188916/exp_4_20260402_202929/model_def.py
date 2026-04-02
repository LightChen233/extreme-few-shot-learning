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

        # 小样本下采用低容量 + MoE，专门处理可能存在的工艺区间突变/峰值
        hidden = 24
        trunk = 16
        expert_hidden = 16

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, trunk),
            nn.Tanh(),
        )

        # 3个小专家：分别学习不同工艺区域的局部规律
        self.expert1 = nn.Sequential(
            nn.Linear(trunk, expert_hidden),
            nn.Tanh(),
            nn.Linear(expert_hidden, 3),
        )
        self.expert2 = nn.Sequential(
            nn.Linear(trunk, expert_hidden),
            nn.Tanh(),
            nn.Linear(expert_hidden, 3),
        )
        self.expert3 = nn.Sequential(
            nn.Linear(trunk, expert_hidden),
            nn.Tanh(),
            nn.Linear(expert_hidden, 3),
        )

        # gating 网络：自动对不同温度-时间区域路由
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )

        # 全局线性捷径，增强外推稳定性
        self.linear_head = nn.Linear(input_dim, 3)

        # 可学习输出缩放，避免某一任务主导
        self.out_scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        h = self.feature(x)

        e1 = self.expert1(h)
        e2 = self.expert2(h)
        e3 = self.expert3(h)

        gates = torch.softmax(self.gate(x), dim=-1)  # [N, 3]
        moe_out = (
            gates[:, 0:1] * e1 +
            gates[:, 1:2] * e2 +
            gates[:, 2:3] * e3
        )

        out = self.linear_head(x) + moe_out
        out = out * self.out_scale
        return out


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)


def _robust_weighted_loss(pred, target):
    """
    多任务稳健损失：
    1) 按目标尺度归一化，避免抗拉/屈服压制应变
    2) SmoothL1 提升抗异常点能力
    3) 轻度 hard-example weighting，聚焦 470/12、440/24 一类突变点
    """
    # 基于 batch 目标尺度进行归一化
    scale = target.detach().abs().mean(dim=0).clamp_min(1.0)
    diff = (pred - target) / scale

    # 每样本每任务 smooth l1
    per_elem = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=0.6)

    # 样本级 hard weighting：残差越大权重越高，但做截断防止失稳
    with torch.no_grad():
        sample_err = diff.abs().mean(dim=1)
        w = 1.0 + 1.5 * torch.clamp(sample_err, 0.0, 2.0)

    return (per_elem.mean(dim=1) * w).mean()


def _pairwise_physics_loss(model, X):
    """
    基于输入维度前两列近似对应 temp/time 的通用弱物理约束：
    - 输出应保持平滑，不应在邻近工艺点剧烈跳变
    - 屈服强度通常不应高于抗拉强度
    - gating 熵正则，避免过早塌缩到单专家
    """
    loss = 0.0

    pred = model(X)

    # 基本物理关系：UTS >= YS
    tensile = pred[:, 1]
    yld = pred[:, 2]
    loss = loss + 0.2 * F.relu(yld - tensile).mean()

    # 邻近样本平滑约束：对输入近的点，预测差不要无约束跳变
    if X.size(0) >= 4:
        with torch.no_grad():
            d = torch.cdist(X, X, p=2)
            d.fill_diagonal_(1e9)
            nn_idx = d.argmin(dim=1)

        pred_nn = pred[nn_idx]
        x_dist = ((X - X[nn_idx]) ** 2).sum(dim=1).sqrt().detach()
        p_dist = ((pred - pred_nn) ** 2).mean(dim=1)

        # 输入越近，约束越强
        smooth_w = torch.exp(-x_dist / (x_dist.mean().clamp_min(1e-6)))
        loss = loss + 0.05 * (smooth_w * p_dist).mean()

    # gating 不要塌缩，保留分段建模能力
    if hasattr(model, "gate"):
        logits = model.gate(X)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
        loss = loss - 0.01 * entropy

    return loss


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练，可修改 loss、batch size、采样策略等"""
    model.train()

    n = len(X_train)
    if n <= 16:
        batch_size = min(4, n)
    else:
        batch_size = min(batch_size, n)

    # 小样本下每个 epoch 做两轮，提升拟合稳定性
    repeats = 2

    for _ in range(repeats):
        perm = torch.randperm(n, device=X_train.device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()

            pred = model(xb)

            data_loss = _robust_weighted_loss(pred, yb)

            # 在全训练集上加弱物理正则，利用全部几何关系
            phys_loss = _pairwise_physics_loss(model, X_train)

            loss = data_loss + phys_loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()