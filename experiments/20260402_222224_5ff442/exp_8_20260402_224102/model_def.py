"""
模型定义 — LLM 唯一可修改的文件。
本版遵循保守改进原则，在当前较优基线上做“小步修正”：

1) 保持小型共享干路 + 多任务头，不重写整体范式
2) 保留轻量 gating，但改成更稳的残差门控
3) 针对已知系统性问题做非常温和的训练侧修正：
   - 外推/突变样本存在，但直接 hard mining 容易过拟合
   - 因此仅使用“有上界的动态重加权”，且基于 detach 残差，限制在 [1, 2]
4) 针对外推点的系统偏差，引入极轻量输出平滑正则：
   - 抗拉/屈服对温度与时间不应出现无物理依据的剧烈振荡
   - 只在 batch 内相近样本上施加非常弱的一致性约束，避免过强形状假设
5) 保留通用物理约束：抗拉强度 >= 屈服强度
6) 使用 AdamW + 梯度裁剪，确保数值稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 小模型主干：29条样本下控制容量
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
        )

        # 轻量门控：仅做温和调制
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.Sigmoid(),
        )

        self.strain_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.tensile_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.yield_head = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        h = self.backbone(x)
        g = self.gate(x)
        # 更保守的残差门控，避免局部工艺区间被放大过头
        h = h * (0.85 + 0.30 * g)

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 相比当前基线只做小幅微调：略降 lr，略增 wd 抑制外推振荡
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.2e-4)


def _pairwise_local_smoothness(xb, pred):
    """
    对 batch 内“输入接近”的样本施加极弱局部平滑，主要作用于强度输出。
    目的是减少外推时无根据的剧烈波动，而不是强行假设全局单调。
    """
    if xb.size(0) < 3:
        return pred.new_tensor(0.0)

    # 使用前两个维度作为 temp/time 主特征；若 feature_agent 做了扩展，这里仍可工作
    xt = xb[:, :2]
    dist = torch.cdist(xt, xt, p=2)  # [B, B]

    # 自适应近邻阈值：取非对角元素中较小分位的距离尺度
    eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
    valid = dist.masked_fill(eye, float("inf"))
    k = max(1, valid.numel() // 4)
    scale = torch.topk(valid.flatten(), k=k, largest=False).values.mean()
    scale = scale.clamp_min(1e-6)

    # 近邻权重，抑制远距离样本错误平滑
    w = torch.exp(-dist / (scale + 1e-6))
    w = w.masked_fill(eye, 0.0)

    # 仅对 tensile / yield 施加更强平滑；strain 保持更自由
    dp_t = pred[:, 1:2] - pred[:, 1:2].transpose(0, 1)
    dp_y = pred[:, 2:3] - pred[:, 2:3].transpose(0, 1)
    dp_s = pred[:, 0:1] - pred[:, 0:1].transpose(0, 1)

    smooth_t = (w * dp_t.pow(2)).sum() / (w.sum() + 1e-6)
    smooth_y = (w * dp_y.pow(2)).sum() / (w.sum() + 1e-6)
    smooth_s = (w * dp_s.pow(2)).sum() / (w.sum() + 1e-6)

    return 0.45 * smooth_t + 0.45 * smooth_y + 0.10 * smooth_s


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    perm = torch.randperm(n, device=X_train.device)

    criterion = nn.SmoothL1Loss(reduction="none", beta=1.5)

    # 延续当前较优方向：略偏重强度，但避免过度牺牲应变
    task_weights = torch.tensor([1.0, 1.18, 1.10], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        per_elem = criterion(pred, yb)  # [B, 3]

        # 温和动态重加权：仅略微强调难例，防止 470@12 等系统性难点被平均化
        # 基于 detach 残差，且上界严格限制，避免训练不稳定
        abs_res = (pred.detach() - yb).abs()
        norm_res = abs_res / (abs_res.mean(dim=0, keepdim=True) + 1e-6)
        sample_w = 1.0 + 0.25 * norm_res.mean(dim=1, keepdim=True)
        sample_w = sample_w.clamp(1.0, 2.0)

        base_loss = (per_elem * task_weights * sample_w).mean()

        # 物理约束：抗拉强度通常不低于屈服强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 抗拉/屈服残差方向通常有关联，但只保留很弱约束
        tensile_res = pred[:, 1] - yb[:, 1]
        yield_res = pred[:, 2] - yb[:, 2]
        coupling_penalty = F.smooth_l1_loss(
            yield_res, tensile_res, reduction="mean", beta=12.0
        )

        # 极弱局部平滑：缓解外推点附近过度振荡
        smooth_penalty = _pairwise_local_smoothness(xb, pred)

        loss = (
            base_loss
            + 0.08 * order_penalty
            + 0.02 * coupling_penalty
            + 0.006 * smooth_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()