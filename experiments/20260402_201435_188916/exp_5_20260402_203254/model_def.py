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

        # 小样本场景：采用轻量共享干路 + 多任务头，避免大网络过拟合
        hidden = 24
        trunk_hidden = 16

        self.input_norm = nn.LayerNorm(input_dim)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Dropout(0.08),
            nn.Linear(hidden, trunk_hidden),
            nn.Tanh(),
        )

        # 多任务独立头，降低“应变优化拖累强度”的 trade-off
        self.head_strain = nn.Sequential(
            nn.Linear(trunk_hidden, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )
        self.head_tensile = nn.Sequential(
            nn.Linear(trunk_hidden, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )
        self.head_yield = nn.Sequential(
            nn.Linear(trunk_hidden, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )

        # 任务不确定性加权参数：自动平衡三目标 loss
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        x = self.input_norm(x)
        h = self.trunk(x)

        strain = self.head_strain(h)
        tensile = self.head_tensile(h)
        yld = self.head_yield(h)

        out = torch.cat([strain, tensile, yld], dim=-1)
        return out


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)


def _pairwise_physics_loss(model, X, pred):
    """
    基于 batch 内样本构造弱物理约束：
    1) 屈服强度 <= 抗拉强度
    2) 若温度接近且时间更长，则性质变化不应出现极端跳变（平滑约束）
       这里只做很弱的局部 Lipschitz/平滑先验，避免硬编码单调方向
    """
    loss = pred.new_tensor(0.0)

    # 屈服强度通常不应高于抗拉强度
    yield_le_tensile = F.relu(pred[:, 2] - pred[:, 1])
    loss = loss + yield_le_tensile.mean()

    # 如果输入至少有两个维度，默认前两维对应 temp/time 或其主特征
    if X.size(1) >= 2 and X.size(0) >= 3:
        temp = X[:, 0]
        time = X[:, 1]

        # 构造 pairwise 距离
        dt = torch.abs(temp[:, None] - temp[None, :])
        dh = torch.abs(time[:, None] - time[None, :])

        # 只在“温度近、时间近”的局部邻域加平滑，防止外推点发散
        near_mask = ((dt < dt.median() + 1e-8) & (dh < dh.median() + 1e-8)).float()
        eye = torch.eye(X.size(0), device=X.device)
        near_mask = near_mask * (1.0 - eye)

        if near_mask.sum() > 0:
            pd = torch.abs(pred[:, None, :] - pred[None, :, :]).mean(dim=-1)
            xd = dt + dh + 1e-6
            smooth = (near_mask * (pd / xd)).sum() / near_mask.sum()
            loss = loss + 0.02 * smooth

    return loss


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """
    单 epoch 训练逻辑：
    - 小样本下全局 shuffle + 小批量
    - 多任务不确定性加权
    - 对 hard examples 动态加权，聚焦 460/470℃、12/24h 一类突变点
    - 使用 Huber 替代 MSE，降低个别异常点造成的训练崩坏
    - 加入弱物理约束：YS <= UTS + 局部平滑
    """
    model.train()
    n = len(X_train)

    # 先做一次当前残差评估，用于 hard example 重加权
    with torch.no_grad():
        pred_all = model(X_train)
        abs_err = torch.abs(pred_all - y_train)

        # 各目标按尺度归一，避免强度目标数值主导
        target_scale = y_train.std(dim=0, unbiased=False).clamp_min(1.0)
        norm_err = abs_err / target_scale

        # 样本级难度：三目标平均
        sample_hardness = norm_err.mean(dim=1)

        # 动态权重：强调难例但不过分极端
        sample_weights = 1.0 + 1.5 * sample_hardness
        sample_weights = sample_weights / sample_weights.mean()

    perm = torch.randperm(n)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]
        wb = sample_weights[idx]

        optimizer.zero_grad()
        pred = model(xb)

        # Huber loss：比 MSE 更稳健
        per_elem = F.smooth_l1_loss(pred, yb, reduction="none", beta=1.0)

        # 任务尺度归一，缓解应变/强度量纲差异
        batch_scale = y_train.std(dim=0, unbiased=False).to(yb.device).clamp_min(1.0)
        per_elem = per_elem / batch_scale

        # 样本加权
        task_loss = (per_elem.mean(dim=1) * wb).mean()

        # 多任务不确定性加权
        # L = sum(exp(-s_i) * L_i + s_i)
        task_losses = per_elem.mean(dim=0)
        mt_loss = 0.0
        for j in range(3):
            mt_loss = mt_loss + torch.exp(-model.log_vars[j]) * task_losses[j] + model.log_vars[j]

        phys_loss = _pairwise_physics_loss(model, xb, pred)

        # 轻微 L2 正则，进一步抑制小样本过拟合
        l2 = pred.new_tensor(0.0)
        for p in model.parameters():
            if p.requires_grad and p.dim() > 1:
                l2 = l2 + (p ** 2).sum()

        loss = 0.35 * task_loss + 0.65 * mt_loss + 0.15 * phys_loss + 1e-5 * l2
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()