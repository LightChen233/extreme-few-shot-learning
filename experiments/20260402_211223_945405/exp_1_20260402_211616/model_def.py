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
        # 小样本场景：采用较小容量 + 共享干路 + 多任务头
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.head_strain = nn.Linear(16, 1)
        self.head_tensile = nn.Linear(16, 1)
        self.head_yield = nn.Linear(16, 1)

        # 多任务不确定性加权参数
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        h = self.backbone(x)
        strain = self.head_strain(h)
        tensile = self.head_tensile(h)
        yld = self.head_yield(h)
        return torch.cat([strain, tensile, yld], dim=1)


def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练：稳健回归 + 难例重加权 + 物理约束 + 梯度裁剪"""
    model.train()
    n = len(X_train)
    if n == 0:
        return

    # 小数据集下稍小 batch 更利于优化
    batch_size = min(6, n)

    # 先打乱
    perm = torch.randperm(n, device=X_train.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad()
        pred = model(xb)

        # --- 基础多任务误差：SmoothL1 比 MSE 对异常点更稳 ---
        per_elem = F.smooth_l1_loss(pred, yb, reduction="none", beta=1.0)  # [B, 3]

        # --- 难例重加权：针对高误差突变区，但限制上界避免不稳定 ---
        with torch.no_grad():
            abs_err = (pred - yb).abs()
            norm_err = abs_err / (yb.abs() + 1.0)
            sample_score = norm_err.mean(dim=1, keepdim=True)  # [B,1]
            sample_w = (1.0 + 1.5 * sample_score).clamp(1.0, 2.5)

        weighted_task_loss = (per_elem * sample_w).mean(dim=0)  # [3]

        # --- 多任务不确定性加权，做 clamp 保证数值稳定 ---
        log_vars = torch.clamp(model.log_vars, min=-3.0, max=3.0)
        precisions = torch.exp(-log_vars).clamp(0.05, 20.0)
        data_loss = (precisions * weighted_task_loss + log_vars).sum()

        # --- 物理软约束 ---
        strain_pred = pred[:, 0]
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 屈服强度通常不应高于抗拉强度
        order_loss = F.relu(yield_pred - tensile_pred).mean()

        # 训练集内部近邻单调/平滑约束：
        # 对相同时间下，温度相近样本不应出现过于剧烈振荡；
        # 对相同温度下，时间相近样本同理。
        # 这里只做输出平滑，不对输入求导，保证稳定。
        smooth_loss = torch.tensor(0.0, device=X_train.device)

        if n >= 4:
            with torch.no_grad():
                # 利用前两维通常是主要工艺变量；若 feature_agent 添加更多特征，此约束仍只是一种弱正则
                x_ref = X_train[:, :min(2, X_train.shape[1])]
                dist = torch.cdist(x_ref, x_ref, p=1)  # [N,N]
                mask = (dist > 0) & (dist < dist.mean().clamp_min(1e-6))
                pairs = mask.nonzero(as_tuple=False)

            if len(pairs) > 0:
                pairs = pairs[: min(16, len(pairs))]
                p1 = model(X_train[pairs[:, 0]])
                p2 = model(X_train[pairs[:, 1]])
                d_out = (p1 - p2).abs()
                smooth_loss = d_out.mean()

        # 总损失：以数据拟合为主，约束为辅
        loss = data_loss + 0.2 * order_loss + 0.03 * smooth_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()