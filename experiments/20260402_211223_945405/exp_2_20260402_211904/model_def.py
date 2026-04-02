"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。
train.py 是固定 runner，不要修改它。

本次改动遵循保守原则：
1) 小样本下适度降容量并加入轻量残差/归一化，提升稳定性
2) 多任务解耦：共享干 + 三个任务头，减轻应变与强度的错误耦合
3) 对强度任务轻微增权，缓解当前系统性低估抗拉/屈服的问题
4) 加入输出物理关系约束：抗拉强度 >= 屈服强度
5) 对 hard example 使用有上界的动态加权，避免训练被少数点完全主导
6) 使用 weight decay + gradient clipping 保证数值稳定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = 32

        self.input_norm = nn.LayerNorm(input_dim)

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.strain_head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
        self.tensile_head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
        self.yield_head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.input_norm(x)
        h = self.shared(x)
        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    batch_size = min(batch_size, n)

    # 小样本下多看几遍数据，但保持单次更新稳定
    n_passes = 2

    for _ in range(n_passes):
        perm = torch.randperm(n, device=X_train.device)

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            pred = model(xb)
            err = pred - yb
            abs_err = err.detach().abs()

            # 以 batch 内相对误差构造 hard-example 权重，限制上界避免爆炸
            normed = abs_err / (abs_err.mean(dim=0, keepdim=True) + 1e-6)
            hard_w = torch.clamp(1.0 + 0.3 * normed, 1.0, 2.0)

            # 任务权重：轻微提高强度任务权重，避免再次过度偏向应变
            task_w = torch.tensor([1.0, 1.2, 1.2], device=yb.device, dtype=yb.dtype).view(1, 3)

            mse = (err ** 2) * hard_w * task_w
            data_loss = mse.mean()

            # 物理关系约束：抗拉强度 >= 屈服强度
            phy_loss = F.relu(pred[:, 2] - pred[:, 1]).mean()

            loss = data_loss + 0.05 * phy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()