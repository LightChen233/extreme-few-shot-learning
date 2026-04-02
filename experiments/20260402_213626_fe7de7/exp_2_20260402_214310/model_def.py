"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。
train.py 是固定 runner，不要修改它。

本次改动遵循“保守改进原则”：
1) 不重写为复杂 MoE，仅做轻量多任务共享干路 + task head
2) 加入小幅正则/Dropout，降低小样本过拟合
3) 用温和的 hard-example 重加权（有上界）
4) 加入通用物理约束：屈服强度不应高于抗拉强度
5) 梯度裁剪保证稳定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 小样本场景下，适度缩小容量，保留一定非线性与多任务表达
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(48, 24),
            nn.ReLU(),
        )

        # 三个任务头，减少目标间负迁移
        self.strain_head = nn.Linear(24, 1)
        self.tensile_head = nn.Linear(24, 1)
        self.yield_head = nn.Linear(24, 1)

    def forward(self, x):
        h = self.backbone(x)
        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    # 历史提示不宜大改；仅做温和 lr + weight decay 微调
    return torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练，可修改 loss、batch size、采样策略等"""
    model.train()
    perm = torch.randperm(len(X_train), device=X_train.device)

    for i in range(0, len(X_train), batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 基础多任务 MSE
        per_sample_per_task = (pred - yb) ** 2  # [B, 3]

        # 轻量 hard-example 重加权：
        # 基于样本平均残差构造权重，强调突变/外推难例，但严格限制上界
        with torch.no_grad():
            abs_res = (pred.detach() - yb).abs().mean(dim=1)  # [B]
            scale = abs_res.mean().clamp_min(1e-6)
            sample_w = 1.0 + 0.8 * (abs_res / scale)
            sample_w = sample_w.clamp(1.0, 2.5)  # 必须有上界，防止不稳定

        data_loss = (per_sample_per_task.mean(dim=1) * sample_w).mean()

        # 通用物理约束：屈服强度 <= 抗拉强度
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]
        phys_loss = F.relu(yield_pred - tensile_pred).mean()

        # 轻度目标平衡：强度任务相对更难，但避免过强干预
        task_weights = torch.tensor([1.0, 1.1, 1.1], device=pred.device, dtype=pred.dtype)
        balanced_loss = (per_sample_per_task.mean(dim=0) * task_weights).mean()

        loss = 0.6 * data_loss + 0.4 * balanced_loss + 0.08 * phys_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()