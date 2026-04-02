"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。

本次改动遵循“渐进式优化”原则，针对当前误差特点做小幅但有针对性的调整：
1) 保持小模型，避免 29 条样本下过拟合
2) 在共享干之上加入“轻量分段/专家混合”结构，专门缓解 460~470℃、长时附近的局部非线性/突变
3) 继续保留多任务解耦，避免应变与强度错误耦合
4) 对强度任务轻微增权，并用有上界的 hard-example 重加权，聚焦系统性低估区域
5) 加入通用物理约束：抗拉强度 >= 屈服强度
6) 保持数值稳定：AdamW、梯度裁剪、无高阶梯度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = 24
        expert_hidden = 16
        n_experts = 2

        self.input_norm = nn.LayerNorm(input_dim)

        # 共享主干：保持小容量
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # 轻量 gating：学习不同工艺区间的软路由
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.SiLU(),
            nn.Linear(8, n_experts)
        )

        # 专家层：每个专家都只做小修正，降低过拟合风险
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, expert_hidden),
                nn.SiLU(),
                nn.Linear(expert_hidden, hidden),
                nn.SiLU(),
            )
            for _ in range(n_experts)
        ])

        # 任务头
        self.strain_head = nn.Sequential(
            nn.Linear(hidden, 12),
            nn.SiLU(),
            nn.Linear(12, 1)
        )
        self.tensile_head = nn.Sequential(
            nn.Linear(hidden, 12),
            nn.SiLU(),
            nn.Linear(12, 1)
        )
        self.yield_head = nn.Sequential(
            nn.Linear(hidden, 12),
            nn.SiLU(),
            nn.Linear(12, 1)
        )

    def forward(self, x):
        x_norm = self.input_norm(x)
        h_shared = self.shared(x_norm)

        gate_logits = self.gate(x_norm)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [B, E]

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(h_shared).unsqueeze(1))  # [B,1,H]
        expert_outs = torch.cat(expert_outs, dim=1)  # [B,E,H]

        h_mix = (expert_outs * gate_w.unsqueeze(-1)).sum(dim=1)

        # 残差式融合：专家只修正共享表示
        h = h_shared + 0.7 * h_mix

        strain = self.strain_head(h)
        tensile = self.tensile_head(h)
        yld = self.yield_head(h)
        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1.5e-3)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练"""
    model.train()
    n = len(X_train)
    batch_size = min(batch_size, n)

    # 小样本下适度增加遍历次数
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

            # batch 内 hard-example 重加权，限制上界，避免少数点主导
            normed = abs_err / (abs_err.mean(dim=0, keepdim=True) + 1e-6)
            hard_w = torch.clamp(1.0 + 0.35 * normed, 1.0, 2.2)

            # 轻微提高强度任务权重，缓解当前系统性低估
            task_w = torch.tensor([1.0, 1.25, 1.2], device=yb.device, dtype=yb.dtype).view(1, 3)

            mse = (err ** 2) * hard_w * task_w
            data_loss = mse.mean()

            # 物理关系约束：抗拉强度 >= 屈服强度
            phy_loss = F.relu(pred[:, 2] - pred[:, 1]).mean()

            # 温和的中心化稳健项：降低极端残差对训练的不稳定影响，但不替代 MSE
            huber_loss = F.smooth_l1_loss(pred, yb, beta=8.0)

            loss = data_loss + 0.05 * phy_loss + 0.10 * huber_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()