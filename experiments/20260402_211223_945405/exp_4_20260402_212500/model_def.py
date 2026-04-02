import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = 20
        expert_hidden = 12
        n_experts = 2

        self.input_norm = nn.LayerNorm(input_dim)

        # 小样本下继续保持紧凑主干
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # 轻量 gating：保留分段建模能力，应对 470/12 等局部突变
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.SiLU(),
            nn.Linear(8, n_experts)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, expert_hidden),
                nn.SiLU(),
                nn.Linear(expert_hidden, hidden),
                nn.SiLU(),
            )
            for _ in range(n_experts)
        ])

        # 任务头：强度任务共享更多信息，减轻 tensile / yield 彼此割裂
        self.strain_head = nn.Sequential(
            nn.Linear(hidden, 10),
            nn.SiLU(),
            nn.Linear(10, 1)
        )

        self.strength_head = nn.Sequential(
            nn.Linear(hidden, 12),
            nn.SiLU(),
            nn.Linear(12, 10),
            nn.SiLU(),
        )
        self.tensile_out = nn.Linear(10, 1)
        self.yield_out = nn.Linear(10, 1)

    def forward(self, x):
        x_norm = self.input_norm(x)
        h_shared = self.shared(x_norm)

        gate_logits = self.gate(x_norm)
        gate_w = torch.softmax(gate_logits, dim=-1)

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(h_shared).unsqueeze(1))
        expert_outs = torch.cat(expert_outs, dim=1)

        h_mix = (expert_outs * gate_w.unsqueeze(-1)).sum(dim=1)

        # 略降低专家修正幅度，减少外推时振荡
        h = h_shared + 0.55 * h_mix

        strain = self.strain_head(h)

        hs = self.strength_head(h)
        tensile = self.tensile_out(hs)
        yld = self.yield_out(hs)

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2.0e-3)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)
    batch_size = min(batch_size, n)

    # 保持适度重复遍历，但不过分强化难例噪声
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

            # 难例重加权：比上一版更温和，减少极少数外推点牵着整体走
            normed = abs_err / (abs_err.mean(dim=0, keepdim=True) + 1e-6)
            hard_w = torch.clamp(1.0 + 0.22 * normed, 1.0, 1.8)

            # 继续轻度偏向强度任务，但减小幅度，避免 trade-off
            task_w = torch.tensor([1.0, 1.18, 1.15], device=yb.device, dtype=yb.dtype).view(1, 3)

            mse_loss = ((err ** 2) * hard_w * task_w).mean()
            huber_loss = F.smooth_l1_loss(pred, yb, beta=10.0)

            # 通用物理约束：抗拉强度 >= 屈服强度
            phy_order = F.relu(pred[:, 2] - pred[:, 1]).mean()

            # 输出间耦合约束：抗拉-屈服间隔不应无端剧烈波动
            true_gap = (yb[:, 1] - yb[:, 2]).detach()
            pred_gap = pred[:, 1] - pred[:, 2]
            gap_loss = F.smooth_l1_loss(pred_gap, true_gap, beta=8.0)

            # 温和的输出尺度正则，抑制外推点过分保守/过分发散
            pred_center = pred - pred.mean(dim=0, keepdim=True)
            center_reg = 1e-4 * (pred_center ** 2).mean()

            loss = (
                mse_loss
                + 0.12 * huber_loss
                + 0.06 * phy_order
                + 0.08 * gap_loss
                + center_reg
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()