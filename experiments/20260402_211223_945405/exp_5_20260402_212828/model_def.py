import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = 18
        expert_hidden = 10
        n_experts = 2

        self.input_norm = nn.LayerNorm(input_dim)

        # 紧凑共享主干：小样本下优先控制容量
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # 轻量 gating，保留对局部突变区间的分段表达
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

        # 应变头
        self.strain_head = nn.Sequential(
            nn.Linear(hidden, 10),
            nn.SiLU(),
            nn.Linear(10, 1)
        )

        # 强度共享头
        self.strength_head = nn.Sequential(
            nn.Linear(hidden, 12),
            nn.SiLU(),
            nn.Linear(12, 10),
            nn.SiLU(),
        )
        self.tensile_out = nn.Linear(10, 1)
        self.yield_out = nn.Linear(10, 1)

        # 很轻的输出偏置，有助于小样本下校正整体保守偏差
        self.output_bias = nn.Parameter(torch.zeros(3))

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

        # 比上一版再略收一点专家残差，降低外推振荡
        h = h_shared + 0.50 * h_mix

        strain = self.strain_head(h)
        hs = self.strength_head(h)
        tensile = self.tensile_out(hs)
        yld = self.yield_out(hs)

        out = torch.cat([strain, tensile, yld], dim=-1)
        return out + self.output_bias.view(1, 3)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2.5e-3)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)
    batch_size = min(batch_size, n)

    # 保持两次遍历，兼顾收敛和稳健性
    n_passes = 2

    # 基于训练集统计构造通用尺度，避免不同任务量纲不均衡
    y_scale = y_train.std(dim=0, unbiased=False).clamp_min(1.0).detach()
    y_mean = y_train.mean(dim=0, keepdim=True).detach()

    for _ in range(n_passes):
        perm = torch.randperm(n, device=X_train.device)

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            pred = model(xb)
            err = pred - yb
            abs_err = err.detach().abs()

            # 任务归一化残差：使 hard weighting 更公平
            normed_err = abs_err / y_scale.view(1, 3)

            # 温和难例重加权：重点照顾470/12这类系统低估点，但设置上界避免爆炸
            hard_w = torch.clamp(1.0 + 0.28 * normed_err, 1.0, 1.9)

            # 维持轻度偏向强度任务，但比现版更平衡，减少 tensile/yield trade-off
            task_w = torch.tensor([1.0, 1.12, 1.10], device=yb.device, dtype=yb.dtype).view(1, 3)

            mse_loss = ((err ** 2) * hard_w * task_w).mean()

            # 用按任务尺度归一化后的 Huber，提高三任务稳定性
            pred_n = (pred - y_mean) / y_scale.view(1, 3)
            yb_n = (yb - y_mean) / y_scale.view(1, 3)
            huber_loss = F.smooth_l1_loss(pred_n, yb_n, beta=0.8)

            # 物理约束：抗拉强度 >= 屈服强度
            phy_order = F.relu(pred[:, 2] - pred[:, 1]).mean()

            # 输出耦合：抗拉-屈服间隔与真实样本保持一致，缓解局部耦合失真
            true_gap = (yb[:, 1] - yb[:, 2]).detach()
            pred_gap = pred[:, 1] - pred[:, 2]
            gap_scale = y_scale[1:].mean().detach().clamp_min(1.0)
            gap_loss = F.smooth_l1_loss(pred_gap / gap_scale, true_gap / gap_scale, beta=0.6)

            # 外推点常见问题是整体保守低估，加入轻度低估非对称惩罚
            # 仅对强度任务启用，且系数较小，避免把随机误差误当系统偏差
            under_tensile = F.relu(yb[:, 1] - pred[:, 1])
            under_yield = F.relu(yb[:, 2] - pred[:, 2])
            asym_loss = (
                0.65 * (under_tensile ** 2).mean() / (y_scale[1] ** 2) +
                0.55 * (under_yield ** 2).mean() / (y_scale[2] ** 2)
            )

            # 输出中心正则：限制小样本下过度发散，但不强压峰值
            center_reg = 8e-5 * ((pred - pred.mean(dim=0, keepdim=True)) ** 2).mean()

            # gating 熵正则：避免过早塌缩到单专家，提升分段建模泛化
            with torch.no_grad():
                x_norm = model.input_norm(xb)
            gate_logits = model.gate(x_norm)
            gate_prob = torch.softmax(gate_logits, dim=-1)
            gate_entropy = -(gate_prob * torch.log(gate_prob.clamp_min(1e-8))).sum(dim=-1).mean()

            loss = (
                mse_loss
                + 0.10 * huber_loss
                + 0.05 * phy_order
                + 0.08 * gap_loss
                + 0.06 * asym_loss
                + center_reg
                - 0.004 * gate_entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()