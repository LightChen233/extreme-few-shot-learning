import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 保守改动：
        # 1) 维持小模型容量，避免 29 条样本过拟合
        # 2) 在共享主干外，增加一个极轻量“高温/长时”门控分支，
        #    用于缓解 440/470 外推点的局部断层/斜率失真
        # 3) 继续使用 tensile = yield + softplus(delta) 的物理参数化

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.SiLU(),
            nn.Linear(24, 12),
            nn.SiLU(),
        )

        # 主专家：学习整体平滑趋势
        self.main_proj = nn.Sequential(
            nn.Linear(12, 10),
            nn.SiLU(),
        )

        # 轻量校正专家：只负责小幅修正，避免 MoE 过强导致不稳
        self.corr_proj = nn.Sequential(
            nn.Linear(12, 8),
            nn.SiLU(),
        )

        # 门控网络：输入级轻量路由
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.Tanh(),
            nn.Linear(6, 1),
            nn.Sigmoid(),
        )

        # 主输出头
        self.strain_head_main = nn.Linear(10, 1)
        self.yield_head_main = nn.Linear(10, 1)
        self.delta_head_main = nn.Linear(10, 1)

        # 校正头（残差幅度受限）
        self.strain_head_corr = nn.Linear(8, 1)
        self.yield_head_corr = nn.Linear(8, 1)
        self.delta_head_corr = nn.Linear(8, 1)

    def forward(self, x):
        h = self.backbone(x)
        h_main = self.main_proj(h)
        h_corr = self.corr_proj(h)

        gate = self.gate(x)  # [B,1], 轻量路由

        # 主趋势
        strain_main = self.strain_head_main(h_main)
        yld_main = self.yield_head_main(h_main)
        delta_main = self.delta_head_main(h_main)

        # 小幅校正：用 tanh 控制残差，防止在小样本下过度修正
        strain_corr = 0.8 * torch.tanh(self.strain_head_corr(h_corr))
        yld_corr = 8.0 * torch.tanh(self.yield_head_corr(h_corr))
        delta_corr = 8.0 * torch.tanh(self.delta_head_corr(h_corr))

        strain = strain_main + gate * strain_corr
        yld = yld_main + gate * yld_corr
        delta = F.softplus(delta_main + gate * delta_corr)
        tensile = yld + delta

        return torch.cat([strain, tensile, yld], dim=-1)


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=4.0e-4,
        weight_decay=2.0e-4
    )


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    model.train()
    n = len(X_train)
    device = X_train.device
    dtype = X_train.dtype

    # Huber 保持稳健；对屈服略增权，修正上一版 yield trade-off
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)
    task_weights = torch.tensor([1.0, 1.10, 1.18], device=device, dtype=dtype)

    # 每个 epoch 打乱
    perm = torch.randperm(n, device=device)

    # 用当前模型全量前向，构造“有上界”的难例重加权
    # 仅小幅度提升大误差样本权重，避免像历史失败版本那样放大小样本方差
    with torch.no_grad():
        pred_all = model(X_train)
        abs_err = (pred_all - y_train).abs()
        sample_err = (abs_err * task_weights).mean(dim=1)  # [N]
        norm_err = sample_err / (sample_err.mean() + 1e-6)
        sample_weights_all = torch.clamp(0.85 + 0.35 * norm_err, 0.85, 1.6)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]
        sw = sample_weights_all[idx].detach()

        pred = model(xb)
        per_elem = criterion(pred, yb)  # [B,3]
        per_sample = (per_elem * task_weights).mean(dim=1)
        base_loss = (per_sample * sw).mean()

        strain_pred = pred[:, 0]
        tensile_pred = pred[:, 1]
        yield_pred = pred[:, 2]

        # 顺序约束（理论上结构已保证，但保留极轻权重）
        order_penalty = F.relu(yield_pred - tensile_pred).mean()

        # 局部平滑：相近样本预测不应剧烈跳变
        if len(xb) >= 2:
            pair_idx = torch.roll(torch.arange(len(xb), device=device), shifts=1)
            x_dist = (xb - xb[pair_idx]).pow(2).mean(dim=1)

            t_diff = (tensile_pred - tensile_pred[pair_idx]).pow(2)
            y_diff = (yield_pred - yield_pred[pair_idx]).pow(2)
            s_diff = (strain_pred - strain_pred[pair_idx]).pow(2)

            smooth_weight = torch.exp(-0.6 * x_dist).detach()
            smooth_penalty = (
                smooth_weight * (0.45 * t_diff + 0.40 * y_diff + 0.15 * s_diff)
            ).mean()
        else:
            smooth_penalty = torch.zeros((), device=device, dtype=dtype)

        # 多任务耦合先验：抗拉与屈服通常协同变化，约束二者相对排序外的方向一致性
        # 不强行绑定数值，仅抑制“抗拉剧烈变而屈服完全反向”的不稳定情况
        if len(xb) >= 2:
            pair_idx = torch.roll(torch.arange(len(xb), device=device), shifts=1)
            dt = tensile_pred - tensile_pred[pair_idx]
            dy = yield_pred - yield_pred[pair_idx]
            coupling_penalty = F.relu(-(dt * dy)).mean()
        else:
            coupling_penalty = torch.zeros((), device=device, dtype=dtype)

        loss = (
            base_loss
            + 0.015 * order_penalty
            + 0.005 * smooth_penalty
            + 0.004 * coupling_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()