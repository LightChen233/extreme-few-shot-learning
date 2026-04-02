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

        # 小样本下使用轻量 MoE：
        # - shared trunk 提取温度/时间及其派生特征的公共表示
        # - gate 学习工艺区间路由，适应 460/470℃、12h 附近的突变/断层
        # - 多专家保持容量可控，避免大网络过拟合
        hidden = 24
        expert_hidden = 20
        n_experts = 3

        self.input_norm = nn.LayerNorm(input_dim)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, n_experts)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, expert_hidden),
                nn.Tanh(),
                nn.Linear(expert_hidden, 3)
            )
            for _ in range(n_experts)
        ])

        # 学习一个全局残差基线，提升外推稳定性
        self.linear_head = nn.Linear(input_dim, 3)

        # 输出尺度参数，初始较小，避免训练初期不稳定
        self.res_scale = nn.Parameter(torch.tensor(0.7))

    def forward(self, x):
        x_in = self.input_norm(x)
        feat = self.trunk(x_in)

        gate_logits = self.gate(x_in)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [B, E]

        expert_outs = torch.stack([expert(feat) for expert in self.experts], dim=1)  # [B, E, 3]
        moe_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)

        linear_out = self.linear_head(x_in)
        out = linear_out + self.res_scale * moe_out
        return out


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)


def _infer_temp_time_cols(X):
    # 在未知特征工程下，尽量稳健地从输入中识别温度/时间列：
    # 取方差前两大的列，其中较大均值者视作温度，另一列视作时间。
    with torch.no_grad():
        if X.ndim != 2 or X.shape[1] < 2:
            return None, None
        var = X.var(dim=0)
        topk = torch.topk(var, k=min(4, X.shape[1])).indices.tolist()
        if len(topk) < 2:
            return None, None
        cand = topk[:2]
        means = [X[:, i].mean().item() for i in cand]
        temp_idx = cand[0] if means[0] >= means[1] else cand[1]
        time_idx = cand[1] if temp_idx == cand[0] else cand[0]
        return temp_idx, time_idx


def _pairwise_monotonicity_loss(model, X, pred):
    # 对外推更友好的软物理约束：
    # 固溶/时效过程下，强度与塑性往往存在 trade-off。
    # 在同温条件下，时间增加时：
    # - 抗拉/屈服变化方向应尽量一致
    # - 应变与强度变化倾向相反
    # 不强行指定绝对单调，只约束局部相对变化关系。
    temp_idx, time_idx = _infer_temp_time_cols(X)
    if temp_idx is None:
        return pred.new_tensor(0.0)

    temp = X[:, temp_idx]
    time = X[:, time_idx]

    loss = pred.new_tensor(0.0)
    count = 0

    n = X.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            # 同温近邻样本
            if torch.abs(temp[i] - temp[j]) < 1e-6 and torch.abs(time[i] - time[j]) > 1e-6:
                # 按时间顺序
                if time[j] > time[i]:
                    d = pred[j] - pred[i]
                else:
                    d = pred[i] - pred[j]

                # d[1], d[2] 应尽量同号；d[0] 与强度变化尽量反号
                loss = loss + F.relu(-(d[1] * d[2])) * 0.02
                loss = loss + F.relu(d[0] * d[1]) * 0.01
                loss = loss + F.relu(d[0] * d[2]) * 0.01
                count += 1

    if count == 0:
        return pred.new_tensor(0.0)
    return loss / count


def _output_relation_loss(pred):
    # 通用材料强度关系：
    # 抗拉强度通常不低于屈服强度
    tensile = pred[:, 1]
    yld = pred[:, 2]
    return F.relu(yld - tensile).mean()


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """
    单 epoch 训练。
    策略：
    1) 小 batch + shuffle
    2) 多任务加权（按目标尺度自适应）
    3) hard example reweighting，聚焦 460/470-12h 等突变难例
    4) 轻量物理约束：UTS >= YS；局部时间演化关系约束
    """
    model.train()
    n = len(X_train)

    # 用目标方差做多任务归一化，避免 tensile 的数值范围主导优化
    with torch.no_grad():
        target_scale = y_train.std(dim=0).clamp_min(1.0)
        inv_var = 1.0 / (target_scale ** 2)

    # 先全量估计一次残差，用于 hard example 重加权
    with torch.no_grad():
        full_pred = model(X_train)
        base_err = torch.abs(full_pred - y_train)
        sample_hard = (base_err / target_scale.unsqueeze(0)).mean(dim=1)
        sample_weight = 1.0 + 1.5 * torch.tanh(sample_hard)
        sample_weight = sample_weight / sample_weight.mean()

    perm = torch.randperm(n)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        # 加权多任务 MSE
        sq = (pred - yb) ** 2
        task_loss = (sq * inv_var.unsqueeze(0)).mean(dim=1)

        # hard example 权重
        w = sample_weight[idx]
        data_loss = (task_loss * w).mean()

        # 物理关系约束
        rel_loss = _output_relation_loss(pred)

        # 在全训练集上间歇性加入局部单调/协同约束，避免 batch 内无可比样本
        mono_loss = pred.new_tensor(0.0)
        if n <= 64:
            full_pred_cur = model(X_train)
            mono_loss = _pairwise_monotonicity_loss(model, X_train, full_pred_cur)

        # gate 熵正则：避免塌缩到单一专家，也避免过度平均
        gate_logits = model.gate(model.input_norm(xb))
        gate_prob = torch.softmax(gate_logits, dim=-1)
        entropy = -(gate_prob * torch.log(gate_prob.clamp_min(1e-8))).sum(dim=1).mean()
        entropy_reg = -0.002 * entropy  # 轻微鼓励分化路由

        loss = data_loss + 0.08 * rel_loss + 0.05 * mono_loss + entropy_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()