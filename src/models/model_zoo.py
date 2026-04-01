"""
灵活的模型架构库 - 供 Program Agent 选择
"""
import torch
import torch.nn as nn

# 1. Transformer 架构
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.embed = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)  # [B, 1, 64]
        x = self.transformer(x).squeeze(1)
        return self.fc(x)

# 2. 残差网络
class ResNetModel(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        identity = x
        x = self.relu(self.fc2(self.relu(x)))
        x = x + identity  # 残差连接
        x = self.relu(self.fc3(x))
        return self.out(x)

# 3. 多任务学习（共享+专用层）
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        # 共享编码器
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # 每个任务的专用头
        self.head_strain = nn.Linear(64, 1)
        self.head_tensile = nn.Linear(64, 1)
        self.head_yield = nn.Linear(64, 1)

    def forward(self, x):
        shared = self.shared(x)
        return torch.cat([
            self.head_strain(shared),
            self.head_tensile(shared),
            self.head_yield(shared)
        ], dim=1)

# 4. 注意力机制
class AttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x).unsqueeze(1)
        x, _ = self.attention(x, x, x)
        return self.fc2(x.squeeze(1))
