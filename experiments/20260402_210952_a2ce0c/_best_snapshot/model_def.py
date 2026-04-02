"""
模型定义 — LLM 唯一可修改的文件。
包含：Net 架构、build_optimizer、train_step（单 epoch 训练逻辑）。
train.py 是固定 runner，不要修改它。
"""
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)


def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def train_step(model, optimizer, X_train, y_train, batch_size=8):
    """单 epoch 训练，可修改 loss、batch size、采样策略等"""
    criterion = nn.MSELoss()
    perm = torch.randperm(len(X_train))
    for i in range(0, len(X_train), batch_size):
        idx = perm[i:i + batch_size]
        optimizer.zero_grad()
        criterion(model(X_train[idx]), y_train[idx]).backward()
        optimizer.step()
