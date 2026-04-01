import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from feature_agent import FeatureAgent
import time

# 模型定义 - Agent 可修改
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 预测 strain, tensile_strength, yield_strength
        )

    def forward(self, x):
        return self.net(x)

# 超参数 - Agent 可修改
lr = 0.001
batch_size = 8

def train_model(time_limit=300):
    """训练模型，固定时间预算"""
    # 加载数据
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    # 特征工程
    agent = FeatureAgent()
    X_train = agent.engineer_features(train_df)
    y_train = train_df[['strain', 'tensile_strength', 'yield_strength']].values
    X_val = agent.engineer_features(val_df)
    y_val = val_df[['strain', 'tensile_strength', 'yield_strength']].values

    # 转 tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    # 模型
    model = Net(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练循环
    start_time = time.time()
    epoch = 0
    while time.time() - start_time < time_limit - 5:
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            if time.time() - start_time >= time_limit - 5:
                break
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        epoch += 1

    # 评估
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()

    print(f"Epochs: {epoch}, Val Loss: {val_loss:.4f}")
    return val_loss, model

if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')
