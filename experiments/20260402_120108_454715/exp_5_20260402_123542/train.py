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
    while time.time() - start_time < time_limit:
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        epoch += 1

    # 评估 - 多维度
    model.eval()
    mae = nn.L1Loss()
    with torch.no_grad():
        pred = model(X_val)
        overall_mse = criterion(pred, y_val).item()
        strain_mse  = criterion(pred[:, 0], y_val[:, 0]).item()
        tensile_mse = criterion(pred[:, 1], y_val[:, 1]).item()
        yield_mse   = criterion(pred[:, 2], y_val[:, 2]).item()
        strain_mae  = mae(pred[:, 0], y_val[:, 0]).item()
        tensile_mae = mae(pred[:, 1], y_val[:, 1]).item()
        yield_mae   = mae(pred[:, 2], y_val[:, 2]).item()
        strain_rel  = (torch.abs(pred[:, 0] - y_val[:, 0]) / (y_val[:, 0].abs() + 1e-8)).mean().item()
        tensile_rel = (torch.abs(pred[:, 1] - y_val[:, 1]) / (y_val[:, 1].abs() + 1e-8)).mean().item()
        yield_rel   = (torch.abs(pred[:, 2] - y_val[:, 2]) / (y_val[:, 2].abs() + 1e-8)).mean().item()

    print(f"Epochs: {epoch}")
    print(f"Val Loss: {overall_mse:.4f}")
    print(f"METRICS strain_mse={strain_mse:.4f} tensile_mse={tensile_mse:.4f} yield_mse={yield_mse:.4f}")
    print(f"METRICS strain_mae={strain_mae:.4f} tensile_mae={tensile_mae:.4f} yield_mae={yield_mae:.4f}")
    print(f"METRICS strain_rel={strain_rel:.4f} tensile_rel={tensile_rel:.4f} yield_rel={yield_rel:.4f}")
    return overall_mse, model

if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')
    print("MODEL_SAVED model.pt")
