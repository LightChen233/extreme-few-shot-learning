"""
多维度评估和反思系统
"""
import torch
import pandas as pd
import numpy as np
from train import Net
from feature_agent import FeatureAgent

def evaluate_model(model_path='model.pt'):
    """多维度评估模型"""
    # 加载数据
    val_df = pd.read_csv('data/val.csv')
    agent = FeatureAgent()
    X_val = agent.engineer_features(val_df)
    y_val = val_df[['strain', 'tensile_strength', 'yield_strength']].values

    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    # 加载模型
    model = Net(X_val.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 预测
    with torch.no_grad():
        pred = model(X_val)

    # 多维度评估
    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()

    metrics = {
        'overall_mse': mse(pred, y_val).item(),
        'overall_mae': mae(pred, y_val).item(),
        'strain_mse': mse(pred[:, 0], y_val[:, 0]).item(),
        'tensile_mse': mse(pred[:, 1], y_val[:, 1]).item(),
        'yield_mse': mse(pred[:, 2], y_val[:, 2]).item(),
        'strain_mae': mae(pred[:, 0], y_val[:, 0]).item(),
        'tensile_mae': mae(pred[:, 1], y_val[:, 1]).item(),
        'yield_mae': mae(pred[:, 2], y_val[:, 2]).item(),
    }

    # 相对误差
    for i, name in enumerate(['strain', 'tensile', 'yield']):
        rel_error = torch.abs(pred[:, i] - y_val[:, i]) / (y_val[:, i] + 1e-8)
        metrics[f'{name}_rel_error'] = rel_error.mean().item()

    return metrics

if __name__ == '__main__':
    metrics = evaluate_model()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
