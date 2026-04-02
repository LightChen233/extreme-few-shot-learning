"""
固定训练 runner — 框架使用，LLM 不可修改。
所有评估打印格式、VAL_PRED、model.pt 保存逻辑都在这里。
LLM 只修改 model_def.py。
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from feature_agent import FeatureAgent
from model_def import Net, build_optimizer, train_step
import time

def train_model(time_limit=300):
    train_df = pd.read_csv('data/train.csv')
    val_df   = pd.read_csv('data/val.csv')

    agent   = FeatureAgent()
    X_train = agent.engineer_features(train_df)
    y_train = train_df[['strain', 'tensile_strength', 'yield_strength']].values
    X_val   = agent.engineer_features(val_df)
    y_val   = val_df[['strain', 'tensile_strength', 'yield_strength']].values

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val   = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    model     = Net(X_train.shape[1])
    optimizer = build_optimizer(model)
    criterion = nn.MSELoss()
    mae_fn    = nn.L1Loss()

    start_time = time.time()
    epoch = 0
    patience_counter = 0
    best_score = float('inf')
    best_state = None

    while time.time() - start_time < time_limit - 3:
        model.train()
        train_step(model, optimizer, X_train, y_train)
        epoch += 1

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_val  = model(X_val)
                val_mse   = criterion(pred_val, y_val_t).item()
                val_score = (
                    ((pred_val[:, 0] - y_val_t[:, 0]).abs() / (y_val_t[:, 0].abs() + 1e-6)).mean() * 2.0 +
                    ((pred_val[:, 1] - y_val_t[:, 1]).abs() / (y_val_t[:, 1].abs() + 1e-6)).mean() * 1.0 +
                    ((pred_val[:, 2] - y_val_t[:, 2]).abs() / (y_val_t[:, 2].abs() + 1e-6)).mean() * 1.0
                ).item()
                elapsed = time.time() - start_time
                print(f"[epoch {epoch:5d} | {elapsed:5.0f}s] val_mse={val_mse:.2f} val_score={val_score:.4f} patience={patience_counter}", flush=True)

                if val_score < best_score:
                    best_score = val_score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 50 and elapsed > 30:
                    print(f"Early stop at epoch {epoch}", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 固定评估 & 打印格式（框架解析依赖这些，不可改）
    model.eval()
    with torch.no_grad():
        pred        = model(X_val)
        overall_mse = criterion(pred, y_val_t).item()
        strain_mse  = criterion(pred[:, 0], y_val_t[:, 0]).item()
        tensile_mse = criterion(pred[:, 1], y_val_t[:, 1]).item()
        yield_mse   = criterion(pred[:, 2], y_val_t[:, 2]).item()
        strain_mae  = mae_fn(pred[:, 0], y_val_t[:, 0]).item()
        tensile_mae = mae_fn(pred[:, 1], y_val_t[:, 1]).item()
        yield_mae   = mae_fn(pred[:, 2], y_val_t[:, 2]).item()
        strain_rel  = (torch.abs(pred[:, 0] - y_val_t[:, 0]) / (y_val_t[:, 0].abs() + 1e-8)).mean().item()
        tensile_rel = (torch.abs(pred[:, 1] - y_val_t[:, 1]) / (y_val_t[:, 1].abs() + 1e-8)).mean().item()
        yield_rel   = (torch.abs(pred[:, 2] - y_val_t[:, 2]) / (y_val_t[:, 2].abs() + 1e-8)).mean().item()

    print(f"Epochs: {epoch}")
    print(f"Val Loss: {overall_mse:.4f}")
    print(f"METRICS strain_mse={strain_mse:.4f} tensile_mse={tensile_mse:.4f} yield_mse={yield_mse:.4f}")
    print(f"METRICS strain_mae={strain_mae:.4f} tensile_mae={tensile_mae:.4f} yield_mae={yield_mae:.4f}")
    print(f"METRICS strain_rel={strain_rel:.4f} tensile_rel={tensile_rel:.4f} yield_rel={yield_rel:.4f}")

    pred_np = pred.numpy()
    for j in range(len(y_val)):
        row = val_df.iloc[j]
        print(f"VAL_PRED temp={row['temp']} time={row['time']} "
              f"strain_err={y_val[j,0]-pred_np[j,0]:.4f} "
              f"tensile_err={y_val[j,1]-pred_np[j,1]:.4f} "
              f"yield_err={y_val[j,2]-pred_np[j,2]:.4f}")

    return overall_mse, model


if __name__ == '__main__':
    val_loss, model = train_model(time_limit=300)
    torch.save(model.state_dict(), 'model.pt')
    print("MODEL_SAVED model.pt")

    test_df = pd.read_csv('data/test.csv')
    agent   = FeatureAgent()
    X_test  = agent.engineer_features(test_df)
    y_test  = test_df[['strain', 'tensile_strength', 'yield_strength']].values
    X_test  = torch.FloatTensor(X_test)

    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy()

    # 检查是否坍塌（所有预测几乎相同）
    pred_std = pred.std(axis=0)
    if pred_std.max() < 1e-3:
        print(f"[Warning] 模型预测坍塌：所有样本预测几乎相同 (std={pred_std})", flush=True)

    result_df = test_df[['temp', 'time']].copy()
    result_df['pred_strain']  = pred[:, 0]
    result_df['pred_tensile'] = pred[:, 1]
    result_df['pred_yield']   = pred[:, 2]
    result_df['true_strain']  = y_test[:, 0]
    result_df['true_tensile'] = y_test[:, 1]
    result_df['true_yield']   = y_test[:, 2]
    result_df.to_csv('test_predictions.csv', index=False)
    print("TEST_PREDICTIONS_SAVED test_predictions.csv")
