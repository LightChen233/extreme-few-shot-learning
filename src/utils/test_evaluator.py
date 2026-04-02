import torch
import pandas as pd
import sys
import importlib.util

def evaluate_test_set(model_path, output_path):
    """评估测试集并保存预测结果"""
    from pathlib import Path

    # 加载 exp 文件夹里的 feature_agent 和 train
    exp_dir = Path(model_path).parent

    # 动态加载 feature_agent
    spec = importlib.util.spec_from_file_location("feature_agent_exp", exp_dir / 'feature_agent.py')
    feature_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_module)
    FeatureAgent = feature_module.FeatureAgent

    # 动态加载 train (Net 定义)，先把 exp_dir 加入 path 让 train.py 能找到 feature_agent
    sys.path.insert(0, str(exp_dir))
    spec = importlib.util.spec_from_file_location("train_exp", exp_dir / 'train.py')
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    Net = train_module.Net
    sys.path.pop(0)

    # 评估
    test_df = pd.read_csv('data/test.csv')
    agent = FeatureAgent()
    X_test = agent.engineer_features(test_df)

    model = Net(X_test.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_test = torch.FloatTensor(X_test)
    with torch.no_grad():
        pred = model(X_test).numpy()

    # 保存预测结果
    result_df = test_df[['temp', 'time']].copy()
    result_df['pred_strain'] = pred[:, 0]
    result_df['pred_tensile'] = pred[:, 1]
    result_df['pred_yield'] = pred[:, 2]
    result_df['true_strain'] = test_df['strain'].values
    result_df['true_tensile'] = test_df['tensile_strength'].values
    result_df['true_yield'] = test_df['yield_strength'].values
    result_df.to_csv(output_path, index=False)
