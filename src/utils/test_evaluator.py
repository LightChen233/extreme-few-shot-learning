import torch
import pandas as pd
import sys
import importlib.util
from src.utils.config_loader import Config

def evaluate_test_set(model_path, output_path):
    """评估测试集并保存预测结果（列名全从 config 读，通用）"""
    from pathlib import Path

    config = Config('config.yaml')
    input_cols  = config.input_cols
    target_cols = config.target_cols
    test_path   = config.test_path

    exp_dir = Path(model_path).parent

    # 动态加载 feature_agent
    spec = importlib.util.spec_from_file_location("feature_agent_exp", exp_dir / 'feature_agent.py')
    feature_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_module)
    FeatureAgent = feature_module.FeatureAgent

    # 优先从 model_def.py 加载 Net，回退到 train.py
    sys.path.insert(0, str(exp_dir))
    model_def_path = exp_dir / 'model_def.py'
    src_path = model_def_path if model_def_path.exists() else exp_dir / 'train.py'
    mod_name = "model_def_exp" if model_def_path.exists() else "train_exp"
    spec = importlib.util.spec_from_file_location(mod_name, src_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    Net = model_module.Net
    sys.path.pop(0)

    test_df = pd.read_csv(test_path)
    agent   = FeatureAgent()
    X_test  = agent.engineer_features(test_df)

    model = Net(X_test.shape[1])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    X_test = torch.FloatTensor(X_test)
    with torch.no_grad():
        pred = model(X_test).numpy()

    result_df = test_df[input_cols].copy()
    for i, col in enumerate(target_cols):
        result_df[f'pred_{col}'] = pred[:, i]
        result_df[f'true_{col}'] = test_df[col].values
    result_df.to_csv(output_path, index=False)
