import pandas as pd
import numpy as np

class FeatureAgent:
    """特征工程 Agent - 可被外循环修改"""

    def __init__(self):
        self.feature_names = []

    def engineer_features(self, df):
        """构造特征 - 这个函数会被 Agent 修改"""
        X = df[['temp', 'time']].copy()

        # 基础特征
        X['temp_squared'] = X['temp'] ** 2
        X['time_squared'] = X['time'] ** 2
        X['temp_time'] = X['temp'] * X['time']

        # 物理启发特征 (Arrhenius-like)
        X['log_time'] = np.log(X['time'] + 1)
        X['inv_temp'] = 1 / (X['temp'] + 273.15)  # 转开尔文

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names

if __name__ == '__main__':
    agent = FeatureAgent()
    train = pd.read_csv('data/train.csv')
    X = agent.engineer_features(train)
    print(f"Features: {agent.get_feature_names()}")
    print(f"Shape: {X.shape}")
