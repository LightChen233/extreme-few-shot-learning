import pandas as pd
import numpy as np


class FeatureAgent:
    """
    通用特征工程基类（seed 版本）。
    只做最基础的变换，不含任何领域假设。
    由 LLM agent 在此基础上进化。
    """

    def __init__(self):
        self.feature_names = []

    def engineer_features(self, df):
        cols = {}

        input_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                      if c not in ('strain', 'tensile_strength', 'yield_strength')]

        # 基础变换
        for col in input_cols:
            x = df[col].astype(float).values
            cols[col] = x
            cols[f'{col}_sq'] = x ** 2
            cols[f'log1p_{col}'] = np.log1p(np.clip(x, 0, None))
            cols[f'sqrt_{col}'] = np.sqrt(np.clip(x, 0, None))

        # 两两交互
        for i, a in enumerate(input_cols):
            for b in input_cols[i+1:]:
                xa = df[a].astype(float).values
                xb = df[b].astype(float).values
                cols[f'{a}_x_{b}'] = xa * xb

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names
