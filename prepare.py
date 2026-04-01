import pandas as pd
import numpy as np
from pathlib import Path
import re

def parse_condition(cond_str):
    """解析工艺条件，如 '420℃-1h' -> (420, 1)"""
    match = re.search(r'(\d+)℃-(\d+)h', str(cond_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def load_data(filepath='data.xlsx'):
    """加载并清理数据"""
    df = pd.read_excel(filepath, header=None)

    data_rows = []
    current_temp, current_time = None, None

    for idx, row in df.iterrows():
        # 检查是否是工艺参数行
        if pd.notna(row[1]):
            temp, time = parse_condition(row[1])
            if temp and time:
                current_temp, current_time = temp, time

        # 提取数据行
        if current_temp and pd.notna(row[2]) and str(row[2]).replace('.','').isdigit():
            data_rows.append({
                'temp': current_temp,
                'time': current_time,
                'strain': float(row[2]),
                'tensile_strength': float(row[3]),
                'yield_strength': float(row[4])
            })

    return pd.DataFrame(data_rows)

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """划分数据集"""
    n = len(df)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[indices[:train_end]].reset_index(drop=True)
    val_df = df.iloc[indices[train_end:val_end]].reset_index(drop=True)
    test_df = df.iloc[indices[val_end:]].reset_index(drop=True)

    return train_df, val_df, test_df

if __name__ == '__main__':
    np.random.seed(42)
    df = load_data()
    print(f"Total samples: {len(df)}")
    print(f"\nData preview:\n{df.head()}")

    train, val, test = split_data(df)
    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    # 保存
    Path('data').mkdir(exist_ok=True)
    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    print("\n✓ Data saved to data/")
