# Feature Agent Instructions

## 目标
从原始特征 (temp, time) 构造有意义的派生特征，提升模型性能。

## 可修改内容
在 `feature_agent.py` 的 `engineer_features()` 方法中添加特征。

## 建议尝试
1. 多项式特征 (temp^2, temp^3, time^2)
2. 交互特征 (temp*time, temp/time)
3. 物理启发特征 (Arrhenius: exp(-1/T), log(time))
4. 归一化/标准化

## 当前特征
- temp, time (原始)
- temp_squared, time_squared
- temp_time (交互)
- log_time, inv_temp (物理)
