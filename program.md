# Program Agent Instructions

## 目标
优化神经网络架构和超参数，最小化验证集损失 (Val Loss)。

## 可修改内容
在 `train.py` 中：
- 网络架构 (层数、宽度、激活函数)
- 学习率 (lr)
- Batch size
- 优化器类型

## 约束
- 训练时间固定 5 分钟
- 输入维度由 Feature Agent 决定
- 输出维度固定为 3 (strain, tensile_strength, yield_strength)

## 建议尝试
1. 调整隐藏层大小 (32, 64, 128)
2. 增加/减少层数
3. 尝试不同激活函数 (ReLU, Tanh, LeakyReLU)
4. 调整学习率 (0.0001 - 0.01)
5. 添加 Dropout 防止过拟合
