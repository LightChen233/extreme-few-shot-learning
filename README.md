# AutoResearch - 极端少样本学习框架

零配置自动化研究系统，适用于任何极端少样本场景。

## 快速开始

```bash
python3 run.py
```

## 项目结构

```
├── run.py                 # 统一入口
├── data/                  # 数据集
├── prompts/              # 所有 prompt 配置
│   ├── feature_agent.txt
│   ├── program_agent.txt
│   ├── domain_knowledge.txt
│   └── reflection.txt
├── src/
│   ├── agents/           # AI Agents
│   │   ├── zero_config_framework.py
│   │   ├── universal_framework.py
│   │   └── reflection.py
│   ├── models/           # 模型和训练
│   │   ├── train.py
│   │   ├── feature_agent.py
│   │   └── model_zoo.py
│   └── utils/            # 工具函数
│       ├── experiment_tracker.py
│       ├── domain_config.py
│       └── evaluator.py
└── experiments/          # 实验日志（自动生成）
```

## 特性

- ✓ 零配置 - 只需数据+任务描述
- ✓ 多维度评估 - 9个指标
- ✓ 反思系统 - 自动分析改进
- ✓ 完整追踪 - 保存所有实验
- ✓ 灵活架构 - Transformer/ResNet/MultiTask
