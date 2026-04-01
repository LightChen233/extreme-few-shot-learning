"""
AutoResearch - 极端少样本学习
"""
from src.agents.autoresearch import ZeroConfigAutoResearch
from src.utils.config_loader import Config

# 加载配置
config = Config('config.yaml')

# 创建研究实例
research = ZeroConfigAutoResearch(
    data_path=config.get('data.train_path'),
    task_description=config.get('data.task_description')
)

# 运行自动优化
research.run(n_iterations=config.get('training.n_iterations', 5))
