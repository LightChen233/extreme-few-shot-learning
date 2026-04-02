"""
AutoResearch - 极端少样本学习
"""
import shutil
from src.agents.autoresearch import ZeroConfigAutoResearch
from src.utils.config_loader import Config

# ── 固定起点：每次从 seed/ 目录恢复，保证公平对比 ──
print("[run] restoring seed files...")
shutil.copy('seed/feature_agent.py', 'src/models/feature_agent.py')
shutil.copy('seed/train.py',         'src/models/train.py')
shutil.copy('seed/model_def.py',     'src/models/model_def.py')

# 加载配置
config = Config('config.yaml')

# 创建研究实例
research = ZeroConfigAutoResearch(
    data_path=config.get('data.train_path'),
    task_description=config.get('data.task_description')
)

# 运行自动优化
research.run(n_iterations=config.get('training.n_iterations', 5))
