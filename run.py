"""
AutoResearch - 极端少样本学习
"""
import subprocess
from src.agents.autoresearch import ZeroConfigAutoResearch
from src.utils.config_loader import Config

# ── 固定起点：每次从同一个 seed commit 开始，保证公平对比 ──
SEED_COMMIT = "0bd2920"
print(f"[run] reset to seed commit {SEED_COMMIT}")
subprocess.run(
    ["git", "checkout", SEED_COMMIT, "--", "src/models/feature_agent.py", "src/models/train.py"],
    check=True
)

# 加载配置
config = Config('config.yaml')

# 创建研究实例
research = ZeroConfigAutoResearch(
    data_path=config.get('data.train_path'),
    task_description=config.get('data.task_description')
)

# 运行自动优化
research.run(n_iterations=config.get('training.n_iterations', 5))
