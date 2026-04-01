import json
import shutil
from pathlib import Path
from datetime import datetime
import uuid

class ExperimentTracker:
    def __init__(self, log_dir='experiments'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.experiments = []
        self.exp_id = uuid.uuid4().hex[:8]

    def log_experiment(self, iteration, metrics, kept):
        """记录实验（包含多维度指标）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"{self.exp_id}/exp_{iteration}_{timestamp}"
        exp_dir = self.log_dir / exp_id
        exp_dir.mkdir(exist_ok=True)

        # 保存代码
        shutil.copy('train.py', exp_dir / 'train.py')
        shutil.copy('feature_agent.py', exp_dir / 'feature_agent.py')

        # 保存结果
        result = {
            'iteration': iteration,
            'timestamp': timestamp,
            'kept': kept,
            'metrics': metrics,
            'exp_dir': str(exp_dir)
        }

        with open(exp_dir / 'result.json', 'w') as f:
            json.dump(result, f, indent=2)

        self.experiments.append(result)
        return exp_id

    def save_summary(self):
        with open(self.log_dir / 'summary.json', 'w') as f:
            json.dump(self.experiments, f, indent=2)
