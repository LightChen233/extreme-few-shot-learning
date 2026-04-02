"""
AutoResearch 基类
"""
import subprocess
import shutil
import os
from pathlib import Path
from src.utils.experiment_tracker import ExperimentTracker
from src.agents.reflection import ReflectionAgent
from src.utils.llm_agent import LLMAgent
from src.utils.config_loader import Config

# 被优化的目标文件列表
MANAGED_FILES = [
    'src/models/feature_agent.py',
    'src/models/train.py',
]

class BaseAutoResearch:
    def __init__(self):
        config = Config('config.yaml')
        api_key = config.get('model.api_key_env')
        # 如果不是环境变量名，直接使用
        if not api_key.startswith('$') and 'sk-' in api_key:
            actual_key = api_key
        else:
            actual_key = os.environ.get(api_key)

        self.llm = LLMAgent(
            api_key=actual_key,
            api_url=config.get('model.api_url'),
            model=config.get('model.model_name')
        )
        self.tracker = ExperimentTracker()
        self.reflector = ReflectionAgent()

    def run_experiment(self):
        result = subprocess.run(['python3', 'src/models/train.py'],
                              capture_output=True, text=True, timeout=360,
                              cwd='/Volumes/Mac DS/Users/mac/Documents/Data/code/extreme-few-shot-learning')
        metrics = {}
        for line in result.stdout.split('\n'):
            if 'Val Loss:' in line:
                metrics['overall_mse'] = float(line.split('Val Loss:')[1].strip())
            elif line.startswith('VAL_PRED'):
                # 每条样本误差：VAL_PRED temp=X time=Y strain_err=A tensile_err=B yield_err=C
                entry = {}
                for token in line.replace('VAL_PRED ', '').split():
                    k, v = token.split('=')
                    entry[k] = float(v)
                metrics.setdefault('val_errors', []).append(entry)
            elif 'TEST_METRICS' in line:
                parts = line.replace('TEST_METRICS ', '').split()
                for part in parts:
                    if '=' in part:
                        k, v = part.split('=')
                        metrics[f'test_{k}'] = float(v)
            elif 'METRICS' in line:
                parts = line.replace('METRICS ', '').split()
                for part in parts:
                    if '=' in part:
                        k, v = part.split('=')
                        metrics[k] = float(v)
        return metrics

    def save_snapshot(self, snapshot_dir):
        """把当前被管理的文件保存到快照目录"""
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for f in MANAGED_FILES:
            shutil.copy(f, snapshot_dir / Path(f).name)

    def restore_snapshot(self, snapshot_dir):
        """从快照目录恢复被管理的文件"""
        snapshot_dir = Path(snapshot_dir)
        for f in MANAGED_FILES:
            src = snapshot_dir / Path(f).name
            if src.exists():
                shutil.copy(src, f)

    def call_llm(self, prompt, max_tokens=2000):
        response = self.llm.call(prompt, max_tokens)
        return self.llm.extract_code(response)
