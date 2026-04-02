"""
AutoResearch 基类
"""
import subprocess
import os
from src.utils.experiment_tracker import ExperimentTracker
from src.agents.reflection import ReflectionAgent
from src.utils.llm_agent import LLMAgent
from src.utils.config_loader import Config

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

    def git_commit(self, msg):
        subprocess.run(['git', 'add', '.'], capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg], capture_output=True)

    def git_revert(self):
        subprocess.run(['git', 'reset', '--hard', 'HEAD~1'], capture_output=True)

    def call_llm(self, prompt, max_tokens=2000):
        response = self.llm.call(prompt, max_tokens)
        return self.llm.extract_code(response)
