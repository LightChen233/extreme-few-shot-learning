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
        # 使用时间戳作为前缀
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"

    def log_experiment(self, iteration, metrics, kept, model_path=None, reflection=None):
        """记录实验（包含多维度指标）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"{self.exp_id}/exp_{iteration}_{timestamp}"
        exp_dir = self.log_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 保存代码
        shutil.copy('src/models/train.py', exp_dir / 'train.py')
        shutil.copy('src/models/feature_agent.py', exp_dir / 'feature_agent.py')
        if Path('src/models/model_def.py').exists():
            shutil.copy('src/models/model_def.py', exp_dir / 'model_def.py')

        # 保存模型（优先用传入路径，否则用默认路径）
        src_model = Path(model_path) if model_path else Path('model.pt')
        if src_model.exists():
            shutil.copy(src_model, exp_dir / 'model.pt')

        # 保存预测结果
        for fname in ['train_predictions.csv', 'val_predictions.csv', 'test_predictions.csv']:
            if Path(fname).exists():
                shutil.copy(fname, exp_dir / fname)

        # 保存 reflection
        if reflection is not None:
            with open(exp_dir / 'reflection.txt', 'w', encoding='utf-8') as f:
                f.write(reflection)

        # 测试集推理（用刚保存的 model.pt，不依赖 train.py __main__）
        model_saved = exp_dir / 'model.pt'
        if model_saved.exists():
            try:
                from src.utils.test_evaluator import evaluate_test_set
                evaluate_test_set(str(model_saved), exp_dir / 'test_predictions.csv')
            except Exception as e:
                print(f"[Warning] 测试集评估失败: {e}")

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
        return exp_dir

    def save_summary(self):
        with open(self.log_dir / 'summary.json', 'w') as f:
            json.dump(self.experiments, f, indent=2)
