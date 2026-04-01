import subprocess
import json
from pathlib import Path

class AutoResearch:
    def __init__(self):
        self.log_file = 'experiments.json'
        self.experiments = []

    def run_experiment(self):
        """运行一次实验"""
        result = subprocess.run(
            ['python3', 'train.py'],
            capture_output=True,
            text=True
        )

        # 解析 val_loss
        for line in result.stdout.split('\n'):
            if 'Val Loss:' in line:
                val_loss = float(line.split('Val Loss:')[1].strip())
                return val_loss
        return float('inf')

    def should_keep(self, new_loss, best_loss):
        """决策：保留还是回退"""
        return new_loss < best_loss

    def git_commit(self, message):
        subprocess.run(['git', 'add', 'train.py', 'feature_agent.py'])
        subprocess.run(['git', 'commit', '-m', message])

    def git_revert(self):
        subprocess.run(['git', 'reset', '--hard', 'HEAD~1'])

    def inner_loop(self, n_iterations=5):
        """内循环：快速迭代"""
        print("=== Starting Inner Loop ===")

        # 初始基线
        print("\n[Baseline] Running initial experiment...")
        best_loss = self.run_experiment()
        print(f"Baseline Val Loss: {best_loss:.4f}")
        self.git_commit(f"baseline: val_loss={best_loss:.4f}")

        for i in range(n_iterations):
            print(f"\n[Iteration {i+1}/{n_iterations}]")
            print("Modify train.py or feature_agent.py, then press Enter...")
            input()

            new_loss = self.run_experiment()
            print(f"New Val Loss: {new_loss:.4f}")

            if self.should_keep(new_loss, best_loss):
                print(f"✓ Improved! {best_loss:.4f} -> {new_loss:.4f}")
                self.git_commit(f"improve: val_loss={new_loss:.4f}")
                best_loss = new_loss
                self.experiments.append({'iter': i+1, 'val_loss': new_loss, 'kept': True})
            else:
                print(f"✗ No improvement. Reverting...")
                self.git_revert()
                self.experiments.append({'iter': i+1, 'val_loss': new_loss, 'kept': False})

        print(f"\n=== Best Val Loss: {best_loss:.4f} ===")
        return best_loss

if __name__ == '__main__':
    ar = AutoResearch()
    ar.inner_loop(n_iterations=3)
