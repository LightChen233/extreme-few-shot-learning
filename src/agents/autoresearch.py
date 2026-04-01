"""
零配置框架 - 继承基类
"""
import pandas as pd
from src.agents.base_framework import BaseAutoResearch

class ZeroConfigAutoResearch(BaseAutoResearch):
    """零配置自动研究"""

    def __init__(self, data_path, task_description):
        super().__init__()
        self.data_path = data_path
        self.task_description = task_description

        # 自动分析数据
        df = pd.read_csv(data_path)
        self.data_summary = f"""
数据形状: {df.shape}
列名: {df.columns.tolist()}
前3行: {df.head(3).to_string()}
"""

    def generate_domain_knowledge(self):
        """自动生成领域知识"""
        with open('prompts/domain_knowledge.txt') as f:
            template = f.read()

        prompt = template.format(
            task_description=self.task_description,
            data_summary=self.data_summary
        )
        return self.call_llm(prompt, max_tokens=1000)

    def run(self, n_iterations=5):
        print("=== 零配置自动研究 ===\n")

        # 生成领域知识
        domain_knowledge = self.generate_domain_knowledge()
        print(f"领域知识:\n{domain_knowledge}\n")

        # 基线
        best_metrics = self.run_experiment()
        print(f"[Baseline] {best_metrics}\n")
        self.git_commit(f"baseline: {best_metrics.get('overall_mse', 0):.4f}")
        exp_id = self.tracker.log_experiment(0, best_metrics, True)

        # 用 tracker 的 exp 根目录初始化 ReflectionAgent
        exp_root = self.tracker.log_dir / self.tracker.exp_id
        exp_root.mkdir(parents=True, exist_ok=True)
        self.reflector = ReflectionAgent(exp_dir=exp_root)

        # 迭代优化
        for i in range(n_iterations):
            print(f"[{i+1}/{n_iterations}]")

            # 读取 prompt 模板
            with open('prompts/feature_agent.txt') as f:
                template = f.read()
            with open('src/models/feature_agent.py') as f:
                code = f.read()

            # 生成新代码
            prompt = template.format(
                task_description=self.task_description,
                val_loss=best_metrics.get('overall_mse', 0),
                domain_knowledge=domain_knowledge,
                reflection_context=self.reflector.get_context_for_agent(),
                current_code=code
            )
            new_code = self.call_llm(prompt)

            with open('src/models/feature_agent.py', 'w') as f:
                f.write(new_code)

            # 测试
            new_metrics = self.run_experiment()
            print(f"新指标: {new_metrics}")

            kept = new_metrics.get('overall_mse', float('inf')) < best_metrics.get('overall_mse', float('inf'))

            # 反思
            reflection = self.reflector.reflect(
                i+1, best_metrics, new_metrics, "", kept
            )
            print(f"反思: {reflection}\n")

            # 决策
            if kept:
                print("✓ 改进\n")
                self.git_commit(f"improve: {new_metrics['overall_mse']:.4f}")
                self.tracker.log_experiment(i+1, new_metrics, True)
                best_metrics = new_metrics
            else:
                print("✗ 回退\n")
                self.git_revert()
                self.tracker.log_experiment(i+1, new_metrics, False)

        self.tracker.save_summary()
        print(f"=== 最佳: {best_metrics.get('overall_mse', 0):.4f} ===")
