"""
零配置框架 - 继承基类
"""
import shutil
import pandas as pd
from pathlib import Path
from src.agents.base_framework import BaseAutoResearch
from src.agents.reflection import ReflectionAgent

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

    def _format_val_errors(self, metrics):
        """格式化样本误差供 prompt 使用"""
        val_errors = metrics.get('val_errors', [])
        if not val_errors:
            return "暂无样本误差数据"
        sorted_errors = sorted(
            val_errors,
            key=lambda e: abs(e.get('strain_err', 0)) + abs(e.get('tensile_err', 0)) + abs(e.get('yield_err', 0)),
            reverse=True
        )
        lines = []
        for e in sorted_errors:
            total = abs(e.get('strain_err', 0)) + abs(e.get('tensile_err', 0)) + abs(e.get('yield_err', 0))
            lines.append(
                f"  temp={e['temp']:.0f}°C, time={e['time']:.0f}h → "
                f"应变误差={e['strain_err']:+.2f}, 抗拉误差={e['tensile_err']:+.2f}, 屈服误差={e['yield_err']:+.2f}"
                f"  [总={total:.1f}]"
            )
        return "\n".join(lines)

    def run(self, n_iterations=5):
        print("=== 零配置自动研究 ===\n")

        # 生成领域知识
        domain_knowledge = self.generate_domain_knowledge()
        print(f"领域知识:\n{domain_knowledge}\n")

        # 基线
        best_metrics = self.run_experiment()
        print(f"[Baseline] {best_metrics}\n")
        self.git_commit(f"baseline: {best_metrics.get('overall_mse', 0):.4f}")
        self.tracker.log_experiment(0, best_metrics, True)

        # 用 tracker 的 exp 根目录初始化 ReflectionAgent（带 API 配置）
        exp_root = self.tracker.log_dir / self.tracker.exp_id
        exp_root.mkdir(parents=True, exist_ok=True)
        from src.utils.config_loader import Config
        import os
        config = Config('config.yaml')
        api_key = config.get('model.api_key_env')
        actual_key = api_key if ('sk-' in api_key and not api_key.startswith('$')) else os.environ.get(api_key)
        self.reflector = ReflectionAgent(
            exp_dir=exp_root,
            api_key=actual_key,
            api_url=config.get('model.api_url'),
            model=config.get('model.model_name')
        )

        # 迭代优化
        for i in range(n_iterations):
            print(f"[{i+1}/{n_iterations}]")
            val_errors_str = self._format_val_errors(best_metrics)
            reflection_context = self.reflector.get_context_for_agent()

            # --- feature agent ---
            with open('prompts/feature_agent.txt') as f:
                fa_template = f.read()
            with open('src/models/feature_agent.py') as f:
                fa_code = f.read()
            fa_prompt = fa_template.format(
                task_description=self.task_description,
                val_loss=best_metrics.get('overall_mse', 0),
                val_errors=val_errors_str,
                domain_knowledge=domain_knowledge,
                reflection_context=reflection_context,
                current_code=fa_code
            )
            new_fa_code = self.call_llm(fa_prompt)
            with open('src/models/feature_agent.py', 'w') as f:
                f.write(new_fa_code)

            # --- program agent ---
            with open('prompts/program_agent.txt') as f:
                pa_template = f.read()
            with open('src/models/train.py') as f:
                pa_code = f.read()
            pa_prompt = pa_template.format(
                task_description=self.task_description,
                metrics=best_metrics,
                val_errors=val_errors_str,
                reflection_context=reflection_context,
                current_code=pa_code
            )
            new_pa_code = self.call_llm(pa_prompt)
            with open('src/models/train.py', 'w') as f:
                f.write(new_pa_code)

            # --- 训练 & 评估 ---
            new_metrics = self.run_experiment()
            print(f"新指标: {new_metrics}")

            kept = new_metrics.get('overall_mse', float('inf')) < best_metrics.get('overall_mse', float('inf'))

            # 立即暂存 model.pt（git_revert 前）
            tmp_model = Path('model_tmp.pt')
            if Path('model.pt').exists():
                shutil.copy('model.pt', tmp_model)

            # 反思（带样本误差）
            reflection = self.reflector.reflect(
                i+1, best_metrics, new_metrics, "", kept,
                val_errors=new_metrics.get('val_errors', [])
            )
            print(f"反思: {reflection}\n")

            # 决策
            if kept:
                print("✓ 改进\n")
                self.git_commit(f"improve: {new_metrics['overall_mse']:.4f}")
                self.tracker.log_experiment(i+1, new_metrics, True, model_path=tmp_model, reflection=reflection)
                best_metrics = new_metrics
            else:
                print("✗ 回退\n")
                self.tracker.log_experiment(i+1, new_metrics, False, model_path=tmp_model, reflection=reflection)
                self.git_revert()

            if tmp_model.exists():
                tmp_model.unlink()

        self.tracker.save_summary()
        print(f"=== 最佳: {best_metrics.get('overall_mse', 0):.4f} ===")
