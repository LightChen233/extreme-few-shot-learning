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

        from src.utils.config_loader import Config
        config = Config('config.yaml')
        self.output_meta  = config.output_meta
        self.input_cols   = config.input_cols
        self.target_cols  = config.target_cols

        # 自动分析数据
        df = pd.read_csv(data_path)
        self.n_train = len(df)

        # 按输入列分组，统计各目标列均值，让 LLM 看到真实趋势方向
        group_cols   = self.input_cols if self.input_cols else df.columns[:2].tolist()
        target_cols  = self.target_cols if self.target_cols else df.select_dtypes('number').columns.tolist()
        stats = df.groupby(group_cols)[target_cols].mean().round(1).reset_index().to_string(index=False)

        self.data_summary = f"""
数据形状: {df.shape}
列名: {df.columns.tolist()}
各条件下的平均性能统计 (非常重要，据此推断物理方向):
{stats}
"""

    def generate_domain_knowledge(self):
        """自动生成领域知识"""
        with open('prompts/domain_knowledge.txt') as f:
            template = f.read()

        prompt = template.format(
            task_description=self.task_description,
            data_summary=self.data_summary
        )
        return self.call_llm_text(prompt, max_tokens=1000)

    def _coverage_analysis(self):
        """分析 val/test 条件相对训练集的覆盖情况"""
        import pandas as pd
        from src.utils.config_loader import Config
        config = Config('config.yaml')

        train_df = pd.read_csv(self.data_path)
        val_df   = pd.read_csv(self.data_path.replace('train', 'val'))
        test_df  = pd.read_csv(config.test_path)

        group_cols = self.input_cols

        train_keys = set(tuple(r) for r in train_df[group_cols].drop_duplicates().values)

        lines = ["各验证/测试条件的训练集覆盖情况："]
        for label, df in [("Val", val_df), ("Test", test_df)]:
            seen = set()
            for _, row in df[group_cols].drop_duplicates().iterrows():
                key = tuple(row)
                if key in seen:
                    continue
                seen.add(key)
                in_train = key in train_keys
                # 找最近邻（欧氏距离）
                dists = train_df[group_cols].apply(
                    lambda r: sum((r[c] - row[c])**2 for c in group_cols)**0.5, axis=1
                )
                nearest = train_df.loc[dists.idxmin(), group_cols].to_dict()
                nearest_str = ", ".join(f"{k}={v}" for k, v in nearest.items())
                status = "✓ 训练集有" if in_train else f"✗ 外推点（最近邻: {nearest_str}）"
                cond = ", ".join(f"{c}={row[c]}" for c in group_cols)
                lines.append(f"  [{label}] {cond} → {status}")

        return "\n".join(lines)

    def _get_err(self, e, key):
        """兼容 strain_err 和 true_strain/pred_strain 两种格式"""
        if f'{key}_err' in e:
            return e[f'{key}_err']
        return e.get(f'true_{key}', 0) - e.get(f'pred_{key}', 0)

    def _format_val_errors(self, metrics):
        """格式化样本误差供 prompt 使用"""
        val_errors = metrics.get('val_errors', [])
        if not val_errors:
            return "暂无样本误差数据"

        short = {col: col.split('_')[0] for col in self.target_cols}
        name_map = {om['key']: f"{om['name']}({om['unit']})" for om in self.output_meta} if self.output_meta else {}

        def total_err(e):
            return sum(abs(self._get_err(e, s)) for s in short.values())

        sorted_errors = sorted(val_errors, key=total_err, reverse=True)
        lines = []
        for e in sorted_errors:
            in_desc = ", ".join(f"{c}={e.get(c, '?')}" for c in self.input_cols)
            err_parts = []
            for col, s in short.items():
                err = self._get_err(e, s)
                label = name_map.get(col, col)
                err_parts.append(f"{label}误差={err:+.2f}")
            total = total_err(e)
            lines.append(f"  {in_desc} → {', '.join(err_parts)}  [总={total:.1f}]")
        return "\n".join(lines)

    def _print_test_summary(self, csv_path, best_val_metrics):
        """用 output_meta 打印直观的 test 集逐条对比"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception:
            return

        meta = self.output_meta  # [{key, name, unit, ...}, ...]
        if not meta:
            # 回退：自动从列名推断
            pred_cols = [c for c in df.columns if c.startswith('pred_')]
            meta = [{'key': c.replace('pred_', ''), 'name': c.replace('pred_', ''), 'unit': ''} for c in pred_cols]

        input_cols = [c for c in df.columns if not c.startswith('pred_') and not c.startswith('true_')]

        print("\n=== Test 集预测对比 ===")
        # 表头
        header = "  ".join(f"{c:>8}" for c in input_cols)
        for m in meta:
            header += f"  {m['name']}(pred)  {m['name']}(true)  误差({m['unit']})"
        print(header)

        for _, row in df.iterrows():
            line = "  ".join(f"{row[c]:>8.1f}" for c in input_cols)
            for m in meta:
                pk, tk = f"pred_{m['key']}", f"true_{m['key']}"
                if pk in row and tk in row:
                    pred_v, true_v = row[pk], row[tk]
                    diff = pred_v - true_v
                    line += f"  {pred_v:>12.1f}  {true_v:>12.1f}  {diff:>+10.1f}"
            print(line)

        # 汇总相对误差
        print("\n  指标汇总 (val):")
        for m in meta:
            rel_key = m.get('rel_key', '')
            mae_key = m.get('mae_key', '')
            if rel_key and rel_key in best_val_metrics:
                print(f"    {m['name']}: MAE={best_val_metrics.get(mae_key, 0):.2f}{m['unit']}  相对误差={best_val_metrics[rel_key]*100:.1f}%")

    def run(self, n_iterations=5):
        print("=== 零配置自动研究 ===\n")

        # 生成领域知识
        domain_knowledge = self.generate_domain_knowledge()
        print(f"领域知识:\n{domain_knowledge}\n")

        # 基线
        best_metrics = self.run_experiment()
        print(f"[Baseline] {best_metrics}\n")
        self.tracker.log_experiment(0, best_metrics, True)
        # 保存 baseline model 作为初始 best_model.pt，确保即使所有迭代都回退也有可用模型
        if Path('model.pt').exists():
            shutil.copy('model.pt', 'best_model.pt')

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

        # 当前最优代码快照（用于回退）
        best_snapshot = exp_root / '_best_snapshot'
        self.save_snapshot(best_snapshot)

        # 预计算覆盖分析（整个 run 只算一次）
        try:
            coverage_analysis = self._coverage_analysis()
        except Exception as e:
            coverage_analysis = f"覆盖分析失败: {e}"
        print(f"覆盖分析:\n{coverage_analysis}\n")

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
                n_train=self.n_train,
                coverage_analysis=coverage_analysis,
                current_code=fa_code.replace('{', '{{').replace('}', '}}')
            )
            new_fa_code = self.call_llm(fa_prompt)
            if new_fa_code:
                with open('src/models/feature_agent.py', 'w') as f:
                    f.write(new_fa_code)

            # --- program agent ---
            with open('prompts/program_agent.txt') as f:
                pa_template = f.read()
            with open('src/models/model_def.py') as f:
                pa_code = f.read()
            pa_prompt = pa_template.format(
                task_description=self.task_description,
                metrics=best_metrics,
                val_errors=val_errors_str,
                reflection_context=reflection_context,
                n_train=self.n_train,
                coverage_analysis=coverage_analysis,
                current_code=pa_code.replace('{', '{{').replace('}', '}}')
            )
            new_pa_code = self.call_llm(pa_prompt)
            if new_pa_code:
                with open('src/models/model_def.py', 'w') as f:
                    f.write(new_pa_code)

            # --- 训练 & 评估 ---
            new_metrics = self.run_experiment()
            print(f"新指标: {new_metrics}")

            kept = new_metrics.get('overall_mse', float('inf')) < best_metrics.get('overall_mse', float('inf'))

            # 暂存本次 model.pt
            tmp_model = Path('model_tmp.pt')
            if Path('model.pt').exists():
                shutil.copy('model.pt', tmp_model)

            # 反思（带样本误差 + 覆盖分析）
            reflection = self.reflector.reflect(
                i+1, best_metrics, new_metrics, "", kept,
                val_errors=new_metrics.get('val_errors', []),
                coverage_analysis=coverage_analysis
            )
            print(f"反思: {reflection}\n")

            # 决策
            if kept:
                print("✓ 改进\n")
                self.tracker.log_experiment(i+1, new_metrics, True, model_path=tmp_model, reflection=reflection)
                best_metrics = new_metrics
                self.save_snapshot(best_snapshot)   # 更新最优快照
                # 同步更新 best_model.pt，供最终 test 评估使用
                if tmp_model.exists():
                    shutil.copy(tmp_model, 'best_model.pt')
            else:
                print("✗ 回退\n")
                self.tracker.log_experiment(i+1, new_metrics, False, model_path=tmp_model, reflection=reflection)
                self.restore_snapshot(best_snapshot)  # 恢复最优快照（log 之后再 restore）

            if tmp_model.exists():
                tmp_model.unlink()

        self.tracker.save_summary()
        print(f"=== 最佳: {best_metrics.get('overall_mse', 0):.4f} ===")

        # 用最优 snapshot 重新生成最终 test 预测
        best_model = Path('best_model.pt')
        if best_model.exists():
            try:
                from src.utils.test_evaluator import evaluate_test_set
                out_path = exp_root / 'best_test_predictions.csv'
                evaluate_test_set(str(best_model), out_path)
                self._print_test_summary(out_path, best_metrics)
                print(f"最终 test 预测已保存: {out_path}")
            except Exception as e:
                print(f"[Warning] 最终 test 评估失败: {e}")

