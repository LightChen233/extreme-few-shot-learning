"""
反思系统
"""
from src.utils.llm_agent import LLMAgent
from src.utils.config_loader import Config
import json
from pathlib import Path

class ReflectionAgent:
    def __init__(self, exp_dir=None, api_key=None, api_url=None, model=None):
        self.llm = LLMAgent(api_key=api_key, api_url=api_url, model=model)
        self.history = []
        self.exp_dir = Path(exp_dir) if exp_dir else None

        config = Config('config.yaml')
        self.input_cols  = config.input_cols
        self.target_cols = config.target_cols
        self.output_meta = config.output_meta  # [{key, name, unit, ...}]

    def reflect(self, iteration, metrics_before, metrics_after, code_change, kept, val_errors=None, coverage_analysis=""):
        with open('prompts/reflection.txt') as f:
            template = f.read()

        prompt = template.format(
            iteration=iteration,
            metrics_before=self._format_metrics(metrics_before),
            metrics_after=self._format_metrics(metrics_after),
            kept_status='✓ 保留' if kept else '✗ 回退',
            val_errors=self._format_val_errors(metrics_after.get('val_errors', [])),
            coverage_analysis=coverage_analysis or "未提供"
        )

        reflection = self.llm.call(prompt, max_tokens=500)
        entry = {'iteration': iteration, 'reflection': reflection, 'kept': kept,
                 'old_mse': metrics_before.get('overall_mse'), 'new_mse': metrics_after.get('overall_mse')}
        self.history.append(entry)

        if self.exp_dir:
            with open(self.exp_dir / 'reflections.jsonl', 'a') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        return reflection

    def get_context_for_agent(self):
        if not self.history:
            return "首次运行，无历史反思。"
        n_kept = sum(1 for h in self.history if h['kept'])
        n_total = len(self.history)
        n_failed = n_total - n_kept
        summary = f"历史反思（共{n_total}次迭代，{n_kept}次改进，{n_failed}次回退）:"
        if n_failed == n_total and n_total >= 2:
            summary += f"\n⚠️ 警告：所有{n_total}次改动均使性能变差！当前代码是迄今最优基线，请做最小化改动，不要重写架构。"
        lines = [summary]
        for h in self.history[-3:]:
            status = "✓" if h['kept'] else "✗"
            lines.append(f"\n迭代{h['iteration']} {status} (MSE: {h.get('old_mse',0):.1f}→{h.get('new_mse',0):.1f}): {h['reflection']}")
        return "\n".join(lines)

    def _format_metrics(self, m):
        if not m:
            return "无"
        lines = [f"  总体 MSE: {m.get('overall_mse', 0):.4f}"]
        if self.output_meta:
            for om in self.output_meta:
                mse_k = om.get('mse_key', '')
                mae_k = om.get('mae_key', '')
                rel_k = om.get('rel_key', '')
                name  = f"{om['name']}({om['unit']})"
                lines.append(
                    f"  {name} → MSE={m.get(mse_k, 0):.4f}, MAE={m.get(mae_k, 0):.4f}, rel={m.get(rel_k, 0):.4f}"
                )
        else:
            # 回退：直接打印所有 *_mse/*_mae 键
            for k, v in m.items():
                if k.endswith('_mse') or k.endswith('_mae'):
                    lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)

    def _get_err(self, e, key):
        if f'{key}_err' in e:
            return e[f'{key}_err']
        return e.get(f'true_{key}', 0) - e.get(f'pred_{key}', 0)

    def _format_val_errors(self, val_errors):
        if not val_errors:
            return "无"

        # 短名映射：tensile_strength -> tensile
        short = {col: col.split('_')[0] for col in self.target_cols}
        name_map = {om['key']: f"{om['name']}({om['unit']})" for om in self.output_meta} if self.output_meta else {}

        def total_err(e):
            return sum(abs(self._get_err(e, s)) for s in short.values())

        sorted_errors = sorted(val_errors, key=total_err, reverse=True)
        lines = []
        for e in sorted_errors:
            # 输入列描述
            in_desc = ", ".join(f"{c}={e.get(c, '?')}" for c in self.input_cols)
            # 各输出误差
            err_parts = []
            for col, s in short.items():
                err = self._get_err(e, s)
                label = name_map.get(col, col)
                err_parts.append(f"{label}误差={err:+.2f}")
            total = total_err(e)
            lines.append(f"  {in_desc} → {', '.join(err_parts)}  [总={total:.1f}]")
        return "\n".join(lines)
