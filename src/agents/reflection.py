"""
反思系统
"""
from src.utils.llm_agent import LLMAgent
import json
from pathlib import Path

class ReflectionAgent:
    def __init__(self, exp_dir=None, api_key=None, api_url=None, model=None):
        self.llm = LLMAgent(api_key=api_key, api_url=api_url, model=model)
        self.history = []
        self.exp_dir = Path(exp_dir) if exp_dir else None

    def reflect(self, iteration, metrics_before, metrics_after, code_change, kept, val_errors=None):
        with open('prompts/reflection.txt') as f:
            template = f.read()

        prompt = template.format(
            iteration=iteration,
            metrics_before=self._format_metrics(metrics_before),
            metrics_after=self._format_metrics(metrics_after),
            kept_status='✓ 保留' if kept else '✗ 回退',
            val_errors=self._format_val_errors(metrics_after.get('val_errors', []))
        )

        reflection = self.llm.call(prompt, max_tokens=500)
        entry = {'iteration': iteration, 'reflection': reflection, 'kept': kept}
        self.history.append(entry)

        # 持久化到 exp 文件夹
        if self.exp_dir:
            reflection_file = self.exp_dir / 'reflections.jsonl'
            with open(reflection_file, 'a') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        return reflection

    def get_context_for_agent(self):
        if not self.history:
            return "首次运行，无历史反思。"
        lines = ["历史反思（最近3次）:"]
        for h in self.history[-3:]:
            status = "✓" if h['kept'] else "✗"
            lines.append(f"\n迭代{h['iteration']} {status}: {h['reflection']}")
        return "\n".join(lines)

    def _format_metrics(self, m):
        if not m:
            return "无"
        return (
            f"  总体 MSE: {m.get('overall_mse', 0):.4f}\n"
            f"  应变 → MSE={m.get('strain_mse', 0):.4f}, MAE={m.get('strain_mae', 0):.4f}\n"
            f"  抗拉 → MSE={m.get('tensile_mse', 0):.4f}, MAE={m.get('tensile_mae', 0):.4f}\n"
            f"  屈服 → MSE={m.get('yield_mse', 0):.4f}, MAE={m.get('yield_mae', 0):.4f}"
        )

    def _format_val_errors(self, val_errors):
        if not val_errors:
            return "无"
        # 按总绝对误差降序排列，突出最难预测的条件
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
