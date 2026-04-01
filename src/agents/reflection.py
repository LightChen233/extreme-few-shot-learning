"""
反思系统
"""
from src.utils.llm_agent import LLMAgent

class ReflectionAgent:
    def __init__(self):
        self.llm = LLMAgent()
        self.history = []

    def reflect(self, iteration, metrics_before, metrics_after, code_change, kept):
        with open('prompts/reflection.txt') as f:
            template = f.read()

        prompt = template.format(
            iteration=iteration,
            metrics_before=self._format_metrics(metrics_before),
            metrics_after=self._format_metrics(metrics_after),
            kept_status='✓ 保留' if kept else '✗ 回退'
        )

        reflection = self.llm.call(prompt, max_tokens=500)
        self.history.append({'iteration': iteration, 'reflection': reflection, 'kept': kept})
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
