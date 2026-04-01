"""
统一的 LLM 调用接口
"""
import requests
import os

class LLMAgent:
    """统一的 LLM 调用"""

    def __init__(self, api_key=None, api_url=None, model="gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = api_url or "https://api.openai.com/v1/chat/completions"
        self.model = model

    def call(self, prompt, max_tokens=2000):
        """调用 LLM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def extract_code(self, text):
        """提取代码块"""
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        return text.strip()
