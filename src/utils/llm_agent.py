"""
统一的 LLM 调用接口
"""
import requests
import os
import time

class LLMAgent:
    """统一的 LLM 调用"""

    def __init__(self, api_key=None, api_url=None, model="gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = api_url or "https://api.openai.com/v1/chat/completions"
        self.model = model

    def call(self, prompt, max_tokens=2000, retries=3, retry_delay=10):
        """调用 LLM，网络错误自动重试"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        last_err = None
        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=120)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    print(f"[LLM] 请求失败 (attempt {attempt+1}/{retries}): {e}. 重试中...", flush=True)
                    time.sleep(retry_delay)
        raise last_err

    def extract_code(self, text):
        """提取代码块"""
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        return text.strip()
