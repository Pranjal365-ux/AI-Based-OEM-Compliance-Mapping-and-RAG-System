# services/llm_services.py

from openai import OpenAI
from config.settings import DEFAULT_CONFIG


class LLMService:
    def __init__(self):
        self.cfg = DEFAULT_CONFIG

        self.client = OpenAI(
            api_key="local",
            base_url=self.cfg.llm.base_url
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 3000,
    ) -> str:
        import time
        max_retries = 3
        backoff = 2
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.llm.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(backoff ** attempt)


llm = LLMService()