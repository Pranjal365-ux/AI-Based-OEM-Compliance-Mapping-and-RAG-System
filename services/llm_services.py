# services/llm_service.py

from groq import Groq
from config.settings import DEFAULT_CONFIG


class LLMService:
    def __init__(self):
        self.cfg = DEFAULT_CONFIG
        self.client = Groq(api_key=self.cfg.groq_api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 3000,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.cfg.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()


llm = LLMService()