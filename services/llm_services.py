# services/llm_services.py

from groq import Groq
from config.settings import DEFAULT_CONFIG


class LLMService:
    def __init__(self):
        self.cfg = DEFAULT_CONFIG
        # Since the Groq SDK is OpenAI compatible, we can override the base_url
        # to point to a local OpenAI-compatible server (e.g. vLLM, LM Studio, Ollama).
        self.client = Groq(
            api_key=self.cfg.llm.api_key or "local",
            base_url=self.cfg.llm.base_url
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 3000,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.cfg.llm.model,
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