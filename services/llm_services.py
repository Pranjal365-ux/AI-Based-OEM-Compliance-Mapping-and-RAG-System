# services/llm_services.py

import time
import logging
from openai import OpenAI
from config.settings import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class LLMService:
    """
    Thin wrapper around a locally-hosted Ollama OpenAI-compatible endpoint.

    Two separate model slots are exposed:

      generate()            → uses cfg.llm.model  (qwen3:8b by default)
                              This is the REASONING model — good for compliance
                              report generation, complex analysis, etc.

      generate_fast()       → uses cfg.llm.extraction_model if set, otherwise
                              falls back to cfg.llm.model with /no_think.
                              This is the EXTRACTION model — pure JSON output,
                              no chain-of-thought overhead.

    Why two models?
    ───────────────
    qwen3:8b is a hybrid reasoning model. Unless suppressed, it emits a large
    <think>…</think> block before the actual answer. On a CPU-only Ollama
    server that can add 2-5 minutes per call — catastrophic when you need
    to extract requirements from 20+ page chunks in parallel.

    Using a non-reasoning instruct model (qwen2.5:7b, llama3.1:8b, mistral:7b)
    for the extraction pass eliminates that overhead entirely. The reasoning
    model is then only invoked for the compliance report, where the extra
    deliberation genuinely improves output quality.
    """

    def __init__(self):
        self.cfg = DEFAULT_CONFIG
        self.client = OpenAI(
            api_key=self.cfg.llm.api_key,
            base_url=self.cfg.llm.base_url,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _call(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = False,
    ) -> str:
        if disable_thinking:
            # Qwen3 (and other hybrid-reasoning models on Ollama) honour
            # a literal "/no_think" appended to the user turn to skip the
            # <think> block.  This works across all serving backends
            # (Ollama, llama.cpp, vLLM) so it's more portable than server-
            # specific request flags.
            prompt = f"{prompt}\n\n/no_think"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw = response.choices[0].message.content.strip()
                # Strip any residual <think>…</think> block that snuck through
                import re
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return raw
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}"
                )
                time.sleep(2 ** attempt)

    # ── public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 3000,
    ) -> str:
        """
        Reasoning model call — used for compliance report generation and any
        task where chain-of-thought genuinely helps.
        """
        return self._call(
            model=self.cfg.llm.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=False,
        )

    def generate_fast(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 3000,
    ) -> str:
        """
        Fast extraction model call — used for RFP requirement extraction.

        If cfg.llm.extraction_model is set (e.g. "qwen2.5:7b"), that model
        is used directly with no thinking-suppression needed.

        If it's None, falls back to the reasoning model with /no_think
        appended so the call is still fast.
        """
        ext_model = self.cfg.llm.extraction_model
        if ext_model:
            return self._call(
                model=ext_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=False,   # non-reasoning model, no need
            )
        else:
            # Fallback: same model, thinking suppressed
            return self._call(
                model=self.cfg.llm.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=True,
            )


llm = LLMService()