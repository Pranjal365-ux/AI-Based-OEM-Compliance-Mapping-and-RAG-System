# # services/llm_services.py

# import time
# import logging
# from openai import OpenAI
# from config.settings import DEFAULT_CONFIG

# logger = logging.getLogger(__name__)


# class LLMService:
#     """
#     Thin wrapper around a locally-hosted Ollama OpenAI-compatible endpoint.

#     Two separate model slots are exposed:

#       generate()            → uses cfg.llm.model  (qwen3:8b by default)
#                               This is the REASONING model — good for compliance
#                               report generation, complex analysis, etc.

#       generate_fast()       → uses cfg.llm.extraction_model if set, otherwise
#                               falls back to cfg.llm.model with /no_think.
#                               This is the EXTRACTION model — pure JSON output,
#                               no chain-of-thought overhead.

#     Why two models?
#     ───────────────
#     qwen3:8b is a hybrid reasoning model. Unless suppressed, it emits a large
#     <think>…</think> block before the actual answer. On a CPU-only Ollama
#     server that can add 2-5 minutes per call — catastrophic when you need
#     to extract requirements from 20+ page chunks in parallel.

#     Using a non-reasoning instruct model (qwen2.5:7b, llama3.1:8b, mistral:7b)
#     for the extraction pass eliminates that overhead entirely. The reasoning
#     model is then only invoked for the compliance report, where the extra
#     deliberation genuinely improves output quality.
#     """

#     def __init__(self):
#         self.cfg = DEFAULT_CONFIG
#         self.client = OpenAI(
#             api_key=self.cfg.llm.api_key,
#             base_url=self.cfg.llm.base_url,
#         )

#     # ── internal ──────────────────────────────────────────────────────────────

#     def _call(
#         self,
#         model: str,
#         prompt: str,
#         temperature: float,
#         max_tokens: int,
#         disable_thinking: bool = False,
#     ) -> str:
#         if disable_thinking:
#             # Qwen3 (and other hybrid-reasoning models on Ollama) honour
#             # a literal "/no_think" appended to the user turn to skip the
#             # <think> block.  This works across all serving backends
#             # (Ollama, llama.cpp, vLLM) so it's more portable than server-
#             # specific request flags.
#             prompt = f"{prompt}\n\n/no_think"

#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 response = self.client.chat.completions.create(
#                     model=model,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                 )
#                 raw = response.choices[0].message.content.strip()
#                 # Strip any residual <think>…</think> block that snuck through
#                 import re
#                 raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
#                 return raw
#             except Exception as exc:
#                 if attempt == max_retries - 1:
#                     raise
#                 logger.warning(
#                     f"LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}"
#                 )
#                 time.sleep(2 ** attempt)

#     # ── public ────────────────────────────────────────────────────────────────

#     def generate(
#         self,
#         prompt: str,
#         temperature: float = 0,
#         max_tokens: int = 3000,
#     ) -> str:
#         """
#         Reasoning model call — used for compliance report generation and any
#         task where chain-of-thought genuinely helps.
#         """
#         return self._call(
#             model=self.cfg.llm.model,
#             prompt=prompt,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             disable_thinking=False,
#         )

#     def generate_fast(
#         self,
#         prompt: str,
#         temperature: float = 0,
#         max_tokens: int = 3000,
#     ) -> str:
#         """
#         Fast extraction model call — used for RFP requirement extraction.

#         If cfg.llm.extraction_model is set (e.g. "qwen2.5:7b"), that model
#         is used directly with no thinking-suppression needed.

#         If it's None, falls back to the reasoning model with /no_think
#         appended so the call is still fast.
#         """
#         ext_model = self.cfg.llm.extraction_model
#         if ext_model:
#             return self._call(
#                 model=ext_model,
#                 prompt=prompt,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 disable_thinking=False,   # non-reasoning model, no need
#             )
#         else:
#             # Fallback: same model, thinking suppressed
#             return self._call(
#                 model=self.cfg.llm.model,
#                 prompt=prompt,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 disable_thinking=True,
#             )


# llm = LLMService()

# services/llm_services.py

import re
import time
import logging
import threading
from collections import deque

from openai import OpenAI
from config.settings import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class _RateLimiter:
    """
    Simple client-side rate limiter (sliding 60s window) so we stay under
    Groq's per-minute request/token limits instead of bursting and getting
    hit with 429s. Thread-safe — pipeline code calls this from multiple
    workers (e.g. RFP extraction's max_workers thread pool).
    """

    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self._lock = threading.Lock()
        self._requests = deque()   # timestamps of recent requests
        self._tokens = deque()     # (timestamp, est_tokens) of recent calls

    def acquire(self, est_tokens: int = 0) -> None:
        with self._lock:
            now = time.monotonic()
            self._evict(now)

            wait = 0.0
            if len(self._requests) >= self.rpm_limit:
                wait = max(wait, 60 - (now - self._requests[0]))
            used_tokens = sum(t for _, t in self._tokens)
            if used_tokens + est_tokens > self.tpm_limit:
                if self._tokens:
                    wait = max(wait, 60 - (now - self._tokens[0][0]))

            if wait > 0:
                logger.info(f"Rate limit guard: sleeping {wait:.1f}s before next Groq call")
                time.sleep(wait)
                now = time.monotonic()
                self._evict(now)

            self._requests.append(now)
            self._tokens.append((now, est_tokens))

    def _evict(self, now: float) -> None:
        while self._requests and now - self._requests[0] > 60:
            self._requests.popleft()
        while self._tokens and now - self._tokens[0][0] > 60:
            self._tokens.popleft()


class LLMService:
    """
    Thin wrapper around the Groq OpenAI-compatible endpoint
    (https://api.groq.com/openai/v1).

    Two separate model slots are exposed:

      generate()            -> uses cfg.llm.model ("openai/gpt-oss-120b")
                              This is the REASONING model - used for
                              compliance report generation, complex
                              analysis, etc.

      generate_fast()       -> uses cfg.llm.extraction_model
                              ("openai/gpt-oss-20b") - a fast, non-reasoning
                              instruct model used for RFP requirement
                              extraction, where we need pure JSON output
                              with minimal latency.

    Both models are served by Groq's LPU inference stack, so both are far
    faster than CPU-hosted local models, while gpt-oss-120b still gives
    strong reasoning quality for the compliance report pass.

    Rate limiting
    -------------
    Groq's developer-tier limits for both models are 250K TPM / 1K RPM.
    A client-side sliding-window limiter (see `_RateLimiter`) throttles
    outgoing calls so we stay comfortably under those limits even when
    extraction runs with multiple parallel workers, avoiding bursty 429s.
    On top of that, `_call` retries with exponential backoff and honours
    the `Retry-After` header on 429 responses from the API itself.
    """

    def __init__(self):
        self.cfg = DEFAULT_CONFIG
        self.client = OpenAI(
            api_key=self.cfg.llm.api_key,
            base_url=self.cfg.llm.base_url,
        )
        self._limiter = _RateLimiter(
            rpm_limit=self.cfg.llm.rpm_limit,
            tpm_limit=self.cfg.llm.tpm_limit,
        )

        if not self.cfg.llm.api_key or self.cfg.llm.api_key == "local":
            logger.warning(
                "GROQ_API_KEY not set - Groq calls will fail. "
                "Add GROQ_API_KEY=<your key> to your .env file."
            )

    # -- internal -----------------------------------------------------------

    def _call(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = False,
    ) -> str:
        if disable_thinking:
            # Harmless no-op for Groq's gpt-oss models, but kept for
            # compatibility in case a hybrid-reasoning model (e.g. a Qwen3
            # variant) is swapped in via config - those honour a literal
            # "/no_think" suffix to skip the <think> block.
            prompt = f"{prompt}\n\n/no_think"

        # Rough token estimate for the rate limiter (chars/4 is a standard
        # approximation); good enough to keep us under the TPM ceiling.
        est_tokens = (len(prompt) // 4) + max_tokens

        max_retries = 5
        for attempt in range(max_retries):
            self._limiter.acquire(est_tokens)
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw = response.choices[0].message.content.strip()
                # Strip any residual <think>...</think> block that snuck through
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return raw
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status == 429:
                    # Respect Retry-After if Groq sent one; otherwise back off.
                    retry_after = None
                    resp = getattr(exc, "response", None)
                    if resp is not None:
                        retry_after = resp.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else (2 ** attempt) * 2
                    logger.warning(
                        f"Groq rate limit hit (attempt {attempt + 1}/{max_retries}); "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}"
                )
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Groq call to {model} failed after {max_retries} retries")

    # -- public ---------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 3000,
    ) -> str:
        """
        Reasoning model call (openai/gpt-oss-120b) - used for compliance
        report generation and any task where chain-of-thought genuinely
        helps.
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
        Fast extraction model call (openai/gpt-oss-20b) - used for RFP
        requirement extraction and per-requirement compliance verdicts.

        If cfg.llm.extraction_model is set, that model is used directly.
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
                disable_thinking=False,
            )
        else:
            return self._call(
                model=self.cfg.llm.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=True,
            )


llm = LLMService()
