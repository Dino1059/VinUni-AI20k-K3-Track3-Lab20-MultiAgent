"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
import time
from dataclasses import dataclass

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Rough public pricing (USD per 1K tokens) used only for benchmark cost estimates. Unknown
# models (e.g. Groq's free-tier models) simply get `cost_usd=None` instead of a guess.
_PRICING_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4.1-mini": (0.0004, 0.0016),
}

# Rate-limit handling: providers like Groq return 429s under load. Retry those specifically
# on a fixed 1-minute cadence rather than the short exponential backoff used for other
# transient errors, since a 429 usually means "wait for the quota window to roll over".
_MAX_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_WAIT_SECONDS = 60


def _is_not_rate_limit_error(exc: BaseException) -> bool:
    """Retry condition for the short exponential-backoff retry: skip RateLimitError so the
    dedicated 1-minute retry loop in `_complete_openai` handles it instead."""

    try:
        from openai import RateLimitError
    except ImportError:
        return True
    return not isinstance(exc, RateLimitError)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client.

    Uses OpenAI's Chat Completions API when `OPENAI_API_KEY` is configured. Without a key
    (e.g. offline dev, CI, or a fresh clone before the student adds credentials), it falls
    back to a deterministic offline mock so the rest of the pipeline stays runnable.
    """

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.openai_model
        self._api_key = settings.openai_api_key
        self._base_url = settings.openai_base_url

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion, retrying transient provider failures."""

        if not self._api_key:
            logger.warning("OPENAI_API_KEY not set; returning offline mock LLM response")
            return self._complete_mock(system_prompt, user_prompt)
        return self._complete_openai(system_prompt, user_prompt)

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        from openai import RateLimitError

        attempt = 1
        while True:
            try:
                return self._call_openai_once(system_prompt, user_prompt)
            except RateLimitError:
                if attempt >= _MAX_RATE_LIMIT_RETRIES:
                    raise
                logger.warning(
                    "Rate limited by LLM provider (attempt %d/%d); waiting %ds before retry",
                    attempt,
                    _MAX_RATE_LIMIT_RETRIES,
                    _RATE_LIMIT_WAIT_SECONDS,
                )
                time.sleep(_RATE_LIMIT_WAIT_SECONDS)
                attempt += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_is_not_rate_limit_error),
        reraise=True,
    )
    def _call_openai_once(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        prices = _PRICING_PER_1K_TOKENS.get(self._model)
        if prices is None or input_tokens is None or output_tokens is None:
            return None
        input_price, output_price = prices
        return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price

    def _complete_mock(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        content = (
            "[offline-mock-llm] No OPENAI_API_KEY configured, so this is a deterministic "
            f"stand-in response.\nSystem: {system_prompt[:120]}\nUser: {user_prompt[:300]}"
        )
        input_tokens = len(system_prompt.split()) + len(user_prompt.split())
        output_tokens = len(content.split())
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
        )
