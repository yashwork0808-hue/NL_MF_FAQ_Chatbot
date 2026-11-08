from __future__ import annotations

import time
from typing import Any, Optional

from src.config import Settings, settings as default_settings
from src.llm.base import LLMClient, LLMError, LLMResponse


class GroqClient(LLMClient):
    def __init__(self, settings: Settings = default_settings) -> None:
        self.settings = settings
        self.settings.require_llm_ready()
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMError(
                "The groq package is not installed. Install requirements first."
            ) from exc
        try:
            self.client = Groq(api_key=self.settings.groq_api_key)
        except Exception as exc:
            raise LLMError(f"Unable to initialize Groq client: {exc}") from exc

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = self._extract_text(completion)
        if not text.strip():
            raise LLMError("Groq returned an empty answer")

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = (
            getattr(usage, "completion_tokens", None) if usage else None
        )
        return LLMResponse(
            text=text,
            provider="groq",
            model=self.settings.llm_model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _extract_text(completion: Any) -> str:
        try:
            choice = completion.choices[0]
            content: Optional[str] = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Groq response shape: {exc}") from exc
        return content or ""
