from __future__ import annotations

from typing import Optional

from src.config import ConfigurationError, Settings, settings as default_settings
from src.llm.base import LLMClient
from src.llm.providers.groq import GroqClient


def create_llm_client(
    settings: Settings = default_settings,
    provider: Optional[str] = None,
) -> LLMClient:
    provider_name = (provider or settings.llm_provider).strip().lower()
    if provider_name == "groq":
        return GroqClient(settings)
    raise ConfigurationError(
        f"LLM provider {provider_name!r} is not implemented. Use 'groq'."
    )
