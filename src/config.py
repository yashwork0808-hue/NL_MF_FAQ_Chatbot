from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SUPPORTED_VECTOR_DBS = {"chroma"}
SUPPORTED_LLM_PROVIDERS = {"groq", "gemini", "openai", "anthropic", "mistral"}


class ConfigurationError(ValueError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    vector_db: str = Field(default="chroma")
    chroma_dir: Path = Field(default=Path("data/chroma"))
    trace_dir: Path = Field(default=Path("data/traces"))
    collection_name: str = Field(default="mf_faq")
    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=60, ge=0)
    top_k: int = Field(default=4, gt=0)
    llm_provider: str = Field(default="groq")
    llm_model: str = Field(default="qwen/qwen3.8-27b")
    groq_api_key: Optional[str] = None
    source_updated_date: Optional[str] = None

    @field_validator("embedding_model", "collection_name", "llm_model")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("LLM_MODEL must be a model identifier without spaces")
        return value

    @field_validator("vector_db")
    @classmethod
    def validate_vector_db(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in SUPPORTED_VECTOR_DBS:
            supported = ", ".join(sorted(SUPPORTED_VECTOR_DBS))
            raise ValueError(f"VECTOR_DB must be one of: {supported}")
        return value

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in SUPPORTED_LLM_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
            raise ValueError(f"LLM_PROVIDER must be one of: {supported}")
        return value

    @field_validator("groq_api_key", "source_updated_date", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def source_date(self) -> str:
        return self.source_updated_date or date.today().isoformat()

    def require_llm_ready(self) -> None:
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is required when LLM_PROVIDER=groq. "
                "Set it in .env or choose another provider."
            )


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
