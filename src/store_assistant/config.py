"""Application settings.

Every tunable lives here and can be overridden with an environment variable or a `.env` file in
the working directory. Nothing is written into the process environment: the API key is passed to
the model provider explicitly, so tests and other tools see exactly the environment they were
started with.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high"]


class Settings(BaseSettings):
    """Runtime configuration. Field names map to upper-cased environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Credentials
    anthropic_api_key: str | None = None
    logfire_token: str | None = None

    # LLM
    llm_model: str = "anthropic:claude-opus-5-5"
    summary_model: str | None = Field(
        default=None, description="Model for the end-of-session summary. Defaults to llm_model."
    )
    llm_effort: Effort = "medium"
    max_output_tokens: int = Field(default=1024, ge=64)
    prompt_caching: bool = True
    refusal_fallback: bool = Field(
        default=True,
        description="Retry requests declined by the safety classifier on fallback_model.",
    )
    fallback_model: str = "anthropic:claude-opus-4-8"

    # Conversation rules
    store_passphrase: str = "open sesame"
    passphrase_max_attempts: int = Field(default=3, ge=1)
    require_passphrase_per_lookup: bool = False
    off_topic_strike_limit: int = Field(default=3, ge=1)
    default_phone_region: str = "US"
    max_store_name_length: int = Field(default=80, ge=1)

    # Storage
    db_path: Path = Path("store_assistant.db")

    # Spend guardrails (apply no matter whose API key is in use)
    max_turns_per_session: int = Field(default=40, ge=1)
    max_requests_per_session: int = Field(default=80, ge=1)
    max_tokens_per_session: int = Field(default=500_000, ge=1000)
    session_cost_limit_usd: Decimal = Field(default=Decimal("0.50"), gt=0)
    daily_cost_limit_usd: Decimal = Field(default=Decimal("5.00"), gt=0)
    max_consecutive_model_errors: int = Field(default=3, ge=1)

    @field_validator("default_phone_region")
    @classmethod
    def _upper_region(cls, value: str) -> str:
        return value.strip().upper()

    @property
    def effective_summary_model(self) -> str:
        return self.summary_model or self.llm_model

    @property
    def uses_anthropic(self) -> bool:
        return is_anthropic_model(self.llm_model)


def is_anthropic_model(model_name: str) -> bool:
    return model_name.startswith("anthropic:")


def get_settings() -> Settings:
    return Settings()
