"""Application settings.

Every tunable lives here and can be overridden with an environment variable or a `.env` file.
The Anthropic API key is read by the Anthropic SDK itself from `ANTHROPIC_API_KEY`; we only load
`.env` into the process environment so the SDK can see it.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

Effort = Literal["low", "medium", "high"]


class Settings(BaseSettings):
    """Runtime configuration. Field names map to upper-cased environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_model: str = "anthropic:claude-opus-5-5"
    summary_model: str | None = Field(
        default=None, description="Model for the end-of-session summary. Defaults to llm_model."
    )
    llm_effort: Effort = "medium"
    max_output_tokens: int = 1024
    prompt_caching: bool = True

    # Conversation rules
    store_passphrase: str = "open sesame"
    passphrase_max_attempts: int = 3
    require_passphrase_per_lookup: bool = False
    off_topic_strike_limit: int = 3
    default_phone_region: str = "US"

    # Storage
    db_path: Path = Path("store_assistant.db")

    # Spend guardrails (apply no matter whose API key is in use)
    max_turns_per_session: int = 40
    max_requests_per_session: int = 80
    max_tokens_per_session: int = 500_000
    session_cost_limit_usd: Decimal = Decimal("0.50")
    daily_cost_limit_usd: Decimal = Decimal("5.00")

    # Tracing
    logfire_token: str | None = None

    @property
    def effective_summary_model(self) -> str:
        return self.summary_model or self.llm_model

    @property
    def uses_anthropic(self) -> bool:
        return self.llm_model.startswith("anthropic:")


def get_settings() -> Settings:
    return Settings()
