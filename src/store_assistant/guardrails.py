"""Spend guardrails that apply no matter whose API key is configured.

Per-session limits are enforced by PydanticAI's `UsageLimits` against a `RunUsage` object that
is shared across every turn of the session. The daily ceiling is enforced from the `usage_log`
table before a new session starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from genai_prices import Usage as PriceUsage
from genai_prices import calc_price
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from store_assistant.config import Settings
from store_assistant.db import Repository


def build_usage_limits(settings: Settings, model: str) -> UsageLimits:
    """Session-wide limits. The cost limit is only enforceable for models with pricing data."""
    priced = _priced(model)
    return UsageLimits(
        request_limit=settings.max_requests_per_session,
        total_tokens_limit=settings.max_tokens_per_session,
        cost_limit=settings.session_cost_limit_usd if priced else None,
    )


def _priced(model: str) -> bool:
    """Whether genai-prices knows this model (so a cost limit can be enforced)."""
    provider, _, name = model.partition(":")
    try:
        calc_price(
            PriceUsage(input_tokens=1, output_tokens=1), model_ref=name, provider_id=provider
        )
    except Exception:
        return False
    return True


def estimate_cost(usage: RunUsage | RequestUsage, model: str) -> Decimal:
    """Estimated USD cost of some usage.

    Prefers the cost PydanticAI computed from the provider response. Falls back to the
    genai-prices table for the model, and to zero when the model is unknown (test models).
    """
    cost = getattr(usage, "cost", None)
    if cost is not None:
        return Decimal(str(cost))
    provider, _, model_ref = model.partition(":")
    try:
        price = calc_price(
            PriceUsage(
                input_tokens=usage.input_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
            ),
            model_ref=model_ref,
            provider_id=provider or "anthropic",
        )
    except Exception:  # unknown model or pricing data unavailable
        return Decimal("0")
    return Decimal(str(price.total_price))


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    spent_today_usd: Decimal
    daily_limit_usd: Decimal

    @property
    def exceeded(self) -> bool:
        return self.spent_today_usd >= self.daily_limit_usd


def check_daily_budget(repo: Repository, settings: Settings) -> BudgetStatus:
    return BudgetStatus(
        spent_today_usd=repo.cost_today(), daily_limit_usd=settings.daily_cost_limit_usd
    )
