"""Keeping the passphrase out of everything that is stored, sent, or traced.

Two layers:

1. `split_secret` runs on the raw user message *before* it reaches the model. If the configured
   passphrase appears in the message, it is replaced with a placeholder and the runner tells the
   verify tool out-of-band that the secret was supplied. The model, the message history, the
   provider, and Logfire therefore never see the secret on the normal path.
2. `redact` runs on every string that is persisted (replies, summaries, tool arguments and
   results, operation notes) as defense in depth for any channel the first layer misses, such as
   the model echoing a near-miss or a store name that contains the secret.
"""

from __future__ import annotations

import re
from typing import Any

STORED_PLACEHOLDER = "[passphrase redacted]"
MODEL_PLACEHOLDER = "[passphrase provided]"


def _pattern(secret: str) -> re.Pattern[str] | None:
    secret = secret.strip()
    if not secret:
        return None
    # Tolerate any whitespace run between the words of the secret, case-insensitively.
    words = [re.escape(w) for w in secret.split()]
    return re.compile(r"\s+".join(words), re.IGNORECASE)


def contains_secret(text: str, secret: str) -> bool:
    pattern = _pattern(secret)
    return bool(pattern and pattern.search(text))


def redact(text: str, secret: str, placeholder: str = STORED_PLACEHOLDER) -> str:
    pattern = _pattern(secret)
    return pattern.sub(placeholder, text) if pattern else text


def split_secret(user_text: str, secret: str) -> tuple[str, bool]:
    """Return (text safe to send to the model, whether the secret was present)."""
    if not contains_secret(user_text, secret):
        return user_text, False
    return redact(user_text, secret, MODEL_PLACEHOLDER), True


def redact_value(value: Any, secret: str) -> Any:
    """Recursively redact every string inside a JSON-like structure."""
    if isinstance(value, str):
        return redact(value, secret)
    if isinstance(value, dict):
        return {k: redact_value(v, secret) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_value(v, secret) for v in value]
    return value
