"""Logfire tracing setup.

With a `LOGFIRE_TOKEN` in the environment, every agent run, model request, and tool call is sent
to Logfire where past conversations can be inspected. Without a token the instrumentation is a
no-op, so the app runs identically offline.
"""

from __future__ import annotations

import os

import logfire

from store_assistant.config import Settings


def configure_tracing(settings: Settings) -> bool:
    """Configure Logfire. Returns True when traces are being exported."""
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    token = settings.logfire_token or os.getenv("LOGFIRE_TOKEN")
    logfire.configure(
        send_to_logfire="if-token-present",
        token=token,
        console=False,
        service_name="store-assistant",
        scrubbing=logfire.ScrubbingOptions(extra_patterns=["passphrase"]),
    )
    logfire.instrument_pydantic_ai()
    return bool(token)
