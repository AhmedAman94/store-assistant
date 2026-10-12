"""Opt-in smoke test against the real model. Costs a few cents.

RUN_LIVE_TESTS=1 uv run pytest tests/test_live.py -s
"""

import os

import pytest

from store_assistant.config import Settings
from store_assistant.db import Repository
from store_assistant.runner import ConversationRunner

pytestmark = pytest.mark.skipif(
    not (os.getenv("RUN_LIVE_TESTS") and os.getenv("ANTHROPIC_API_KEY")),
    reason="set RUN_LIVE_TESTS=1 and ANTHROPIC_API_KEY to run",
)


def test_live_save_and_retrieve(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        _env_file=None, db_path=tmp_path / "live.db", store_passphrase="open sesame"
    )  # type: ignore[call-arg]
    repo = Repository(settings.db_path)
    runner = ConversationRunner(settings, repo)
    runner.start()
    runner.send("Please save a store called Blue Bottle Coffee, phone (415) 555-0142.")
    store = repo.get_store("Blue Bottle Coffee")
    assert store is not None and store.phone_e164 == "+14155550142"

    runner.send("What's the phone number for Blue Bottle Coffee?")
    reply = runner.send("open sesame").reply
    assert "555" in reply and "0142" in reply

    result = runner.send("I'm done, thanks.")
    assert result.ended and result.end_reason == "user_done"
    assert result.summary
    print("\nSummary:", result.summary, "\nCost: $", result.cost_usd)
    repo.close()
