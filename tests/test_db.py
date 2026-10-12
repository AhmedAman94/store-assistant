from datetime import UTC, datetime, timedelta
from decimal import Decimal

from store_assistant.db import Repository


def test_save_and_get_store_is_case_insensitive(repo: Repository) -> None:
    record, created = repo.save_store("Trader Joe's", "+14155552671")
    assert created
    assert repo.get_store("trader joe's") == record
    assert repo.get_store("  TRADER  JOE'S ") == record
    assert repo.get_store("Safeway") is None


def test_saving_existing_store_updates_phone(repo: Repository) -> None:
    repo.save_store("Whole Foods", "+12125550199")
    record, created = repo.save_store("whole foods", "+16465550123")
    assert not created
    assert record.phone_e164 == "+16465550123"
    assert len(repo.list_stores()) == 1


def test_conversation_lifecycle_and_transcript(repo: Repository) -> None:
    conversation_id = repo.start_conversation("test:model")
    repo.add_message(conversation_id, 1, "user", "hello")
    repo.add_message(conversation_id, 1, "assistant", "hi")
    repo.end_conversation(
        conversation_id,
        end_reason="user_done",
        summary="Nothing happened.",
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=Decimal("0.001"),
    )
    conversation = repo.get_conversation(conversation_id)
    assert conversation is not None
    assert conversation.end_reason == "user_done"
    assert conversation.summary == "Nothing happened."
    assert conversation.estimated_cost_usd == Decimal("0.001")
    assert [m.role for m in repo.get_transcript(conversation_id)] == ["user", "assistant"]


def test_cost_today_sums_only_todays_usage(repo: Repository) -> None:
    conversation_id = repo.start_conversation("test:model")
    for cost in ("0.10", "0.25"):
        repo.log_usage(
            conversation_id,
            model="test:model",
            input_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
            output_tokens=1,
            estimated_cost_usd=Decimal(cost),
        )
    assert repo.cost_today() == Decimal("0.35")
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).isoformat(timespec="seconds")
    assert repo.cost_since(tomorrow) == Decimal("0")
