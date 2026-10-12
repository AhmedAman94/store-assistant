"""Conversation-level tests: the runner, termination paths, summaries, and spend guardrails."""

import json
from decimal import Decimal

import pytest

from conftest import call, say, scripted_model, text_model
from store_assistant.config import Settings
from store_assistant.db import Repository
from store_assistant.runner import ConversationRunner, DailyBudgetExceeded

SUMMARY = "The user saved one store and left."


def make_runner(settings: Settings, repo: Repository, *responses) -> ConversationRunner:  # type: ignore[no-untyped-def]
    return ConversationRunner(
        settings, repo, model=scripted_model(*responses), summary_model=text_model(SUMMARY)
    )


def test_save_then_retrieve_round_trip_with_passphrase(
    settings: Settings, repo: Repository
) -> None:
    runner = make_runner(
        settings,
        repo,
        call("save_store", name="Trader Joe's", phone="415-555-2671"),
        say("Saved Trader Joe's."),
        call("get_store", name="Trader Joe's"),
        say("Please enter the passphrase."),
        call("verify_passphrase", passphrase="open sesame"),
        call("get_store", name="trader joe's"),
        say("Trader Joe's phone is +14155552671."),
    )
    runner.start()
    first = runner.send("Save Trader Joe's, 415-555-2671")
    assert first.reply == "Saved Trader Joe's." and not first.ended
    second = runner.send("What's Trader Joe's number?")
    assert "passphrase" in second.reply.lower()
    third = runner.send("open sesame")
    assert "+14155552671" in third.reply

    store = repo.get_store("Trader Joe's")
    assert store is not None and store.phone_e164 == "+14155552671"
    user_messages = [m.content for m in repo.get_transcript(1) if m.role == "user"]
    assert user_messages[-1] == "[passphrase redacted]", "passphrase must not be stored"
    assert len(user_messages) == 3


def test_user_done_ends_session_and_saves_summary(settings: Settings, repo: Repository) -> None:
    runner = make_runner(settings, repo, call("end_session"), say("Goodbye!"))
    runner.start()
    result = runner.send("I'm done")
    assert result.ended and result.end_reason == "user_done"
    assert result.summary == SUMMARY
    conversation = repo.get_conversation(1)
    assert conversation is not None
    assert conversation.end_reason == "user_done"
    assert conversation.summary == SUMMARY
    assert conversation.ended_at is not None
    with pytest.raises(RuntimeError):
        runner.send("anything")


def test_repeated_off_topic_ends_session(settings: Settings, repo: Repository) -> None:
    responses = []
    for _ in range(3):
        responses += [call("flag_off_topic"), say("I can only help with stores.")]
    runner = make_runner(settings, repo, *responses)
    runner.start()
    assert not runner.send("What's the weather?").ended
    assert not runner.send("Tell me a joke").ended
    final = runner.send("Who won the game?")
    assert final.ended and final.end_reason == "off_scope"
    conversation = repo.get_conversation(1)
    assert conversation is not None and conversation.end_reason == "off_scope"


def test_turn_limit_ends_session(settings: Settings, repo: Repository) -> None:
    settings.max_turns_per_session = 2
    runner = make_runner(settings, repo, say("one"), say("two"))
    runner.start()
    assert not runner.send("hi").ended
    result = runner.send("hi again")
    assert result.ended and result.end_reason == "turn_limit"
    assert "maximum length" in result.reply


def test_request_limit_ends_session_with_budget_reason(
    settings: Settings, repo: Repository
) -> None:
    settings.max_requests_per_session = 1
    runner = make_runner(settings, repo, say("one"), say("two"))
    runner.start()
    assert not runner.send("hi").ended
    result = runner.send("hi again")
    assert result.ended and result.end_reason == "budget_limit"
    assert "budget" in result.reply
    conversation = repo.get_conversation(1)
    assert conversation is not None and conversation.summary  # fallback summary still saved


def test_daily_budget_blocks_new_sessions(settings: Settings, repo: Repository) -> None:
    settings.daily_cost_limit_usd = Decimal("1.00")
    conversation_id = repo.start_conversation("x")
    repo.log_usage(
        conversation_id,
        model="x",
        input_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        output_tokens=1,
        estimated_cost_usd=Decimal("1.00"),
    )
    runner = make_runner(settings, repo, say("hi"))
    with pytest.raises(DailyBudgetExceeded) as excinfo:
        runner.start()
    assert excinfo.value.status.exceeded


def test_summary_falls_back_when_summary_model_fails(settings: Settings, repo: Repository) -> None:
    def boom(messages, info):  # type: ignore[no-untyped-def]
        raise RuntimeError("summary model down")

    from pydantic_ai.models.function import FunctionModel

    runner = ConversationRunner(
        settings,
        repo,
        model=scripted_model(
            call("save_store", name="Costco", phone="415-555-0101"),
            say("Saved."),
            call("end_session"),
            say("Bye"),
        ),
        summary_model=FunctionModel(boom, model_name="broken"),
    )
    runner.start()
    runner.send("save Costco 415-555-0101")
    result = runner.send("done")
    assert result.ended
    assert result.summary is not None
    assert "Costco" in result.summary and "finished" in result.summary


def test_usage_is_logged_per_turn(settings: Settings, repo: Repository) -> None:
    runner = make_runner(settings, repo, say("one"), say("two"))
    runner.start()
    runner.send("hi")
    runner.send("hi")
    assert runner.usage.requests == 2
    rows = repo._execute("SELECT COUNT(*) AS n FROM usage_log").fetchone()
    assert rows["n"] == 2


def test_concurrent_sends_are_serialized(settings: Settings, repo: Repository) -> None:
    import threading

    runner = make_runner(settings, repo, say("one"), say("two"))
    runner.start()
    results: list[int] = []

    def go(text: str) -> None:
        results.append(runner.send(text).turn)

    threads = [threading.Thread(target=go, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [1, 2]
    transcript = repo.get_transcript(1)
    assert [(m.turn, m.role) for m in transcript] == [
        (0, "assistant"),
        (1, "user"),
        (1, "assistant"),
        (2, "user"),
        (2, "assistant"),
    ]


def test_tool_calls_are_persisted_with_passphrase_redacted(
    settings: Settings, repo: Repository
) -> None:
    runner = make_runner(
        settings,
        repo,
        call("verify_passphrase", passphrase="open sesame"),
        call("get_store", name="Nowhere"),
        say("Not found."),
    )
    runner.start()
    result = runner.send("look up Nowhere, passphrase open sesame")
    assert [c.tool_name for c in result.tool_calls] == ["verify_passphrase", "get_store"]
    assert result.tool_calls[0].args == {"passphrase": "[redacted]"}
    assert result.tool_calls[0].result["status"] == "verified"
    assert result.tool_calls[1].result["status"] == "not_found"
    stored = repo.get_tool_calls(1)
    assert len(stored) == 2 and stored[0].turn == 1
    assert "open sesame" not in json.dumps([c.args for c in stored])
