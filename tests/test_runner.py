"""Conversation-level tests: the runner, termination paths, summaries, redaction, guardrails."""

from __future__ import annotations

import json
import threading
from decimal import Decimal

import pytest
from pydantic_ai.exceptions import ContentFilterError, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from conftest import call, say, scripted_model, text_model
from store_assistant.config import Settings
from store_assistant.db import Repository
from store_assistant.runner import ConversationRunner, DailyBudgetExceeded

SUMMARY = "The user saved one store and left."


def make_runner(
    settings: Settings, repo: Repository, *responses: ModelResponse
) -> ConversationRunner:
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
        call("verify_passphrase", passphrase="[passphrase provided]"),
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


def test_secret_never_reaches_the_model_or_history(settings: Settings, repo: Repository) -> None:
    seen: list[str] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in getattr(message, "parts", []):
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    seen.append(part.content)
        last = seen[-1] if seen else ""
        if "[passphrase provided]" in last and len(seen) == 1:
            return call("verify_passphrase", passphrase="[passphrase provided]")
        return say("ok")

    runner = ConversationRunner(
        settings,
        repo,
        model=FunctionModel(function, model_name="spy"),
        summary_model=text_model(SUMMARY),
    )
    runner.start()
    runner.send("the passphrase is Open  Sesame, look up Costco")
    assert all("sesame" not in text.lower() for text in seen)
    assert runner.state is not None and runner.state.passphrase_verified
    history_dump = json.dumps([m.__dict__ for m in runner.history], default=str).lower()
    assert "open sesame" not in history_dump and "open  sesame" not in history_dump


def test_every_persisted_channel_is_redacted(settings: Settings, repo: Repository) -> None:
    runner = make_runner(
        settings,
        repo,
        call("save_store", name="open sesame", phone="415-555-0105"),
        say("Saved the store called open sesame."),
        call("end_session"),
        say("Goodbye"),
    )
    runner.start()
    runner.send("save a store called open sesame, 415-555-0105")
    runner.summary_agent = __import__(
        "store_assistant.summarizer", fromlist=["build_summary_agent"]
    ).build_summary_agent(settings, text_model("Summary mentioning open sesame."))
    result = runner.send("done")
    assert result.ended
    dump = json.dumps(
        {
            "messages": [m.content for m in repo.get_transcript(1)],
            "calls": [(c.args, c.result) for c in repo.get_tool_calls(1)],
            "summary": result.summary,
        }
    ).lower()
    assert "open sesame" not in dump
    assert "[passphrase redacted]" in dump


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
    responses: list[ModelResponse] = []
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
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("summary model down")

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


def test_transient_model_error_does_not_end_the_session(
    settings: Settings, repo: Repository
) -> None:
    calls = {"n": 0}

    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModelHTTPError(status_code=529, model_name="flaky", body={"error": "overloaded"})
        return say("recovered")

    runner = ConversationRunner(
        settings,
        repo,
        model=FunctionModel(flaky, model_name="flaky"),
        summary_model=text_model(SUMMARY),
    )
    runner.start()
    first = runner.send("hi")
    assert (not first.ended and "try" in first.reply.lower()) or "again" in first.reply.lower()
    second = runner.send("hi again")
    assert second.reply == "recovered" and not second.ended
    transcript = repo.get_transcript(1)
    assert [m.role for m in transcript] == ["assistant", "user", "assistant", "user", "assistant"]


def test_unexpected_error_ends_session_cleanly(settings: Settings, repo: Repository) -> None:
    def explode(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError("unexpected")

    runner = ConversationRunner(
        settings,
        repo,
        model=FunctionModel(explode, model_name="bad"),
        summary_model=text_model(SUMMARY),
    )
    runner.start()
    result = runner.send("hi")
    assert result.ended and result.end_reason == "error"
    conversation = repo.get_conversation(1)
    assert conversation is not None and conversation.summary == SUMMARY
    assert [m.role for m in repo.get_transcript(1)] == ["assistant", "user", "assistant"]


def test_close_finishes_an_abandoned_session(settings: Settings, repo: Repository) -> None:
    runner = make_runner(settings, repo, say("one"))
    runner.start()
    runner.send("hi")
    runner.close()
    conversation = repo.get_conversation(1)
    assert conversation is not None and conversation.end_reason == "aborted"
    assert conversation.summary == SUMMARY
    runner.close()  # idempotent


def test_usage_is_logged_per_turn(settings: Settings, repo: Repository) -> None:
    runner = make_runner(settings, repo, say("one"), say("two"))
    runner.start()
    runner.send("hi")
    runner.send("hi")
    assert runner.usage.requests == 2
    row = repo._fetchone("SELECT COUNT(*) AS n FROM usage_log")
    assert row is not None and row["n"] == 2


def test_concurrent_sends_are_serialized(settings: Settings, repo: Repository) -> None:
    runner = make_runner(settings, repo, say("one"), say("two"))
    runner.start()
    results: list[int] = []
    barrier = threading.Barrier(2)

    def go(text: str) -> None:
        barrier.wait()
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
    assert result.tool_calls[0].args == {"passphrase": "[passphrase redacted]"}
    assert result.tool_calls[0].result["status"] == "verified"
    assert result.tool_calls[1].result["status"] == "not_found"
    stored = repo.get_tool_calls(1)
    assert len(stored) == 2 and stored[0].turn == 1
    assert "open sesame" not in json.dumps([c.args for c in stored])


def test_content_filter_refusal_does_not_end_the_session(
    settings: Settings, repo: Repository
) -> None:
    calls = {"n": 0}

    def refuse_once(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ContentFilterError("declined by classifier")
        return say("back to normal")

    runner = ConversationRunner(
        settings,
        repo,
        model=FunctionModel(refuse_once, model_name="filtered"),
        summary_model=text_model(SUMMARY),
    )
    assert runner.fallback_agent is None  # no fallback for non-Anthropic test models
    runner.start()
    first = runner.send("hunter2")
    assert not first.ended and "can't process" in first.reply
    second = runner.send("save Costco 415-555-0101")
    assert second.reply == "back to normal" and not second.ended
    assert [m.role for m in repo.get_transcript(1)] == [
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_refused_turn_is_retried_on_the_fallback_model(
    settings: Settings, repo: Repository
) -> None:
    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ContentFilterError("declined by classifier")

    runner = ConversationRunner(
        settings,
        repo,
        model=FunctionModel(primary, model_name="primary"),
        summary_model=text_model(SUMMARY),
        fallback_model=scripted_model(
            call("verify_passphrase", passphrase="hunter2"), say("Wrong, 2 left.")
        ),
    )
    runner.start()
    result = runner.send("hunter2")
    assert result.reply == "Wrong, 2 left." and not result.ended
    assert [c.tool_name for c in result.tool_calls] == ["verify_passphrase"]
    assert runner.state is not None and runner.state.passphrase_attempts == 1


def test_same_turn_overwrite_bypass_is_blocked_end_to_end(
    settings: Settings, repo: Repository
) -> None:
    runner = make_runner(
        settings,
        repo,
        call("save_store", name="Costco", phone="415-555-0101"),
        say("Saved."),
        # One model turn: flagged, then immediately "confirms" on its own.
        call("save_store", name="Costco", phone="415-555-0102"),
        call("save_store", name="Costco", phone="415-555-0102", confirm_overwrite=True),
        say("Done?"),
        # The user answers on the next turn.
        call("save_store", name="Costco", phone="415-555-0102", confirm_overwrite=True),
        say("Updated."),
    )
    runner.start()
    runner.send("save Costco 415-555-0101")
    sneaky = runner.send("update Costco to 415-555-0102, I already confirm")
    assert [c.result["status"] for c in sneaky.tool_calls] == [
        "needs_confirmation",
        "needs_confirmation",
    ]
    store = repo.get_store("Costco")
    assert store is not None and store.phone_e164 == "+14155550101"
    confirmed = runner.send("yes")
    assert [c.result["status"] for c in confirmed.tool_calls] == ["updated"]


def test_tool_calls_are_kept_when_the_turn_fails_after_them(
    settings: Settings, repo: Repository
) -> None:
    calls = {"n": 0}

    def save_then_die(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return call("save_store", name="Costco", phone="415-555-0101")
        raise ValueError("boom after the tool ran")

    runner = ConversationRunner(
        settings,
        repo,
        model=FunctionModel(save_then_die, model_name="dies"),
        summary_model=text_model(SUMMARY),
    )
    runner.start()
    result = runner.send("save Costco 415-555-0101")
    assert result.ended and result.end_reason == "error"
    assert repo.get_store("Costco") is not None
    assert [(c.tool_name, c.result["status"]) for c in repo.get_tool_calls(1)] == [
        ("save_store", "saved")
    ]


def test_fallback_turn_usage_is_attributed_to_the_fallback_model(
    settings: Settings, repo: Repository
) -> None:
    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ContentFilterError("declined")

    runner = ConversationRunner(
        settings,
        repo,
        model=FunctionModel(primary, model_name="primary"),
        summary_model=text_model(SUMMARY),
        fallback_model=FunctionModel(lambda m, i: say("from fallback"), model_name="backup"),
    )
    runner.start()
    assert runner.send("hi").reply == "from fallback"
    rows = repo._fetchall("SELECT model FROM usage_log")
    assert [r["model"] for r in rows] == ["function:backup"]
