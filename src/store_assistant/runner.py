"""Orchestrates one conversation: turns, guardrails, termination, and the saved summary.

`ConversationRunner` is UI-agnostic. The CLI and the Streamlit page both drive it the same way:
call `start()`, then `send(text)` per user message, and read `TurnResult.ended` to know when the
session is over.
"""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from pydantic_ai import Agent, AgentRunResult, capture_run_messages
from pydantic_ai.exceptions import ContentFilterError, ModelAPIError, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage
from pydantic_core import to_jsonable_python

from store_assistant._async import run_sync
from store_assistant.agent import build_agent, model_label
from store_assistant.config import Settings, is_anthropic_model
from store_assistant.db import Repository, ToolCallRecord
from store_assistant.guardrails import (
    BudgetStatus,
    build_usage_limits,
    check_daily_budget,
    estimate_cost,
)
from store_assistant.redaction import STORED_PLACEHOLDER, redact, redact_value, split_secret
from store_assistant.session import EndReason, SessionState
from store_assistant.summarizer import build_summary_agent, summarize
from store_assistant.tools import Deps

log = logging.getLogger(__name__)

GREETING = (
    "Hi! I can save a store's name and phone number, or look up a saved store's phone. "
    "What would you like to do?"
)
MODEL_ERROR_REPLY = (
    "I had trouble reaching the language model just now. Please send that message again."
)
REFUSAL_REPLY = (
    "I can't process that message. I can save a store's phone number or look one up with the "
    "passphrase; what would you like to do?"
)


class DailyBudgetExceeded(RuntimeError):
    def __init__(self, status: BudgetStatus) -> None:
        self.status = status
        super().__init__(
            f"Daily spend limit reached: ${status.spent_today_usd:.4f} of "
            f"${status.daily_limit_usd:.2f} used today."
        )


@dataclass(frozen=True, slots=True)
class TurnResult:
    turn: int
    reply: str
    ended: bool
    end_reason: EndReason | None
    summary: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    tool_calls: tuple[ToolCallRecord, ...] = ()


def extract_tool_events(
    messages: list[ModelMessage], secret: str
) -> list[tuple[str, dict[str, Any], Any]]:
    """Pair each tool call with its result from a run's new messages, redacting the secret."""
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    events: list[tuple[str, dict[str, Any], Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            for response_part in message.parts:
                if isinstance(response_part, ToolCallPart):
                    args = redact_value(response_part.args_as_dict(), secret)
                    if response_part.tool_name == "verify_passphrase" and "passphrase" in args:
                        args["passphrase"] = STORED_PLACEHOLDER
                    pending[response_part.tool_call_id] = (response_part.tool_name, args)
        elif isinstance(message, ModelRequest):
            for request_part in message.parts:
                if (
                    isinstance(request_part, ToolReturnPart)
                    and request_part.tool_call_id in pending
                ):
                    name, args = pending.pop(request_part.tool_call_id)
                    result = redact_value(to_jsonable_python(request_part.content), secret)
                    events.append((name, args, result))
                elif (
                    isinstance(request_part, RetryPromptPart)
                    and request_part.tool_call_id in pending
                ):
                    name, args = pending.pop(request_part.tool_call_id)
                    detail = redact_value(to_jsonable_python(request_part.content), secret)
                    events.append((name, args, {"error": detail}))
    return events


class ConversationRunner:
    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        *,
        model: Model | str | None = None,
        summary_model: Model | str | None = None,
        fallback_model: Model | str | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.model_name = model_label(model if model is not None else settings.llm_model)
        self.agent = build_agent(settings, model)
        # Anthropic's safety classifiers can decline a request that looks like credential
        # guessing (several wrong passphrases in a row). When that happens the turn is retried
        # on a fallback model. Only configured for Anthropic primaries unless given explicitly.
        self._fallback_model: Model | str | None = fallback_model
        if (
            fallback_model is None
            and settings.refusal_fallback
            and is_anthropic_model(self.model_name)
        ):
            self._fallback_model = settings.fallback_model
        self._fallback_agent: Agent[Deps, str] | None = None
        self.summary_agent = build_summary_agent(settings, summary_model)
        self.usage = RunUsage()
        self.usage_limits = build_usage_limits(settings, self.model_name)
        self.history: list[ModelMessage] = []
        self.state: SessionState | None = None
        self.summary: str | None = None
        self._lock = threading.Lock()
        self._consecutive_errors = 0
        self._turn_model = self.model_name
        self._turn_messages: list[ModelMessage] = []

    @property
    def fallback_agent(self) -> Agent[Deps, str] | None:
        """Built on first use so a fallback that is never needed costs nothing."""
        if self._fallback_agent is None and self._fallback_model is not None:
            self._fallback_agent = build_agent(self.settings, self._fallback_model)
        return self._fallback_agent

    # ---- lifecycle ----

    @property
    def ended(self) -> bool:
        return self.state is not None and self.state.ended

    def start(self) -> str:
        """Open a conversation, enforcing the daily budget. Returns the greeting."""
        budget = check_daily_budget(self.repo, self.settings)
        if budget.exceeded:
            raise DailyBudgetExceeded(budget)
        conversation_id = self.repo.start_conversation(self.model_name)
        self.state = SessionState(conversation_id=conversation_id)
        self.repo.add_message(conversation_id, 0, "assistant", GREETING)
        return GREETING

    def send(self, user_text: str) -> TurnResult:
        """Process one user message and return the assistant's reply plus session status.

        Calls are serialized: a second send (for example a double submit in a web UI) waits for
        the first to finish instead of racing it.
        """
        with self._lock:
            return self._send(user_text)

    def _send(self, user_text: str) -> TurnResult:
        if self.state is None:
            raise RuntimeError("Call start() before send().")
        state = self.state
        if state.ended:
            raise RuntimeError("This conversation has already ended.")
        secret = self.settings.store_passphrase

        state.turn += 1
        self.repo.add_message(state.conversation_id, state.turn, "user", redact(user_text, secret))

        # The secret never reaches the model: it is swapped for a placeholder here and the
        # verify tool is told out-of-band that it was supplied.
        model_text, supplied = split_secret(user_text, secret)
        deps = Deps(
            settings=self.settings, repo=self.repo, state=state, passphrase_supplied=supplied
        )
        before = _snapshot(self.usage)

        try:
            result = self._run_turn(model_text, deps)
        except UsageLimitExceeded:
            reply = "We've reached this session's usage budget, so I'm wrapping up here."
            state.end("budget_limit")
        except ContentFilterError as exc:
            # Declined by the primary model and by the fallback (if any). Per-message event,
            # not a session failure.
            log.warning("content filter on turn %s: %s", state.turn, exc)
            reply = REFUSAL_REPLY
        except ModelAPIError as exc:
            log.warning("model request failed on turn %s: %s", state.turn, exc)
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.settings.max_consecutive_model_errors:
                reply = "I keep failing to reach the language model, so I'm ending the session."
                state.end("error")
            else:
                reply = MODEL_ERROR_REPLY
        except Exception:
            log.exception("unexpected error on turn %s", state.turn)
            reply = "Something went wrong on my side, so I'm ending the session."
            state.end("error")
        else:
            self._consecutive_errors = 0
            reply = result.output.strip() or "(no reply)"
            self.history = result.all_messages()
        finally:
            self._log_usage_delta(before, self._turn_model)
            # Tool calls are persisted even when the turn failed afterwards, since their
            # effects on the database already happened.
            for name, args, outcome in extract_tool_events(self._turn_messages, secret):
                self.repo.add_tool_call(state.conversation_id, state.turn, name, args, outcome)

        reply = redact(reply, secret)
        if not state.ended and state.turn >= self.settings.max_turns_per_session:
            state.end("turn_limit")
            reply += " We've reached the maximum length for this session, so I'm wrapping up."
        self.repo.add_message(state.conversation_id, state.turn, "assistant", reply)

        summary = self._finish(state.end_reason or "error") if state.ended else None
        return TurnResult(
            turn=state.turn,
            reply=reply,
            ended=state.ended,
            end_reason=state.end_reason,
            summary=summary,
            input_tokens=self.usage.input_tokens + self.usage.cache_read_tokens,
            output_tokens=self.usage.output_tokens,
            cost_usd=self.session_cost(),
            tool_calls=tuple(
                c for c in self.repo.get_tool_calls(state.conversation_id) if c.turn == state.turn
            ),
        )

    def _run_turn(self, model_text: str, deps: Deps) -> AgentRunResult[str]:
        """Run one turn, retrying on the fallback model if the primary is declined."""
        state_snapshot = copy.deepcopy(deps.state)
        self._turn_model = self.model_name
        self._turn_messages = []
        try:
            return self._run_agent(self.agent, model_text, deps)
        except ContentFilterError as exc:
            fallback = self.fallback_agent
            if fallback is None:
                raise
            log.warning(
                "primary model declined turn %s (%s); retrying on fallback", deps.state.turn, exc
            )
            # Undo any state the refused run touched (for example a counted passphrase attempt)
            # so the retry starts from the same point. Database writes by tools that ran before
            # the refusal are not rolled back; the retry sees them (for example as
            # needs_confirmation), which is safe.
            for field_name in deps.state.__dataclass_fields__:
                setattr(deps.state, field_name, getattr(state_snapshot, field_name))
            self._turn_model = model_label(self._fallback_model or self.model_name)
            self._turn_messages = []
            return self._run_agent(fallback, model_text, deps)

    def _run_agent(
        self, agent: Agent[Deps, str], model_text: str, deps: Deps
    ) -> AgentRunResult[str]:
        """Run the agent on the background loop, keeping the messages even if it raises."""

        async def go() -> tuple[
            AgentRunResult[str] | None, list[ModelMessage], BaseException | None
        ]:
            with capture_run_messages() as messages:
                try:
                    result = await agent.run(
                        model_text,
                        deps=deps,
                        usage=self.usage,
                        usage_limits=self.usage_limits,
                        message_history=self.history or None,
                    )
                except Exception as exc:  # re-raised below, after the messages are kept
                    return None, list(messages), exc
            return result, list(messages), None

        result, messages, error = run_sync(go())
        new_messages = messages[len(self.history) :] if self.history else messages
        self._turn_messages = new_messages
        if error is not None:
            raise error
        assert result is not None
        return result

    def finish(self, reason: EndReason) -> str:
        """End the session, write the summary to the database, and return it."""
        with self._lock:
            return self._finish(reason)

    def _finish(self, reason: EndReason) -> str:
        state = self.state
        if state is None:
            raise RuntimeError("Call start() before finish().")
        if self.summary is not None:
            return self.summary
        state.end(reason)
        transcript = self.repo.get_transcript(state.conversation_id)
        before = _snapshot(self.usage)
        try:
            summary = summarize(
                self.summary_agent,
                transcript,
                state,
                state.end_reason,
                usage=self.usage,
                usage_limits=self.usage_limits,
            )
        finally:
            self._log_usage_delta(before)
        summary = redact(summary, self.settings.store_passphrase)
        self.summary = summary
        self.repo.end_conversation(
            state.conversation_id,
            end_reason=state.end_reason or reason,
            summary=summary,
            input_tokens=self.usage.input_tokens + self.usage.cache_read_tokens,
            output_tokens=self.usage.output_tokens,
            estimated_cost_usd=self.session_cost(),
        )
        return summary

    def close(self) -> None:
        """Finish an unfinished session as aborted (no-op if already finished)."""
        if self.state is not None and self.summary is None:
            self.finish("aborted")

    # ---- helpers ----

    def session_cost(self) -> Decimal:
        return estimate_cost(self.usage, self.model_name)

    def _log_usage_delta(self, before: RunUsage, model: str | None = None) -> None:
        delta = self.usage - before
        if delta.input_tokens == 0 and delta.output_tokens == 0 and delta.cache_read_tokens == 0:
            return
        assert self.state is not None
        model = model or self.model_name
        self.repo.log_usage(
            self.state.conversation_id,
            model=model,
            input_tokens=delta.input_tokens,
            cache_read_tokens=delta.cache_read_tokens,
            cache_write_tokens=delta.cache_write_tokens,
            output_tokens=delta.output_tokens,
            estimated_cost_usd=estimate_cost(delta, model),
        )


def _snapshot(usage: RunUsage) -> RunUsage:
    return replace(usage, details=dict(usage.details))
