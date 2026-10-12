"""Orchestrates one conversation: turns, guardrails, termination, and the saved summary.

`ConversationRunner` is UI-agnostic. The CLI and the Streamlit page both drive it the same way:
call `start()`, then `send(text)` per user message, and read `TurnResult.ended` to know when the
session is over.
"""

from __future__ import annotations

import threading
from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic_ai.exceptions import ModelAPIError, UsageLimitExceeded
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
from store_assistant.agent import build_agent
from store_assistant.config import Settings
from store_assistant.db import Repository, ToolCallRecord
from store_assistant.guardrails import (
    BudgetStatus,
    build_usage_limits,
    check_daily_budget,
    estimate_cost,
)
from store_assistant.session import EndReason, SessionState
from store_assistant.summarizer import build_summary_agent, summarize
from store_assistant.tools import Deps

GREETING = (
    "Hi! I can save a store's name and phone number, or look up a saved store's phone. "
    "What would you like to do?"
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


REDACTED_ARGS = {"passphrase"}


def extract_tool_events(messages: list[ModelMessage]) -> list[tuple[str, dict[str, Any], Any]]:
    """Pair each tool call with its result from a run's new messages. Secrets are redacted."""
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    events: list[tuple[str, dict[str, Any], Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            for response_part in message.parts:
                if isinstance(response_part, ToolCallPart):
                    args = {
                        k: ("[redacted]" if k in REDACTED_ARGS else v)
                        for k, v in response_part.args_as_dict().items()
                    }
                    pending[response_part.tool_call_id] = (response_part.tool_name, args)
        elif isinstance(message, ModelRequest):
            for request_part in message.parts:
                if (
                    isinstance(request_part, ToolReturnPart)
                    and request_part.tool_call_id in pending
                ):
                    name, args = pending.pop(request_part.tool_call_id)
                    events.append((name, args, to_jsonable_python(request_part.content)))
                elif (
                    isinstance(request_part, RetryPromptPart)
                    and request_part.tool_call_id in pending
                ):
                    name, args = pending.pop(request_part.tool_call_id)
                    events.append((name, args, {"error": to_jsonable_python(request_part.content)}))
    return events


class ConversationRunner:
    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        *,
        model: Model | str | None = None,
        summary_model: Model | str | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.model_name = _model_label(model if model is not None else settings.llm_model)
        self.agent = build_agent(settings, model)
        self.summary_agent = build_summary_agent(settings, summary_model)
        self.usage = RunUsage()
        self.usage_limits = build_usage_limits(settings, self.model_name)
        self.history: list[ModelMessage] = []
        self.state: SessionState | None = None
        self.summary: str | None = None
        self._lock = threading.Lock()

    # ---- lifecycle ----

    @property
    def started(self) -> bool:
        return self.state is not None

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
            self.start()
        state = self.state
        assert state is not None
        if state.ended:
            raise RuntimeError("This conversation has already ended.")

        state.turn += 1
        self.repo.add_message(state.conversation_id, state.turn, "user", self._redact(user_text))
        deps = Deps(settings=self.settings, repo=self.repo, state=state)
        before = copy(self.usage)

        try:
            result = run_sync(
                self.agent.run(
                    user_text,
                    deps=deps,
                    usage=self.usage,
                    usage_limits=self.usage_limits,
                    message_history=self.history or None,
                )
            )
        except UsageLimitExceeded:
            reply = "We've reached this session's usage budget, so I'm wrapping up here."
            state.end("budget_limit")
        except ModelAPIError as exc:
            reply = (
                f"I hit a problem talking to the language model ({exc.__class__.__name__}). "
                "Ending the session."
            )
            state.end("error")
        else:
            reply = result.output.strip() or "(no reply)"
            self.history = result.all_messages()
            for name, args, outcome in extract_tool_events(result.new_messages()):
                self.repo.add_tool_call(state.conversation_id, state.turn, name, args, outcome)

        self._log_usage_delta(before)
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

    def finish(self, reason: EndReason) -> str:
        """End the session, write the summary to the database, and return it."""
        with self._lock:
            return self._finish(reason)

    def _finish(self, reason: EndReason) -> str:
        state = self.state
        assert state is not None
        if self.summary is not None:
            return self.summary
        state.end(reason)
        transcript = self.repo.get_transcript(state.conversation_id)
        before = copy(self.usage)
        summary = summarize(
            self.summary_agent,
            transcript,
            state,
            state.end_reason,
            usage=self.usage,
            usage_limits=self.usage_limits,
        )
        self._log_usage_delta(before)
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

    # ---- helpers ----

    def session_cost(self) -> Decimal:
        return estimate_cost(self.usage, self.model_name)

    def _log_usage_delta(self, before: RunUsage) -> None:
        delta = self.usage - before
        if delta.input_tokens == 0 and delta.output_tokens == 0 and delta.cache_read_tokens == 0:
            return
        assert self.state is not None
        self.repo.log_usage(
            self.state.conversation_id,
            model=self.model_name,
            input_tokens=delta.input_tokens,
            cache_read_tokens=delta.cache_read_tokens,
            cache_write_tokens=delta.cache_write_tokens,
            output_tokens=delta.output_tokens,
            estimated_cost_usd=estimate_cost(delta, self.model_name),
        )

    def _redact(self, text: str) -> str:
        """Keep the passphrase out of the stored transcript."""
        secret = self.settings.store_passphrase.strip()
        if secret and secret.casefold() in text.casefold():
            import re

            return re.sub(re.escape(secret), "[passphrase redacted]", text, flags=re.IGNORECASE)
        return text


def _model_label(model: Model | str) -> str:
    return model if isinstance(model, str) else f"{model.system}:{model.model_name}"
