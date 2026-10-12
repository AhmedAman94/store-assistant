"""End-of-session summary generation.

The summary is written by a (configurable) model from the redacted transcript and the session's
operations log. If the model call fails or the budget is exhausted, a templated summary built
from the operations log is used so the database row is never left empty.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from store_assistant._async import run_sync
from store_assistant.agent import resolve_model
from store_assistant.config import Settings
from store_assistant.db import MessageRecord
from store_assistant.session import EndReason, SessionState

SUMMARY_INSTRUCTIONS = """\
You summarize a finished conversation between a user and Store Assistant, an agent that saves
and retrieves store phone numbers. Write two to four plain sentences covering: which stores were
saved or updated (with their phone numbers), which lookups were attempted and whether they
succeeded, any validation or passphrase problems, and why the conversation ended.
Never include the passphrase itself. Output only the summary text.
"""

END_REASON_TEXT: dict[str, str] = {
    "user_done": "the user said they were finished",
    "off_scope": "the user repeatedly went off-topic",
    "turn_limit": "the session reached its maximum number of turns",
    "budget_limit": "the session reached its usage budget",
    "aborted": "the user closed the session",
    "error": "an error occurred",
}


def build_summary_agent(settings: Settings, model: Model | str | None = None) -> Agent[None, str]:
    chosen = resolve_model(
        settings, model if model is not None else settings.effective_summary_model
    )
    return Agent(
        chosen,
        output_type=str,
        instructions=SUMMARY_INSTRUCTIONS,
        model_settings={"max_tokens": 400},
        name="store_assistant_summarizer",
    )


def format_transcript(transcript: list[MessageRecord]) -> str:
    lines = []
    for message in transcript:
        speaker = "User" if message.role == "user" else "Assistant"
        lines.append(f"{speaker}: {message.content}")
    return "\n".join(lines) if lines else "(no messages)"


def build_summary_prompt(
    transcript: list[MessageRecord], state: SessionState, end_reason: EndReason | None
) -> str:
    operations = "\n".join(f"- {op}" for op in state.operations) or "- (none)"
    why = END_REASON_TEXT.get(end_reason or "", "unknown reason")
    return (
        f"Transcript:\n{format_transcript(transcript)}\n\n"
        f"Operations performed:\n{operations}\n\n"
        f"The conversation ended because {why}."
    )


def fallback_summary(state: SessionState, end_reason: EndReason | None) -> str:
    why = END_REASON_TEXT.get(end_reason or "", "unknown reason")
    if state.operations:
        ops = "; ".join(state.operations)
        return f"Operations: {ops}. The conversation ended because {why}."
    return f"No store operations were completed. The conversation ended because {why}."


def summarize(
    agent: Agent[None, str],
    transcript: list[MessageRecord],
    state: SessionState,
    end_reason: EndReason | None,
    *,
    usage: RunUsage | None = None,
    usage_limits: UsageLimits | None = None,
) -> str:
    """Return a model-written summary, or the templated fallback if the model call fails."""
    prompt = build_summary_prompt(transcript, state, end_reason)
    try:
        result = run_sync(agent.run(prompt, usage=usage, usage_limits=usage_limits))
    except Exception:  # budget exhausted, network error, provider error
        return fallback_summary(state, end_reason)
    text = result.output.strip()
    return text or fallback_summary(state, end_reason)
