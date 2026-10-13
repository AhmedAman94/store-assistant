"""Streamlit chat UI. Run with: uv run streamlit run src/store_assistant/web.py

Reuses ConversationRunner exactly as the CLI does. Set ANTHROPIC_API_KEY in .env for the real
model, or tick "Demo mode" to run with no API calls. Alongside the chat it shows what the agent
did each turn (tool calls and results), live session stats against the guardrails, the saved
stores, and the history of past conversations with their summaries.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

from store_assistant.config import Settings
from store_assistant.db import Repository, ToolCallRecord
from store_assistant.demo_model import make_demo_model, make_demo_summary_model
from store_assistant.runner import ConversationRunner, DailyBudgetExceeded
from store_assistant.tracing import configure_tracing
from store_assistant.validation import mask_phone

ROOT = Path(__file__).resolve().parents[2]
LOGO = ROOT / "assets" / "ndr_logo_clean.png"
NAVY = "#1E2A5A"
RED = "#A6322B"

st.set_page_config(
    page_title="Store Assistant",
    page_icon=str(LOGO) if LOGO.exists() else "📞",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    f"""
    <style>
      [data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; }}
      .block-container {{ padding-top: 1.2rem; max-width: 1100px; }}
      h1 {{ color: {NAVY}; letter-spacing: 0.02em; }}
      .tool-pill {{ display:inline-block; padding:2px 10px; border-radius:999px;
                    background:{NAVY}; color:white; font-size:0.78rem; margin-right:6px; }}
      .status-ok {{ color:#1b7f3b; font-weight:600; }}
      .status-bad {{ color:{RED}; font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource  # type: ignore[untyped-decorator,unused-ignore]
def _tracing() -> bool:
    return configure_tracing(Settings())


def _new_runner(demo: bool) -> ConversationRunner:
    settings = Settings()
    repo = Repository(settings.db_path)
    if demo:
        return ConversationRunner(
            settings, repo, model=make_demo_model(), summary_model=make_demo_summary_model()
        )
    return ConversationRunner(settings, repo)


def _reset(demo: bool) -> None:
    previous: ConversationRunner | None = st.session_state.get("runner")
    if previous is not None:
        previous.close()  # finish the abandoned session as "aborted" so it gets a summary
        previous.repo.close()
    runner = _new_runner(demo)
    try:
        runner.start()
    except DailyBudgetExceeded as exc:
        st.session_state.runner = None
        st.session_state.error = str(exc)
        return
    st.session_state.runner = runner
    st.session_state.error = None


def _status(value: Any) -> str:
    return str(value.get("status", "")) if isinstance(value, dict) else str(value)


def _render_tool_calls(calls: list[ToolCallRecord], expanded: bool = False) -> None:
    if not calls:
        return
    label = f"🔧 {len(calls)} tool call{'s' if len(calls) != 1 else ''} this turn"
    with st.expander(label, expanded=expanded):
        for c in calls:
            status = _status(c.result)
            good = status in {"saved", "updated", "verified", "already_verified", "found", "ended"}
            css = "status-ok" if good else "status-bad"
            st.markdown(
                f'<span class="tool-pill">{c.tool_name}</span><span class="{css}">{status}</span>',
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            left.code(json.dumps(c.args, indent=2), language="json")
            right.code(json.dumps(c.result, indent=2, default=str), language="json")


def _render_transcript(repo: Repository, conversation_id: int) -> None:
    calls_by_turn: dict[int, list[ToolCallRecord]] = defaultdict(list)
    for c in repo.get_tool_calls(conversation_id):
        calls_by_turn[c.turn].append(c)
    transcript = repo.get_transcript(conversation_id)
    last_turn = transcript[-1].turn if transcript else 0
    for message in transcript:
        with st.chat_message(message.role):
            st.write(message.content)
            if message.role == "assistant" and message.turn > 0:
                _render_tool_calls(
                    calls_by_turn.get(message.turn, []), expanded=message.turn == last_turn
                )


# ---- sidebar ------------------------------------------------------------------------------------

tracing_on = _tracing()
has_key = bool(Settings().anthropic_api_key)

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=150)
    st.markdown("### Store Assistant")
    st.caption("Applied AI take-home · PydanticAI + Claude")
    demo = st.toggle("Demo mode (no API calls)", value=not has_key, disabled=not has_key)
    if not has_key:
        st.caption("No ANTHROPIC_API_KEY found, so demo mode is forced.")
    if st.button("New conversation", width="stretch", type="primary"):
        _reset(demo)

if "runner" not in st.session_state:
    _reset(demo)

maybe_runner: ConversationRunner | None = st.session_state.runner
if maybe_runner is None or maybe_runner.state is None:
    st.error(st.session_state.get("error") or "Could not start a conversation.")
    st.stop()
    raise SystemExit  # unreachable: st.stop() ends the script run
runner: ConversationRunner = maybe_runner
assert runner.state is not None

settings = runner.settings
state = runner.state
repo = runner.repo

with st.sidebar:
    st.divider()
    st.markdown("#### Session")
    st.caption(f"Model: `{runner.model_name}`")
    cost = float(runner.session_cost())
    cap = float(settings.session_cost_limit_usd)
    m1, m2 = st.columns(2)
    m1.metric("Turn", f"{state.turn} / {settings.max_turns_per_session}")
    m2.metric("Est. cost", f"${cost:.4f}")
    st.progress(min(cost / cap, 1.0) if cap else 0.0, text=f"Session cost cap ${cap:.2f}")
    st.progress(
        min(state.turn / settings.max_turns_per_session, 1.0),
        text=f"Turn cap {settings.max_turns_per_session}",
    )
    tokens_in = runner.usage.input_tokens + runner.usage.cache_read_tokens
    st.caption(
        f"Tokens in {tokens_in:,} (cached {runner.usage.cache_read_tokens:,}) · "
        f"out {runner.usage.output_tokens:,} · requests {runner.usage.requests}"
    )
    if state.passphrase_locked:
        passphrase = "🔒 locked"
    elif state.passphrase_verified:
        passphrase = "✅ verified"
    else:
        passphrase = "⏳ not yet verified"
    st.markdown(f"**Passphrase:** {passphrase}")
    st.markdown(
        f"**Off-topic strikes:** {state.off_topic_strikes} / {settings.off_topic_strike_limit}"
    )
    st.caption(
        f"Demo passphrase: `{settings.store_passphrase}` "
        "(set STORE_PASSPHRASE in .env to change). The assistant itself never reveals it."
    )
    st.caption("Logfire tracing: " + ("on" if tracing_on else "off (set LOGFIRE_TOKEN)"))

    st.divider()
    st.markdown("#### Saved stores")
    stores = repo.list_stores()
    if stores:
        unlocked = state.passphrase_verified
        st.dataframe(
            [
                {"Store": s.name, "Phone": s.phone_e164 if unlocked else mask_phone(s.phone_e164)}
                for s in stores
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Phones unmasked for this session."
            if unlocked
            else "Phones are masked until the passphrase is verified in the chat."
        )
    else:
        st.caption("None yet. Try: *Save Trader Joe's, (415) 555-2671*")

# ---- main ---------------------------------------------------------------------------------------

st.title("Store Assistant")
st.caption(
    "Save a store's phone number, or look one up with the passphrase. "
    "Say *I'm done* to finish and get a saved summary."
)

chat_tab, history_tab, about_tab = st.tabs(["💬 Chat", "🕘 History", "📘 How it works"])

# Placed at top level so Streamlit pins the input to the bottom of the page.
user_text = None if runner.ended else st.chat_input("Save a store, look one up, or say you're done")

with chat_tab:
    # The database transcript is the source of truth: it survives reruns and interruptions.
    _render_transcript(repo, state.conversation_id)
    if user_text:
        with st.chat_message("user"):
            st.write(user_text)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            runner.send(user_text)
        st.rerun()

    st.caption(
        f"turn {state.turn} · tokens in {tokens_in:,} / out {runner.usage.output_tokens:,}"
        f" · est. ${cost:.4f}"
    )
    if runner.ended:
        conversation = repo.get_conversation(state.conversation_id)
        summary = conversation.summary if conversation else runner.summary
        st.success(f"**Session ended ({state.end_reason}). Summary saved:**\n\n{summary}")
        st.info("Start a new conversation from the sidebar.")

with history_tab:
    conversations = repo.list_conversations(50)
    if not conversations:
        st.caption("No conversations yet.")
    else:
        st.dataframe(
            [
                {
                    "ID": c.id,
                    "Started": c.started_at.replace("T", " ")[:19],
                    "Reason": c.end_reason or "in progress",
                    "Cost": f"${c.estimated_cost_usd:.4f}",
                    "Summary": c.summary or "",
                }
                for c in conversations
            ],
            hide_index=True,
            width="stretch",
        )
        chosen = st.selectbox(
            "Open a conversation",
            options=[c.id for c in conversations],
            format_func=lambda i: f"#{i}",
        )
        if chosen:
            selected = repo.get_conversation(chosen)
            if selected and selected.summary:
                st.success(f"**Summary ({selected.end_reason}):** {selected.summary}")
            _render_transcript(repo, chosen)

with about_tab:
    st.markdown(
        f"""
**What it does.** Saves stores (name + phone) and retrieves a phone by name, behind a passphrase.
Ends on *I'm done* or after {settings.off_topic_strike_limit} off-topic turns, then writes a
summary to the database.

**Where the rules live.** The model (Claude via PydanticAI) chooses tools and phrasing. Plain
Python enforces everything that must never bend:

- Phone validation and E.164 normalization with libphonenumber. Bad numbers are reprompted.
- The passphrase is captured before the message reaches the model and replaced with a
  placeholder, so the model, the message history, and the traces never see it. It is compared in
  code; {settings.passphrase_max_attempts} wrong attempts lock lookups for the session.
- Duplicate store names require an overwrite confirmation that is tracked in session state: the
  model cannot skip it by setting a flag.
- Off-topic turns are counted by the model's judgement, but the counting, the reset on a
  successful on-topic action, and the termination are in code.
- Spend guardrails: {settings.max_turns_per_session} turns, {settings.max_requests_per_session}
  requests, \\${settings.session_cost_limit_usd} per session, \\${settings.daily_cost_limit_usd}
  per day, prompt caching on.

**Inspecting the past.** Every transcript, tool call, usage row, and summary is in SQLite
(see the History tab or `store-assistant show <id>`). With a Logfire token, every model
request is also traced with full spans.

**Tests.** The pytest suite runs with no API key using PydanticAI's `FunctionModel`, plus an
opt-in live round trip.
"""
    )
