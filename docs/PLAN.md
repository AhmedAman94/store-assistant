# Store Assistant: End-to-End Build Plan

This plan records every decision made
during planning and the order the build followed.

## 1. Goal

A conversational agent that saves and retrieves store records, with:

- Phone validation and reprompting on bad input.
- A passphrase gate in front of lookups, enforced in code.
- Any number of save and retrieve operations per session, in any order.
- Session termination on "I'm done" style utterances or repeated off-topic turns, followed by
  an LLM-written summary saved to the database.
- Tracing of every LLM call so past conversations can be inspected.
- At least two tests (we will ship many more).
- A CLI demo, with a Streamlit web UI as a stretch goal.

## 2. Locked decisions

| Area | Decision | Rationale |
|---|---|---|
| Language | Python 3.12, `uv` for env and deps | Spec requires Python; `uv` is fast and reproducible |
| Agent framework | PydanticAI | Typed tools, dependency injection for session state, fake models for tests, native Logfire tracing |
| LLM | Claude Opus 5.5 (`claude-opus-5-5`) via Anthropic | High-quality tool use. One config value; Haiku 5.5 documented as the cheap switch |
| Summary model | Configurable; recommended Haiku 5.5 | Summaries are easy; 40x cheaper than Opus. Defaults to the main model unless set |
| Database | SQLite, stdlib `sqlite3`, repository class | Real SQL, zero setup, single inspectable file |
| Tracing | Pydantic Logfire | One-line integration with PydanticAI; full conversation and tool-call inspection |
| Phone validation | `phonenumbers`, US default region, international with country code | Normalize to E.164 so formats collapse to one canonical value |
| Passphrase | From env var `STORE_PASSPHRASE`, default `open sesame`; captured by the runner before the model sees it | Compared in code; three wrong attempts lock lookups for the session |
| Duplicate store name | Ask the user to confirm overwrite | Safer than silent overwrite |
| Off-topic policy | Three off-topic strikes end the session | "Repeatedly" interpreted as a strike counter; only a successful save, verify, or found lookup resets it |
| UI | Rich CLI first; Streamlit if time allows | Fast to demo and script |
| Tests | pytest with PydanticAI `TestModel` / `FunctionModel`; one optional live test | Deterministic, no API spend |
| Key handling | `.env` only, gitignored; `.env.example` committed; reviewers use their own key, or `--demo` mode with no key | Decided: options 1 and 2. A key is never shipped or embedded; it cannot be hidden from someone running the code locally |
| Spend guardrails | Built into the app regardless of whose key is used | See section 5 |

## 3. Architecture

The language model decides which tool to call and how to phrase replies. Plain Python
enforces every rule that must never be bent. The model never holds state.

```
CLI / Streamlit
      |
ConversationRunner        owns the loop, turn limits, termination, summary save
      |
PydanticAI Agent          system prompt + tools, model from config
      |          \
   tools        SessionState (deps)   passphrase_verified, strikes, attempts, ops log, ended
      |
Repository (SQLite)       stores, conversations, messages, usage
      |
Logfire tracing           every model request, tool call, token usage
```

### Tools exposed to the model

| Tool | Behavior enforced in code |
|---|---|
| `save_store(name, phone, confirm_overwrite=False)` | Validates phone; returns structured error on failure. If name exists and not confirmed, returns a "needs confirmation" result so the model asks the user |
| `verify_passphrase(passphrase)` | Compares against config in code. Sets `passphrase_verified`. Tracks attempts; after 3 failures lookups are locked for the rest of the session |
| `get_store(name)` | Refuses unless `passphrase_verified` is true. Case-insensitive lookup. Returns phone or not-found |
| `flag_off_topic()` | Increments strike counter and returns strikes remaining. Code ends the session at 3 |
| `end_session()` | Marks session ended with reason `user_done` |

Design notes:

- The passphrase is verified once per session by default. A config flag can require it per lookup.
- The runner swaps the secret for a placeholder before the model sees the message, so the model
  never relays it; the tool is told out-of-band that it was supplied.
- Tool results are small structured objects (Pydantic models), not free text, so the model's
  behavior is driven by explicit status fields.
- The system prompt describes scope (store save and retrieve only), the reprompt behavior, and
  when to call each tool. It does not contain the passphrase.

### Data model (SQLite)

- `stores`: `id`, `name` (unique, case-insensitive via `name_normalized`), `phone_e164`,
  `created_at`, `updated_at`.
- `conversations`: `id`, `started_at`, `ended_at`, `end_reason`, `summary`, `model`,
  `input_tokens`, `output_tokens`, `estimated_cost_usd`.
- `messages`: `conversation_id`, `turn`, `role`, `content`, `created_at`. Local transcript so
  past conversations are inspectable without the Logfire dashboard.
- `usage_log`: per-request usage rows for the daily budget check.
- `tool_calls`: every tool invocation per turn with redacted arguments and result.

### Termination and summary flow

1. Runner detects `session.ended` after any turn, or the turn cap is reached.
2. Runner calls the summarizer agent with the transcript. Output: 2 to 4 sentences covering
   operations performed, outcomes, and why the session ended.
3. Runner writes the summary and end reason to `conversations`, prints it to the user, exits.
4. If the summary call itself fails, a templated fallback summary built from the ops log is
   saved so the database row is never left empty.

## 4. Spend guardrails

All guardrails are active no matter whose key is in `.env`. Defaults live in `config.py` and
can be overridden by env vars.

| Guardrail | Default | Mechanism |
|---|---|---|
| Max user turns per session | 40 | Runner ends session gracefully with summary |
| Max model requests per session | 80 | PydanticAI `UsageLimits(request_limit=...)` covers tool-loop iterations |
| Max total tokens per session | 500,000 | PydanticAI `UsageLimits(total_tokens_limit=...)` |
| Max output tokens per response | 1,024 | Model settings |
| Per-session cost ceiling | $0.50 | Computed from usage and a per-model price table; session ends with summary when hit |
| Daily cost ceiling across sessions | $5.00 | Sum of `usage_log` for today; new sessions refused with a clear message |
| Prompt caching | On | Cache the system prompt and tool definitions so repeated turns pay cache-read rates |
| Cost readout | On | CLI footer shows tokens and estimated cost after every turn |
| Demo mode | `--demo` flag | Scripted fake model; zero API calls; full flow visible |

Prices come from the `genai-prices` data bundled with PydanticAI and are documented as estimates.

## 5. Project layout

```
ndr_assessment/
  pyproject.toml               deps, ruff, mypy, pytest config
  uv.lock
  README.md                    setup, run, test, architecture summary, design notes
  .env.example                 ANTHROPIC_API_KEY, STORE_PASSPHRASE, model, budgets, LOGFIRE_TOKEN
  .gitignore                   .env, *.db, .venv, __pycache__
  .pre-commit-config.yaml      ruff format + lint
  docs/
    PLAN.md                    this file
    ARCHITECTURE.md            diagrams and design rationale
  src/store_assistant/
    __init__.py
    config.py                  pydantic-settings Settings
    validation.py              phone normalization
    db.py                      Repository over sqlite3, schema migration on startup
    session.py                 SessionState dataclass
    tools.py                   tool result models + tool implementations
    redaction.py               secret capture before the model, redaction of persisted text
    agent.py                   PydanticAI Agent, system prompt, tool registration
    guardrails.py              UsageLimits, price table, cost accounting, daily budget
    summarizer.py              summary agent + templated fallback
    runner.py                  ConversationRunner
    tracing.py                 Logfire configuration (no-op when token absent)
    demo_model.py              scripted FunctionModel for --demo
    cli.py                     Typer app, Rich chat loop
    web.py                     Streamlit UI (stretch)
  tests/
    conftest.py                temp SQLite, isolated settings, scripted FunctionModel helpers
    test_validation.py
    test_db.py
    test_tools.py              every rule enforced in code
    test_runner.py             termination, redaction, guardrails, error handling
    test_cli.py                Typer runner with hostile input
    test_demo.py               full scripted conversation through the demo model
    test_agent.py
    test_live.py               skipped unless RUN_LIVE_TESTS is set and a key is configured
```

## 6. Tests

| Test | What it proves |
|---|---|
| Phone validation | Accepts `(555) 123-4567`, `555.123.4567`, `+44 20 7946 0958`; rejects `12345`, letters, empty |
| Save store | Valid input writes a row; invalid phone returns a structured error and no row |
| Duplicate name | Second save without confirmation returns "needs confirmation"; with confirmation updates phone |
| Passphrase gate | `get_store` refuses before verification; correct passphrase unlocks; three failures lock |
| Retrieve | Case-insensitive lookup returns E.164 phone; unknown name returns not-found |
| Termination: done | "I'm done" path sets `end_reason=user_done` and saves a summary row |
| Termination: off-topic | Three off-topic strikes end the session with `end_reason=off_scope` |
| Guardrails | Turn cap and cost ceiling end the session; daily budget refuses a new session |
| Live smoke | Real model saves and retrieves one store end to end; skipped by default |

Tests use `TestModel` for tool-plumbing checks and `FunctionModel` where a scripted sequence of
tool calls is needed. No test spends API credit unless explicitly opted in.

## 7. Build order and milestones

Status (2026-10-12): all milestones complete, live-verified against Claude Opus 5.5 with Logfire
tracing confirmed, then hardened after an independent three-reviewer audit (code-enforced
overwrite confirmation, out-of-band passphrase capture, full redaction, error handling, masked
phone displays, environment-isolated tests).

1. **Scaffold.** `uv init`, pyproject with deps, ruff and mypy config, `.env.example`,
   `.gitignore`, `git init`. Exit criterion: `uv run pytest` runs an empty suite.
2. **Data layer.** `validation.py`, `db.py`, tests for both. No LLM involved.
3. **Agent core.** `session.py`, `tools.py`, `agent.py`, `guardrails.py`. Tests with
   `TestModel` and `FunctionModel` for every tool rule.
4. **Runner and termination.** `runner.py`, `summarizer.py`, `tracing.py`. Termination tests.
5. **CLI.** `cli.py` with Rich, cost footer, `--demo` mode via `demo_model.py`.
6. **Live verification.** Run a real conversation with your key, confirm traces in Logfire,
   confirm rows in SQLite, run the live smoke test once.
7. **Docs.** README, ARCHITECTURE.md, design notes.
8. **Stretch.** Streamlit `web.py` reusing `ConversationRunner`.

## 8. Dependencies

Runtime: `pydantic-ai-slim[anthropic,logfire]`, `pydantic-settings`, `phonenumbers`, `typer`,
`rich`, `genai-prices`. Dev: `pytest`, `ruff`, `mypy`, `pre-commit`.
Stretch: `streamlit`. Exact versions pinned in `uv.lock`.

## 9. Design notes for discussion

- Why rules live in code, not the prompt: passphrase, validation, strikes, budgets.
- How the model is swapped with one config value, and what the cheap path costs.
- What the traces show and how to find a past conversation.
- How the tests avoid API spend while still exercising real tool logic.
- Known limits: single-user, no auth beyond the passphrase, SQLite not for concurrency,
  price table is an estimate.
