# Architecture

## The one design decision that matters

The language model is responsible for *understanding and phrasing*. Python is responsible for
*rules*. The model never holds state and cannot bend a rule, because the rule is checked inside a
tool it has to call.

## System overview

```mermaid
flowchart TB
    subgraph UI["User interfaces"]
        CLI["Rich CLI<br/>cli.py"]
        WEB["Streamlit chat<br/>web.py"]
    end

    subgraph CORE["Conversation engine"]
        RUN["ConversationRunner<br/>runner.py<br/><i>turn loop · termination · summary</i>"]
        ASY["Background event loop<br/>_async.py"]
        GUARD["Guardrails<br/>guardrails.py<br/><i>UsageLimits · cost · daily cap</i>"]
        AGENT["PydanticAI Agent<br/>agent.py<br/><i>instructions + 5 tools</i>"]
        SUMM["Summarizer agent<br/>summarizer.py"]
        STATE[("SessionState<br/>session.py<br/><i>verified · attempts · strikes · ended</i>")]
    end

    subgraph TOOLS["Tools (rules enforced here) · tools.py"]
        T1["save_store"]
        T2["verify_passphrase"]
        T3["get_store"]
        T4["flag_off_topic"]
        T5["end_session"]
        VAL["validation.py<br/><i>libphonenumber → E.164</i>"]
    end

    subgraph DATA["Persistence · db.py"]
        DB[("SQLite<br/>stores · conversations<br/>messages · usage_log")]
    end

    subgraph EXT["External"]
        LLM["Claude Opus 5.5<br/>Anthropic API<br/><i>prompt caching on</i>"]
        LF["Logfire<br/>tracing.py"]
    end

    CFG["Settings<br/>config.py · .env<br/><i>model · passphrase · limits</i>"]

    CLI --> RUN
    WEB --> RUN
    RUN --> ASY --> AGENT
    RUN --> GUARD
    RUN --> SUMM
    AGENT <-->|"requests / tool calls"| LLM
    SUMM <--> LLM
    AGENT --> T1 & T2 & T3 & T4 & T5
    T1 --> VAL
    T1 & T2 & T3 & T4 & T5 --> STATE
    T1 & T3 --> DB
    RUN --> DB
    SUMM --> DB
    AGENT -. spans .-> LF
    SUMM -. spans .-> LF
    CFG -.-> RUN & AGENT & GUARD & T2
```

## One turn, step by step

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as CLI / Streamlit
    participant R as ConversationRunner
    participant A as PydanticAI Agent
    participant M as Claude
    participant T as Tools
    participant S as SessionState
    participant DB as SQLite

    U->>UI: "What's Trader Joe's number?"
    UI->>R: send(text)
    R->>DB: add_message(user, redacted)
    R->>A: run(text, deps, shared usage, limits, history)
    A->>M: messages + tool schemas
    M-->>A: tool_call get_store("Trader Joe's")
    A->>T: get_store
    T->>S: passphrase_verified?
    S-->>T: false
    T-->>A: {status: passphrase_required}
    A->>M: tool result
    M-->>A: "Please enter the passphrase."
    A-->>R: output + usage
    R->>DB: log_usage(delta) · add_message(assistant)
    R-->>UI: TurnResult(reply, ended=false, cost)
    UI-->>U: "Please enter the passphrase."

    U->>UI: "open sesame"
    UI->>R: send(text)
    R->>DB: add_message(user, "[passphrase redacted]")
    R->>A: run(...)
    A->>M: messages
    M-->>A: tool_call verify_passphrase("open sesame")
    A->>T: verify_passphrase
    T->>S: compare_digest OK → verified = true
    T-->>A: {status: verified}
    A->>M: tool result
    M-->>A: tool_call get_store("Trader Joe's")
    A->>T: get_store
    T->>DB: SELECT by name_normalized
    DB-->>T: +14155552671
    T-->>A: {status: found, phone}
    A->>M: tool result
    M-->>A: "Trader Joe's number is +1 415-555-2671."
    A-->>R: output + usage
    R-->>UI: TurnResult(reply)
    UI-->>U: reply
```

## Session lifecycle

```mermaid
stateDiagram-v2
    [*] --> Idle: start() · daily budget OK
    state Idle {
        [*] --> Unverified
        Unverified --> Verified: verify_passphrase correct
        Unverified --> Unverified: incorrect (attempts < 3)
        Unverified --> Locked: 3rd incorrect
        Verified --> Unverified: lookup done<br/>(only if REQUIRE_PASSPHRASE_PER_LOOKUP)
    }
    Idle --> Idle: save_store / get_store<br/>(resets off-topic strikes)
    Idle --> Idle: flag_off_topic<br/>(strikes 1–2)
    Idle --> Ended: end_session → user_done
    Idle --> Ended: flag_off_topic, 3rd strike → off_scope
    Idle --> Ended: turn ≥ MAX_TURNS → turn_limit
    Idle --> Ended: UsageLimitExceeded → budget_limit
    Idle --> Ended: Ctrl-C / EOF → aborted
    Ended --> Summarized: summarizer (or templated fallback)
    Summarized --> [*]: conversations row updated
```

## Data model

```mermaid
erDiagram
    STORES {
        int id PK
        text name
        text name_normalized UK "casefolded, whitespace-collapsed"
        text phone_e164
        text created_at
        text updated_at
    }
    CONVERSATIONS {
        int id PK
        text started_at
        text ended_at
        text end_reason "user_done | off_scope | turn_limit | budget_limit | aborted | error"
        text summary
        text model
        int input_tokens
        int output_tokens
        text estimated_cost_usd
    }
    MESSAGES {
        int id PK
        int conversation_id FK
        int turn
        text role "user | assistant"
        text content "passphrase redacted"
        text created_at
    }
    USAGE_LOG {
        int id PK
        int conversation_id FK
        text model
        int input_tokens
        int cache_read_tokens
        int cache_write_tokens
        int output_tokens
        text estimated_cost_usd
        text created_at
    }
    CONVERSATIONS ||--o{ MESSAGES : "has"
    CONVERSATIONS ||--o{ USAGE_LOG : "has"
```

## Spend guardrails

```mermaid
flowchart LR
    A["start()"] --> B{"cost_today ≥<br/>DAILY_COST_LIMIT?"}
    B -- yes --> X["refuse session"]
    B -- no --> C["send(text)"]
    C --> D{"turn ≤ MAX_TURNS?"}
    D -- no --> E["end: turn_limit"]
    D -- yes --> F["agent.run with shared RunUsage"]
    F --> G{"UsageLimits<br/>requests · tokens · cost"}
    G -- exceeded --> H["end: budget_limit"]
    G -- ok --> I["log usage delta → usage_log"]
    I --> C
    E & H --> J["summary saved"]
```

## Turn lifecycle

1. The UI calls `ConversationRunner.send(text)`.
2. The runner stores the (passphrase-redacted) user message, then runs the agent with the
   session's shared `RunUsage` and `UsageLimits`, so limits are cumulative across turns.
3. The model may call tools. Each tool reads and mutates `SessionState` and the repository.
4. The runner logs the usage delta for the turn, applies the turn cap, and stores the reply.
5. If the state is now `ended`, the runner calls the summarizer, saves the summary and end reason
   on the conversation row, and returns them in `TurnResult`.

## Why each rule lives where it does

| Rule | Where | Why not in the prompt |
|---|---|---|
| Phone validation | `validation.py`, called by `save_store` | Models hallucinate "valid"; libphonenumber doesn't |
| Passphrase | `verify_passphrase` compares against config with `hmac.compare_digest` | The model never sees the secret, so it can't leak it or be talked into skipping it |
| Lookup gate | `get_store` returns `passphrase_required` unless state says verified | The gate is a boolean the model can't set |
| Three-strike lock | `verify_passphrase` counts attempts | Deterministic and testable |
| Overwrite confirmation | `save_store` returns `needs_confirmation` unless `confirm_overwrite=true` | Prevents silent data loss |
| Off-topic termination | `flag_off_topic` counts consecutive strikes; on-topic tools reset | "Repeatedly" needs a counter |
| Turn / request / token / cost caps | `guardrails.py` + runner | Protects whoever's API key is in use |
| Daily cap | `usage_log` table summed on `start()` | Survives process restarts |

## Data model

- `stores(id, name, name_normalized UNIQUE, phone_e164, created_at, updated_at)`
- `conversations(id, started_at, ended_at, end_reason, summary, model, input_tokens, output_tokens, estimated_cost_usd)`
- `messages(id, conversation_id, turn, role, content, created_at)`
- `usage_log(id, conversation_id, model, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens, estimated_cost_usd, created_at)`

## Testing strategy

- **Tool tests** call the tool functions directly with a fake context. They prove the rules.
- **Runner tests** use PydanticAI's `FunctionModel` to script exact tool-call sequences. They
  prove termination, summaries, and guardrails without an API.
- **Demo test** drives a full 13-turn conversation through the rule-based demo model, which
  exercises the real runner, tools, DB, redaction, and summary end to end.
- **Live test** (opt-in) does one real save/lookup/done round trip against Claude.

## Swapping pieces

- Model: change `LLM_MODEL`. PydanticAI supports Anthropic, OpenAI, Gemini, Groq, Ollama, etc.
- Database: replace `db.py`; nothing else imports `sqlite3`.
- UI: anything that can call `start()` and `send()` works; see `web.py`.

## Production path

What this take-home deliberately leaves out, and how each piece would be added without
changing the core design. None of it is needed for a single-user local demo, and building it
now would add setup friction for no benefit.

| Concern | Today | Production step |
|---|---|---|
| Users and auth | Single user, no login | Put the app behind an identity provider (OIDC). Add a `users` table and a `user_id` column on `stores` and `conversations`, scoped in the repository. Keep the passphrase as a per-lookup second factor |
| Database | SQLite file via `Repository` | Swap the repository for Postgres behind the same interface, add migrations (Alembic). Nothing outside `db.py` changes |
| Concurrency | One `ConversationRunner` per session, sends serialized with a lock | Move `SessionState` and message history to Redis so any instance can serve any session. Keep the lock as a per-session Redis lock |
| Async work | Summaries run inline at session end | Only if volume demands it: a queue (SQS, Redis Streams, or Kafka) for summarization, audit export, and analytics. The summarizer is already a separate agent, so it moves without changes |
| Model and cost | One model per config, per-session and daily caps | Per-user budgets, a routing table (cheaper model for summaries by default), and cost alerts from Logfire |
| Observability | Logfire traces plus local transcripts | Alerting on error rate, latency, and spend; sampling for high volume; retention policy for traces that contain message content |
| Secrets | `.env` file | A secrets manager (AWS Secrets Manager, Vault) and key rotation. The passphrase becomes a hashed per-user secret, not a shared string |
| Abuse and safety | Three-strike off-topic policy, attempt lock | Rate limiting per user and IP, input length caps, prompt-injection evals in CI, and a review of what the model may say about the passphrase |
| Privacy | Passphrase redacted from transcripts and traces | PII classification of transcripts, data-retention windows, user deletion flows |
| Delivery | `uv run` locally | Container image, health endpoint, CI running the test suite and the live smoke test on a schedule |

## Deliberate limits

- Lookups lock for the whole session after three bad passphrases (simple, safe, documented).
- Passphrase comparison is case-insensitive and whitespace-trimmed for friendliness.
- Cost estimates rely on `genai-prices` data bundled with PydanticAI.
