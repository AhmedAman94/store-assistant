"""SQLite persistence.

A single `Repository` class wraps all SQL so the rest of the app never touches the database
directly. Swapping to Postgres later means replacing this one module.

Tables:
- stores: one row per store, keyed by a normalized name for case-insensitive lookups.
- conversations: one row per session with its end reason, summary, and total usage.
- messages: the full transcript, so past conversations can be inspected locally.
- usage_log: one row per model request, used for the daily spend ceiling.
- tool_calls: every tool the model invoked, with arguments and result, per turn.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from store_assistant.validation import normalize_store_name

SCHEMA = """
CREATE TABLE IF NOT EXISTS stores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL UNIQUE,
    phone_e164 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    end_reason TEXT,
    summary TEXT,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd TEXT NOT NULL DEFAULT '0'
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    estimated_cost_usd TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_log(created_at);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    turn INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation ON tool_calls(conversation_id, id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class StoreRecord:
    id: int
    name: str
    phone_e164: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    id: int
    started_at: str
    ended_at: str | None
    end_reason: str | None
    summary: str | None
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    turn: int
    tool_name: str
    args: dict[str, Any]
    result: Any
    created_at: str


@dataclass(frozen=True, slots=True)
class MessageRecord:
    turn: int
    role: str
    content: str
    created_at: str


class Repository:
    """All database access for the Store Assistant."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        # PydanticAI runs synchronous tools in a worker thread, so the connection must be
        # shareable across threads; the lock serializes access.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def _fetchone(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            row: sqlite3.Row | None = self._conn.execute(sql, params).fetchone()
            return row

    def _fetchall(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    # ---- stores -------------------------------------------------------------------------------

    def get_store(self, name: str) -> StoreRecord | None:
        row = self._fetchone(
            "SELECT * FROM stores WHERE name_normalized = ?", (normalize_store_name(name),)
        )
        return _store_from_row(row) if row else None

    def save_store(self, name: str, phone_e164: str) -> tuple[StoreRecord, bool]:
        """Insert or update a store. Returns the record and whether it was newly created."""
        clean_name = " ".join(name.split())
        key = normalize_store_name(clean_name)
        now = _now()
        # One critical section for read-then-write so concurrent saves of the same name
        # (tools run sequentially, but the CLI, web UI, and tests share this class) cannot race.
        with self._lock:
            existing = self.get_store(clean_name)
            if existing is None:
                self._conn.execute(
                    "INSERT INTO stores (name, name_normalized, phone_e164, created_at, "
                    "updated_at) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(name_normalized) DO UPDATE SET name = excluded.name, "
                    "phone_e164 = excluded.phone_e164, updated_at = excluded.updated_at",
                    (clean_name, key, phone_e164, now, now),
                )
                created = True
            else:
                self._conn.execute(
                    "UPDATE stores SET name = ?, phone_e164 = ?, updated_at = ? WHERE id = ?",
                    (clean_name, phone_e164, now, existing.id),
                )
                created = False
            record = self.get_store(clean_name)
        assert record is not None
        return record, created

    def list_stores(self) -> list[StoreRecord]:
        rows = self._fetchall("SELECT * FROM stores ORDER BY name_normalized")
        return [_store_from_row(r) for r in rows]

    # ---- conversations --------------------------------------------------------------------------

    def start_conversation(self, model: str) -> int:
        cur = self._execute(
            "INSERT INTO conversations (started_at, model) VALUES (?, ?)", (_now(), model)
        )
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def end_conversation(
        self,
        conversation_id: int,
        *,
        end_reason: str,
        summary: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: Decimal,
    ) -> None:
        self._execute(
            "UPDATE conversations SET ended_at = ?, end_reason = ?, summary = ?, "
            "input_tokens = ?, output_tokens = ?, estimated_cost_usd = ? WHERE id = ?",
            (
                _now(),
                end_reason,
                summary,
                input_tokens,
                output_tokens,
                str(estimated_cost_usd),
                conversation_id,
            ),
        )

    def get_conversation(self, conversation_id: int) -> ConversationRecord | None:
        row = self._fetchone("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        return _conversation_from_row(row) if row else None

    def list_conversations(self, limit: int = 50) -> list[ConversationRecord]:
        rows = self._fetchall("SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,))
        return [_conversation_from_row(r) for r in rows]

    # ---- messages -------------------------------------------------------------------------------

    def add_message(self, conversation_id: int, turn: int, role: str, content: str) -> None:
        self._execute(
            "INSERT INTO messages (conversation_id, turn, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, turn, role, content, _now()),
        )

    def get_transcript(self, conversation_id: int) -> list[MessageRecord]:
        rows = self._fetchall(
            "SELECT turn, role, content, created_at FROM messages WHERE conversation_id = ? "
            "ORDER BY id",
            (conversation_id,),
        )
        return [
            MessageRecord(
                turn=r["turn"], role=r["role"], content=r["content"], created_at=r["created_at"]
            )
            for r in rows
        ]

    # ---- tool calls ----

    def add_tool_call(
        self, conversation_id: int, turn: int, tool_name: str, args: dict[str, Any], result: Any
    ) -> None:
        self._execute(
            "INSERT INTO tool_calls (conversation_id, turn, tool_name, args_json, result_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                turn,
                tool_name,
                json.dumps(args, default=str),
                json.dumps(result, default=str),
                _now(),
            ),
        )

    def get_tool_calls(self, conversation_id: int) -> list[ToolCallRecord]:
        rows = self._fetchall(
            "SELECT turn, tool_name, args_json, result_json, created_at FROM tool_calls "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        return [
            ToolCallRecord(
                turn=r["turn"],
                tool_name=r["tool_name"],
                args=json.loads(r["args_json"]),
                result=json.loads(r["result_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ---- usage ----

    def log_usage(
        self,
        conversation_id: int,
        *,
        model: str,
        input_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        output_tokens: int,
        estimated_cost_usd: Decimal,
    ) -> None:
        self._execute(
            "INSERT INTO usage_log (conversation_id, model, input_tokens, cache_read_tokens, "
            "cache_write_tokens, output_tokens, estimated_cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                model,
                input_tokens,
                cache_read_tokens,
                cache_write_tokens,
                output_tokens,
                str(estimated_cost_usd),
                _now(),
            ),
        )

    def cost_since(self, since_iso: str) -> Decimal:
        rows = self._fetchall(
            "SELECT estimated_cost_usd FROM usage_log WHERE created_at >= ?", (since_iso,)
        )
        return sum((Decimal(r["estimated_cost_usd"]) for r in rows), Decimal("0"))

    def cost_today(self) -> Decimal:
        start_of_day = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.cost_since(start_of_day.isoformat(timespec="seconds"))


def _store_from_row(row: sqlite3.Row) -> StoreRecord:
    return StoreRecord(
        id=row["id"],
        name=row["name"],
        phone_e164=row["phone_e164"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _conversation_from_row(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        id=row["id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        end_reason=row["end_reason"],
        summary=row["summary"],
        model=row["model"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        estimated_cost_usd=Decimal(row["estimated_cost_usd"]),
    )
