"""In-memory state for one conversation.

This object is passed to every tool through PydanticAI dependency injection. The language model
never sees or controls it, which is what makes the passphrase gate, strike counter, and
termination rules enforceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EndReason = Literal["user_done", "off_scope", "turn_limit", "budget_limit", "aborted", "error"]


@dataclass(slots=True)
class SessionState:
    conversation_id: int
    turn: int = 0
    passphrase_verified: bool = False
    passphrase_attempts: int = 0
    passphrase_locked: bool = False
    off_topic_strikes: int = 0
    ended: bool = False
    end_reason: EndReason | None = None
    operations: list[str] = field(default_factory=list)

    def record(self, event: str) -> None:
        """Append a short human-readable note used by the summary fallback."""
        self.operations.append(event)

    def end(self, reason: EndReason) -> None:
        if not self.ended:
            self.ended = True
            self.end_reason = reason

    def reset_passphrase(self) -> None:
        self.passphrase_verified = False
        self.passphrase_attempts = 0
        self.passphrase_locked = False
