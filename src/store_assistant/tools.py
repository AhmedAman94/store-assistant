"""Tools the agent can call, and the structured results they return.

Every rule that must hold regardless of what the language model does is enforced here:
phone validation, the passphrase gate, attempt locking, overwrite confirmation, the off-topic
strike counter, and session termination. The model only sees small status objects and phrases
replies around them. Once the session has ended, every tool refuses to act.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from store_assistant.config import Settings
from store_assistant.db import Repository
from store_assistant.redaction import contains_secret, redact
from store_assistant.session import SessionState
from store_assistant.validation import normalize_phone, normalize_store_name


@dataclass(slots=True)
class Deps:
    """Dependencies injected into every tool call."""

    settings: Settings
    repo: Repository
    state: SessionState
    passphrase_supplied: bool = False
    """Set by the runner when the user's message contained the passphrase this turn.

    The secret itself is replaced with a placeholder before the model sees the message, so the
    model never needs to relay it; the tool trusts this flag instead.
    """


# ---- result models ------------------------------------------------------------------------------

ENDED: Final = "session_ended"


class SaveStoreResult(BaseModel):
    status: Literal[
        "saved", "updated", "invalid_name", "invalid_phone", "needs_confirmation", "session_ended"
    ]
    store_name: str
    phone: str | None = Field(default=None, description="Stored phone in E.164 form")
    existing_phone: str | None = Field(
        default=None, description="Phone currently on file when confirmation is needed"
    )
    message: str


class VerifyPassphraseResult(BaseModel):
    status: Literal["verified", "already_verified", "incorrect", "locked", "session_ended"]
    attempts_remaining: int
    message: str


class GetStoreResult(BaseModel):
    status: Literal["found", "not_found", "passphrase_required", "locked", "session_ended"]
    store_name: str
    phone: str | None = None
    message: str


class OffTopicResult(BaseModel):
    status: Literal["redirected", "session_ended"]
    strikes: int
    strikes_remaining: int
    session_ended: bool
    message: str


class EndSessionResult(BaseModel):
    status: Literal["ended", "session_ended"]
    message: str


ENDED_MESSAGE = "The session has already ended. Say goodbye; no further actions are possible."


def _record(deps: Deps, event: str) -> None:
    deps.state.record(redact(event, deps.settings.store_passphrase))


# ---- tools ---------------------------------------------------------------------------------------


def save_store(
    ctx: RunContext[Deps], name: str, phone: str, confirm_overwrite: bool = False
) -> SaveStoreResult:
    """Save a store's name and phone number to the database.

    Call this once you have both a store name and a phone number from the user. The phone is
    validated here; on `invalid_phone` ask the user for the phone again, on `invalid_name` ask for
    the name. If the store already exists you get `needs_confirmation`: ask the user whether to
    overwrite, and only call again with `confirm_overwrite=true` after they say yes. The
    confirmation is tracked in code, so `confirm_overwrite` is ignored unless this exact store
    was just flagged.
    """
    deps = ctx.deps
    state = deps.state
    clean_name = " ".join(name.split())
    if state.ended:
        return SaveStoreResult(status=ENDED, store_name=clean_name, message=ENDED_MESSAGE)

    if not clean_name or len(clean_name) > deps.settings.max_store_name_length:
        reason = "empty" if not clean_name else "too long"
        return SaveStoreResult(
            status="invalid_name",
            store_name=clean_name[:40],
            message=(
                f"Store name is {reason}. Ask the user for a store name of at most "
                f"{deps.settings.max_store_name_length} characters."
            ),
        )
    if contains_secret(clean_name, deps.settings.store_passphrase) or "[passphrase" in clean_name:
        return SaveStoreResult(
            status="invalid_name",
            store_name="",
            message="That name is not allowed as a store name. Ask the user for a different name.",
        )

    validation = normalize_phone(phone, deps.settings.default_phone_region)
    if not validation.ok or validation.e164 is None:
        _record(deps, f"Rejected invalid phone for '{clean_name}'")
        return SaveStoreResult(
            status="invalid_phone",
            store_name=clean_name,
            message=validation.error or "Invalid phone number.",
        )

    key = normalize_store_name(clean_name)
    existing = deps.repo.get_store(clean_name)
    # A confirmation only counts if it was requested for this store on an *earlier* turn, so
    # the user had a chance to answer. The model cannot confirm on its own within one turn.
    confirmed = (
        confirm_overwrite
        and state.pending_overwrite == key
        and state.turn > state.pending_overwrite_turn
    )
    if existing is not None and not confirmed:
        state.pending_overwrite = key
        state.pending_overwrite_turn = state.turn
        _record(deps, f"Asked to confirm overwrite of '{existing.name}'")
        return SaveStoreResult(
            status="needs_confirmation",
            store_name=existing.name,
            existing_phone=existing.phone_e164,
            message=(
                f"'{existing.name}' already exists with phone {existing.phone_e164}. "
                "Ask the user to confirm before overwriting, then call again with "
                "confirm_overwrite=true."
            ),
        )

    record, created = deps.repo.save_store(clean_name, validation.e164)
    state.pending_overwrite = None
    state.off_topic_strikes = 0
    verb = "Saved" if created else "Updated"
    _record(deps, f"{verb} store '{record.name}' with phone {record.phone_e164}")
    return SaveStoreResult(
        status="saved" if created else "updated",
        store_name=record.name,
        phone=record.phone_e164,
        message=f"{verb} '{record.name}' with phone {validation.pretty}.",
    )


def verify_passphrase(ctx: RunContext[Deps], passphrase: str = "") -> VerifyPassphraseResult:
    """Check the passphrase the user typed. Required before any store lookup.

    Pass exactly what the user typed. If their message shows "[passphrase provided]", pass that
    text as-is. You do not know the passphrase and must never guess or try candidates yourself.
    """
    deps = ctx.deps
    state = deps.state
    max_attempts = deps.settings.passphrase_max_attempts
    if state.ended:
        return VerifyPassphraseResult(status=ENDED, attempts_remaining=0, message=ENDED_MESSAGE)

    if state.passphrase_locked:
        return VerifyPassphraseResult(
            status="locked",
            attempts_remaining=0,
            message="Too many incorrect passphrases. Lookups are locked for this session.",
        )
    if state.passphrase_verified:
        return VerifyPassphraseResult(
            status="already_verified",
            attempts_remaining=max_attempts,
            message="Passphrase already verified for this session.",
        )

    expected = deps.settings.store_passphrase.strip().casefold()
    supplied = " ".join((passphrase or "").split()).casefold()
    matched = deps.passphrase_supplied or (
        bool(supplied) and hmac.compare_digest(expected.encode(), supplied.encode())
    )
    if matched:
        state.passphrase_verified = True
        state.passphrase_attempts = 0
        state.off_topic_strikes = 0
        _record(deps, "Passphrase verified")
        return VerifyPassphraseResult(
            status="verified",
            attempts_remaining=max_attempts,
            message="Passphrase accepted. You may now look up stores.",
        )

    if not supplied:
        # An empty attempt is not an attempt; ask again without spending one.
        remaining = max(max_attempts - state.passphrase_attempts, 0)
        return VerifyPassphraseResult(
            status="incorrect",
            attempts_remaining=remaining,
            message="No passphrase was given. Ask the user to type it.",
        )

    state.passphrase_attempts += 1
    remaining = max(max_attempts - state.passphrase_attempts, 0)
    if remaining == 0:
        state.passphrase_locked = True
        _record(deps, "Passphrase locked after repeated failures")
        return VerifyPassphraseResult(
            status="locked",
            attempts_remaining=0,
            message="Incorrect passphrase. Lookups are now locked for this session.",
        )
    _record(deps, "Incorrect passphrase attempt")
    return VerifyPassphraseResult(
        status="incorrect",
        attempts_remaining=remaining,
        message=f"Incorrect passphrase. {remaining} attempt(s) remaining.",
    )


def get_store(ctx: RunContext[Deps], name: str) -> GetStoreResult:
    """Look up a saved store's phone number by name.

    Only works after the passphrase has been verified. If you get `passphrase_required`, ask
    the user for the passphrase and call `verify_passphrase` first.
    """
    deps = ctx.deps
    state = deps.state
    clean_name = " ".join(name.split())
    if state.ended:
        return GetStoreResult(status=ENDED, store_name=clean_name, message=ENDED_MESSAGE)

    if state.passphrase_locked:
        return GetStoreResult(
            status="locked",
            store_name=clean_name,
            message="Lookups are locked for this session after repeated passphrase failures.",
        )
    if not state.passphrase_verified:
        return GetStoreResult(
            status="passphrase_required",
            store_name=clean_name,
            message="The passphrase must be verified before looking up a store.",
        )

    record = deps.repo.get_store(clean_name)
    if record is None:
        _record(deps, f"Lookup for '{clean_name}' found nothing")
        return GetStoreResult(
            status="not_found",
            store_name=clean_name,
            message=f"No store named '{clean_name}' is saved.",
        )
    state.off_topic_strikes = 0
    if deps.settings.require_passphrase_per_lookup:
        state.reset_passphrase()
    _record(deps, f"Retrieved '{record.name}' ({record.phone_e164})")
    return GetStoreResult(
        status="found",
        store_name=record.name,
        phone=record.phone_e164,
        message=f"'{record.name}' has phone {record.phone_e164}.",
    )


def flag_off_topic(ctx: RunContext[Deps]) -> OffTopicResult:
    """Record that the user asked about something outside saving or retrieving stores.

    Call this once per off-topic user message. After repeated off-topic messages the session
    ends automatically; when `session_ended` is true, say goodbye.
    """
    deps = ctx.deps
    state = deps.state
    limit = deps.settings.off_topic_strike_limit
    if state.ended:
        return OffTopicResult(
            status=ENDED,
            strikes=state.off_topic_strikes,
            strikes_remaining=0,
            session_ended=True,
            message=ENDED_MESSAGE,
        )
    state.off_topic_strikes += 1
    remaining = max(limit - state.off_topic_strikes, 0)
    if remaining == 0:
        state.end("off_scope")
        _record(deps, "Session ended after repeated off-topic requests")
        return OffTopicResult(
            status=ENDED,
            strikes=state.off_topic_strikes,
            strikes_remaining=0,
            session_ended=True,
            message="The conversation is ending because it stayed off-topic. Say goodbye.",
        )
    _record(deps, "Off-topic request redirected")
    return OffTopicResult(
        status="redirected",
        strikes=state.off_topic_strikes,
        strikes_remaining=remaining,
        session_ended=False,
        message=(
            f"Off-topic strike {state.off_topic_strikes} of {limit}. Redirect the user to "
            "saving or retrieving stores."
        ),
    )


def end_session(ctx: RunContext[Deps]) -> EndSessionResult:
    """End the conversation because the user said they are finished (for example "I'm done")."""
    deps = ctx.deps
    state = deps.state
    if state.ended:
        return EndSessionResult(status=ENDED, message=ENDED_MESSAGE)
    state.end("user_done")
    _record(deps, "User ended the session")
    return EndSessionResult(
        status="ended", message="Session ended. Say a brief goodbye; a summary will be saved."
    )


ALL_TOOLS: list[Callable[..., Any]] = [
    save_store,
    verify_passphrase,
    get_store,
    flag_off_topic,
    end_session,
]
