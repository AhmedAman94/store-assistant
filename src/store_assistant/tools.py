"""Tools the agent can call, and the structured results they return.

Every rule that must hold regardless of what the language model does is enforced here:
phone validation, the passphrase gate, attempt locking, the off-topic strike counter, and
session termination. The model only sees small status objects and phrases replies around them.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from store_assistant.config import Settings
from store_assistant.db import Repository
from store_assistant.session import SessionState
from store_assistant.validation import normalize_phone


@dataclass(slots=True)
class Deps:
    """Dependencies injected into every tool call."""

    settings: Settings
    repo: Repository
    state: SessionState


# ---- result models ------------------------------------------------------------------------------


class SaveStoreResult(BaseModel):
    status: Literal["saved", "updated", "invalid_phone", "needs_confirmation"]
    store_name: str
    phone: str | None = Field(default=None, description="Stored phone in E.164 form")
    existing_phone: str | None = Field(
        default=None, description="Phone currently on file when confirmation is needed"
    )
    message: str


class VerifyPassphraseResult(BaseModel):
    status: Literal["verified", "already_verified", "incorrect", "locked"]
    attempts_remaining: int
    message: str


class GetStoreResult(BaseModel):
    status: Literal["found", "not_found", "passphrase_required", "locked"]
    store_name: str
    phone: str | None = None
    message: str


class OffTopicResult(BaseModel):
    strikes: int
    strikes_remaining: int
    session_ended: bool
    message: str


class EndSessionResult(BaseModel):
    status: Literal["ended"]
    message: str


# ---- tools ---------------------------------------------------------------------------------------


def save_store(
    ctx: RunContext[Deps], name: str, phone: str, confirm_overwrite: bool = False
) -> SaveStoreResult:
    """Save a store's name and phone number to the database.

    Call this once you have both a store name and a phone number from the user. The phone is
    validated here; on `invalid_phone` ask the user for the phone again. If the store already
    exists you get `needs_confirmation`: ask the user whether to overwrite, and only call again
    with `confirm_overwrite=true` after they say yes.
    """
    deps = ctx.deps
    state = deps.state
    state.off_topic_strikes = 0
    clean_name = " ".join(name.split())
    if not clean_name:
        return SaveStoreResult(
            status="invalid_phone",
            store_name=clean_name,
            message="Store name is empty. Ask the user for the store name.",
        )

    validation = normalize_phone(phone, deps.settings.default_phone_region)
    if not validation.ok or validation.e164 is None:
        state.record(f"Rejected invalid phone for '{clean_name}'")
        return SaveStoreResult(
            status="invalid_phone",
            store_name=clean_name,
            message=validation.error or "Invalid phone number.",
        )

    existing = deps.repo.get_store(clean_name)
    if existing is not None and not confirm_overwrite:
        state.record(f"Asked to confirm overwrite of '{existing.name}'")
        return SaveStoreResult(
            status="needs_confirmation",
            store_name=existing.name,
            existing_phone=existing.phone_e164,
            message=(
                f"'{existing.name}' already exists with phone {existing.phone_e164}. "
                "Ask the user to confirm before overwriting."
            ),
        )

    record, created = deps.repo.save_store(clean_name, validation.e164)
    verb = "Saved" if created else "Updated"
    state.record(f"{verb} store '{record.name}' with phone {record.phone_e164}")
    return SaveStoreResult(
        status="saved" if created else "updated",
        store_name=record.name,
        phone=record.phone_e164,
        message=f"{verb} '{record.name}' with phone {validation.pretty}.",
    )


def verify_passphrase(ctx: RunContext[Deps], passphrase: str) -> VerifyPassphraseResult:
    """Check the secret passphrase the user typed. Required before any store lookup.

    Pass exactly what the user typed. You do not know the passphrase and must never guess it.
    """
    deps = ctx.deps
    state = deps.state
    state.off_topic_strikes = 0
    max_attempts = deps.settings.passphrase_max_attempts

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
    supplied = (passphrase or "").strip().casefold()
    if supplied and hmac.compare_digest(expected.encode(), supplied.encode()):
        state.passphrase_verified = True
        state.passphrase_attempts = 0
        state.record("Passphrase verified")
        return VerifyPassphraseResult(
            status="verified",
            attempts_remaining=max_attempts,
            message="Passphrase accepted. You may now look up stores.",
        )

    state.passphrase_attempts += 1
    remaining = max(max_attempts - state.passphrase_attempts, 0)
    if remaining == 0:
        state.passphrase_locked = True
        state.record("Passphrase locked after repeated failures")
        return VerifyPassphraseResult(
            status="locked",
            attempts_remaining=0,
            message="Incorrect passphrase. Lookups are now locked for this session.",
        )
    state.record("Incorrect passphrase attempt")
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
    state.off_topic_strikes = 0
    clean_name = " ".join(name.split())

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
    if deps.settings.require_passphrase_per_lookup:
        state.reset_passphrase()
    if record is None:
        state.record(f"Lookup for '{clean_name}' found nothing")
        return GetStoreResult(
            status="not_found",
            store_name=clean_name,
            message=f"No store named '{clean_name}' is saved.",
        )
    state.record(f"Retrieved '{record.name}' ({record.phone_e164})")
    return GetStoreResult(
        status="found",
        store_name=record.name,
        phone=record.phone_e164,
        message=f"'{record.name}' has phone {record.phone_e164}.",
    )


def flag_off_topic(ctx: RunContext[Deps]) -> OffTopicResult:
    """Record that the user asked about something outside saving or retrieving stores.

    Call this whenever a user message is off-topic. After repeated off-topic messages the session
    ends automatically; when `session_ended` is true, say goodbye.
    """
    state = ctx.deps.state
    limit = ctx.deps.settings.off_topic_strike_limit
    state.off_topic_strikes += 1
    remaining = max(limit - state.off_topic_strikes, 0)
    if remaining == 0:
        state.end("off_scope")
        state.record("Session ended after repeated off-topic requests")
        return OffTopicResult(
            strikes=state.off_topic_strikes,
            strikes_remaining=0,
            session_ended=True,
            message="The conversation is ending because it stayed off-topic. Say goodbye.",
        )
    state.record("Off-topic request redirected")
    return OffTopicResult(
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
    state = ctx.deps.state
    state.end("user_done")
    state.record("User ended the session")
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
