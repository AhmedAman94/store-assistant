"""Unit tests for the rules enforced in code: validation, passphrase gate, strikes, termination."""

from conftest import FakeCtx
from store_assistant.tools import (
    end_session,
    flag_off_topic,
    get_store,
    save_store,
    verify_passphrase,
)


def test_save_store_rejects_invalid_phone_and_writes_nothing(ctx: FakeCtx) -> None:
    result = save_store(ctx, "Trader Joe's", "555-123")  # type: ignore[arg-type]
    assert result.status == "invalid_phone"
    assert "not a valid phone number" in result.message
    assert ctx.deps.repo.get_store("Trader Joe's") is None


def test_save_store_normalizes_phone_and_confirms(ctx: FakeCtx) -> None:
    result = save_store(ctx, "Trader Joe's", "(415) 555-2671")  # type: ignore[arg-type]
    assert result.status == "saved"
    assert result.phone == "+14155552671"
    assert ctx.deps.repo.get_store("trader joe's") is not None
    assert any("Saved store" in op for op in ctx.deps.state.operations)


def test_duplicate_store_requires_confirmation_before_overwrite(ctx: FakeCtx) -> None:
    save_store(ctx, "Whole Foods", "212-555-0199")  # type: ignore[arg-type]
    blocked = save_store(ctx, "whole foods", "646-555-0123")  # type: ignore[arg-type]
    assert blocked.status == "needs_confirmation"
    assert blocked.existing_phone == "+12125550199"
    store = ctx.deps.repo.get_store("Whole Foods")
    assert store is not None and store.phone_e164 == "+12125550199"

    confirmed = save_store(ctx, "whole foods", "646-555-0123", confirm_overwrite=True)  # type: ignore[arg-type]
    assert confirmed.status == "updated"
    store = ctx.deps.repo.get_store("Whole Foods")
    assert store is not None and store.phone_e164 == "+16465550123"


def test_lookup_is_blocked_until_passphrase_verified(ctx: FakeCtx) -> None:
    ctx.deps.repo.save_store("Safeway", "+14155550100")
    assert get_store(ctx, "Safeway").status == "passphrase_required"  # type: ignore[arg-type]
    assert verify_passphrase(ctx, "Open Sesame ").status == "verified"  # type: ignore[arg-type]
    found = get_store(ctx, "safeway")  # type: ignore[arg-type]
    assert found.status == "found"
    assert found.phone == "+14155550100"
    assert get_store(ctx, "Nowhere").status == "not_found"  # type: ignore[arg-type]


def test_three_wrong_passphrases_lock_lookups(ctx: FakeCtx) -> None:
    first = verify_passphrase(ctx, "nope")  # type: ignore[arg-type]
    assert first.status == "incorrect" and first.attempts_remaining == 2
    assert verify_passphrase(ctx, "still nope").attempts_remaining == 1  # type: ignore[arg-type]
    assert verify_passphrase(ctx, "wrong").status == "locked"  # type: ignore[arg-type]
    # Even the correct passphrase no longer helps, and lookups are refused.
    assert verify_passphrase(ctx, "open sesame").status == "locked"  # type: ignore[arg-type]
    assert get_store(ctx, "Safeway").status == "locked"  # type: ignore[arg-type]


def test_passphrase_can_be_required_per_lookup(ctx: FakeCtx) -> None:
    ctx.deps.settings.require_passphrase_per_lookup = True
    ctx.deps.repo.save_store("Safeway", "+14155550100")
    verify_passphrase(ctx, "open sesame")  # type: ignore[arg-type]
    assert get_store(ctx, "Safeway").status == "found"  # type: ignore[arg-type]
    assert get_store(ctx, "Safeway").status == "passphrase_required"  # type: ignore[arg-type]


def test_off_topic_strikes_end_the_session_on_the_third(ctx: FakeCtx) -> None:
    assert flag_off_topic(ctx).strikes_remaining == 2  # type: ignore[arg-type]
    assert flag_off_topic(ctx).strikes_remaining == 1  # type: ignore[arg-type]
    third = flag_off_topic(ctx)  # type: ignore[arg-type]
    assert third.session_ended
    assert ctx.deps.state.ended and ctx.deps.state.end_reason == "off_scope"


def test_on_topic_action_resets_strikes(ctx: FakeCtx) -> None:
    flag_off_topic(ctx)  # type: ignore[arg-type]
    flag_off_topic(ctx)  # type: ignore[arg-type]
    save_store(ctx, "Costco", "415-555-0101")  # type: ignore[arg-type]
    assert ctx.deps.state.off_topic_strikes == 0


def test_end_session_marks_user_done(ctx: FakeCtx) -> None:
    assert end_session(ctx).status == "ended"  # type: ignore[arg-type]
    assert ctx.deps.state.end_reason == "user_done"
