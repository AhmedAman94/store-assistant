"""Unit tests for the rules enforced in code: validation, passphrase gate, strikes, termination."""

from pydantic_ai import RunContext

from store_assistant.tools import (
    Deps,
    end_session,
    flag_off_topic,
    get_store,
    save_store,
    verify_passphrase,
)

Ctx = RunContext[Deps]


def test_save_store_rejects_invalid_phone_and_writes_nothing(ctx: Ctx) -> None:
    result = save_store(ctx, "Trader Joe's", "555-123")
    assert result.status == "invalid_phone"
    assert "not a valid phone number" in result.message
    assert ctx.deps.repo.get_store("Trader Joe's") is None


def test_save_store_rejects_empty_or_oversized_name(ctx: Ctx) -> None:
    assert save_store(ctx, "   ", "415-555-2671").status == "invalid_name"
    assert save_store(ctx, "x" * 200, "415-555-2671").status == "invalid_name"
    assert ctx.deps.repo.list_stores() == []


def test_save_store_normalizes_phone_and_confirms(ctx: Ctx) -> None:
    result = save_store(ctx, "Trader Joe's", "(415) 555-2671")
    assert result.status == "saved"
    assert result.phone == "+14155552671"
    assert ctx.deps.repo.get_store("trader joe's") is not None
    assert any("Saved store" in op for op in ctx.deps.state.operations)


def test_duplicate_store_requires_confirmation_before_overwrite(ctx: Ctx) -> None:
    save_store(ctx, "Whole Foods", "212-555-0199")
    blocked = save_store(ctx, "whole foods", "646-555-0123")
    assert blocked.status == "needs_confirmation"
    assert blocked.existing_phone == "+12125550199"
    store = ctx.deps.repo.get_store("Whole Foods")
    assert store is not None and store.phone_e164 == "+12125550199"

    ctx.deps.state.turn += 1  # the user answers on the next turn
    confirmed = save_store(ctx, "whole foods", "646-555-0123", confirm_overwrite=True)
    assert confirmed.status == "updated"
    store = ctx.deps.repo.get_store("Whole Foods")
    assert store is not None and store.phone_e164 == "+16465550123"


def test_confirm_overwrite_flag_is_ignored_without_a_pending_confirmation(ctx: Ctx) -> None:
    """The model cannot skip the confirmation by setting the flag on its first call."""
    save_store(ctx, "Costco", "415-555-0101")
    sneaky = save_store(ctx, "Costco", "415-555-0102", confirm_overwrite=True)
    assert sneaky.status == "needs_confirmation"
    store = ctx.deps.repo.get_store("Costco")
    assert store is not None and store.phone_e164 == "+14155550101"
    # A confirmation pending for one store does not authorize overwriting another.
    save_store(ctx, "Target", "415-555-0103")
    other = save_store(ctx, "Target", "415-555-0104", confirm_overwrite=True)
    assert other.status == "needs_confirmation"


def test_lookup_is_blocked_until_passphrase_verified(ctx: Ctx) -> None:
    ctx.deps.repo.save_store("Safeway", "+14155550100")
    assert get_store(ctx, "Safeway").status == "passphrase_required"
    assert verify_passphrase(ctx, "Open Sesame ").status == "verified"
    found = get_store(ctx, "safeway")
    assert found.status == "found"
    assert found.phone == "+14155550100"
    assert get_store(ctx, "Nowhere").status == "not_found"


def test_passphrase_supplied_out_of_band_verifies_without_the_secret(ctx: Ctx) -> None:
    ctx.deps.passphrase_supplied = True
    assert verify_passphrase(ctx, "[passphrase provided]").status == "verified"


def test_three_wrong_passphrases_lock_lookups(ctx: Ctx) -> None:
    first = verify_passphrase(ctx, "nope")
    assert first.status == "incorrect" and first.attempts_remaining == 2
    assert verify_passphrase(ctx, "still nope").attempts_remaining == 1
    assert verify_passphrase(ctx, "wrong").status == "locked"
    # Even the correct passphrase no longer helps, and lookups are refused.
    assert verify_passphrase(ctx, "open sesame").status == "locked"
    assert get_store(ctx, "Safeway").status == "locked"


def test_empty_passphrase_attempt_is_not_counted(ctx: Ctx) -> None:
    result = verify_passphrase(ctx, "   ")
    assert result.status == "incorrect"
    assert ctx.deps.state.passphrase_attempts == 0


def test_passphrase_can_be_required_per_lookup(ctx: Ctx) -> None:
    ctx.deps.settings.require_passphrase_per_lookup = True
    ctx.deps.repo.save_store("Safeway", "+14155550100")
    verify_passphrase(ctx, "open sesame")
    assert get_store(ctx, "Nowhere").status == "not_found"  # a miss does not consume it
    assert get_store(ctx, "Safeway").status == "found"
    assert get_store(ctx, "Safeway").status == "passphrase_required"


def test_off_topic_strikes_end_the_session_on_the_third(ctx: Ctx) -> None:
    assert flag_off_topic(ctx).strikes_remaining == 2
    assert flag_off_topic(ctx).strikes_remaining == 1
    third = flag_off_topic(ctx)
    assert third.session_ended
    assert ctx.deps.state.ended and ctx.deps.state.end_reason == "off_scope"


def test_only_successful_on_topic_actions_reset_strikes(ctx: Ctx) -> None:
    flag_off_topic(ctx)
    flag_off_topic(ctx)
    save_store(ctx, "Costco", "bad phone")  # failed save: no reset
    assert ctx.deps.state.off_topic_strikes == 2
    verify_passphrase(ctx, "wrong")  # failed verify: no reset
    assert ctx.deps.state.off_topic_strikes == 2
    get_store(ctx, "Costco")  # blocked lookup: no reset
    assert ctx.deps.state.off_topic_strikes == 2
    save_store(ctx, "Costco", "415-555-0101")
    assert ctx.deps.state.off_topic_strikes == 0


def test_tools_refuse_to_act_after_the_session_ended(ctx: Ctx) -> None:
    ctx.deps.repo.save_store("Safeway", "+14155550100")
    verify_passphrase(ctx, "open sesame")
    assert end_session(ctx).status == "ended"
    assert ctx.deps.state.end_reason == "user_done"
    assert save_store(ctx, "Costco", "415-555-0101").status == "session_ended"
    assert get_store(ctx, "Safeway").status == "session_ended"
    assert verify_passphrase(ctx, "open sesame").status == "session_ended"
    assert flag_off_topic(ctx).status == "session_ended"
    assert end_session(ctx).status == "session_ended"
    assert ctx.deps.repo.get_store("Costco") is None


def test_operation_notes_never_contain_the_passphrase(ctx: Ctx) -> None:
    save_store(ctx, "open sesame", "415-555-0101")
    assert all("open sesame" not in op.lower() for op in ctx.deps.state.operations)


def test_confirmation_cannot_be_given_in_the_same_turn(ctx: Ctx) -> None:
    """A model calling save twice in one turn must not count as the user confirming."""
    state = ctx.deps.state
    state.turn = 3
    save_store(ctx, "Costco", "415-555-0101")
    assert save_store(ctx, "Costco", "415-555-0102").status == "needs_confirmation"
    same_turn = save_store(ctx, "Costco", "415-555-0102", confirm_overwrite=True)
    assert same_turn.status == "needs_confirmation"
    state.turn = 4  # the user has answered on a later turn
    assert save_store(ctx, "Costco", "415-555-0102", confirm_overwrite=True).status == "updated"


def test_store_names_containing_the_passphrase_are_rejected(ctx: Ctx) -> None:
    assert save_store(ctx, "Open Sesame", "415-555-0101").status == "invalid_name"
    assert save_store(ctx, "the open  sesame shop", "415-555-0101").status == "invalid_name"
    assert save_store(ctx, "[passphrase provided]", "415-555-0101").status == "invalid_name"
    assert ctx.deps.repo.list_stores() == []


def test_not_found_lookup_does_not_reset_strikes(ctx: Ctx) -> None:
    verify_passphrase(ctx, "open sesame")
    flag_off_topic(ctx)
    assert get_store(ctx, "Nowhere").status == "not_found"
    assert ctx.deps.state.off_topic_strikes == 1
