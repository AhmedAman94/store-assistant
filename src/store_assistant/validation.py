"""Phone number validation and normalization.

Uses Google's libphonenumber (via the `phonenumbers` package). Numbers without a country code are
interpreted in the default region (US); numbers with a leading `+` are parsed as international.
Valid numbers are normalized to E.164 (for example `+15551234567`) so every stored phone has one
canonical form.
"""

from __future__ import annotations

from dataclasses import dataclass

import phonenumbers
from phonenumbers import NumberParseException


@dataclass(frozen=True, slots=True)
class PhoneValidation:
    ok: bool
    e164: str | None = None
    pretty: str | None = None
    error: str | None = None


def normalize_phone(raw: str, default_region: str = "US") -> PhoneValidation:
    """Validate a phone string and return its E.164 form, or a human-readable error."""
    text = (raw or "").strip()
    if not text:
        return PhoneValidation(ok=False, error="Phone number is empty.")
    try:
        parsed = phonenumbers.parse(text, default_region)
    except NumberParseException:
        return PhoneValidation(
            ok=False,
            error=(
                f"'{text}' does not look like a phone number. Use a format like 415-555-2671, "
                "(415) 555-2671, or +44 20 7946 0958 for international numbers."
            ),
        )
    if not phonenumbers.is_valid_number(parsed):
        return PhoneValidation(
            ok=False,
            error=(
                f"'{text}' is not a valid phone number for its region. Check the digit count "
                "and area code, and include a country code (for example +44) for non-US numbers."
            ),
        )
    return PhoneValidation(
        ok=True,
        e164=phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
        pretty=phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
    )


def normalize_store_name(name: str) -> str:
    """Canonical key for case-insensitive, whitespace-insensitive store lookups."""
    return " ".join((name or "").split()).casefold()


def mask_phone(e164: str) -> str:
    """Hide all but the last two digits, for displays that sit outside the passphrase gate."""
    digits = e164.lstrip("+")
    if len(digits) <= 2:
        return "*" * len(digits)
    return "+" + "*" * (len(digits) - 2) + digits[-2:]
