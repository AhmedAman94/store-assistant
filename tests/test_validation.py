import pytest

from store_assistant.validation import normalize_phone, normalize_store_name


@pytest.mark.parametrize(
    ("raw", "e164"),
    [
        ("(415) 555-2671", "+14155552671"),
        ("415-555-2671", "+14155552671"),
        ("415.555.2671", "+14155552671"),
        ("4155552671", "+14155552671"),
        ("+1 415 555 2671", "+14155552671"),
        ("+44 20 7946 0958", "+442079460958"),
        ("  +44 (0)20 7946 0958 ", "+442079460958"),
    ],
)
def test_valid_phones_normalize_to_e164(raw: str, e164: str) -> None:
    result = normalize_phone(raw)
    assert result.ok, result.error
    assert result.e164 == e164


@pytest.mark.parametrize(
    "raw", ["", "   ", "12345", "555-123", "call me", "123-456-789012", "+1 000 000 0000"]
)
def test_invalid_phones_are_rejected_with_a_message(raw: str) -> None:
    result = normalize_phone(raw)
    assert not result.ok
    assert result.e164 is None
    assert result.error


def test_region_default_is_configurable() -> None:
    assert normalize_phone("020 7946 0958", default_region="GB").e164 == "+442079460958"
    assert not normalize_phone("020 7946 0958", default_region="US").ok


def test_store_names_normalize_case_and_whitespace() -> None:
    assert normalize_store_name("  Trader   Joe's ") == "trader joe's"
    assert normalize_store_name("WHOLE FOODS") == normalize_store_name("whole foods")
