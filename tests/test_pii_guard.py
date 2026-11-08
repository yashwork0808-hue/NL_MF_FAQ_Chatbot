import pytest

from src.query.pii_guard import PIIGuard


@pytest.mark.parametrize(
    "text,raw,expected_type",
    [
        ("My PAN is ABCDE1234F", "ABCDE1234F", "PAN"),
        ("Aadhaar 1234 5678 9012", "1234 5678 9012", "Aadhaar"),
        ("account number 123456789012", "123456789012", "account_number"),
        ("OTP is 123456", "123456", "OTP"),
        ("email me at user@example.com", "user@example.com", "email"),
        ("call 9876543210", "9876543210", "phone"),
        ("call +91 98765 43210", "98765 43210", "phone"),
    ],
)
def test_pii_is_detected_and_redacted(text, raw, expected_type):
    result = PIIGuard().scan(text)

    assert result.has_pii is True
    assert expected_type in result.detected_types
    assert raw not in result.redacted_text
    assert "[REDACTED_" in result.redacted_text


def test_clean_question_is_unchanged():
    text = "What is the ELSS lock-in period?"
    result = PIIGuard().scan(text)

    assert result.has_pii is False
    assert result.redacted_text == text
    assert result.detected_types == []
