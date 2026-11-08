from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple


PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
AADHAAR_PATTERN = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
)
OTP_PATTERN = re.compile(
    r"\b(?:otp|one[\s-]?time[\s-]?password|verification\s+code)\b"
    r"(?:\s*(?:is|:|-))?\s*\d{4,8}\b",
    re.IGNORECASE,
)
ACCOUNT_PATTERN = re.compile(
    r"\b(?:account|a/c|acct)(?:\s*(?:number|no\.?))?\b\D{0,20}\d{6,18}\b",
    re.IGNORECASE,
)
LONG_NUMBER_PATTERN = re.compile(r"(?<!\d)\d{11,18}(?!\d)")

PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("PAN", PAN_PATTERN),
    ("email", EMAIL_PATTERN),
    ("phone", PHONE_PATTERN),
    ("OTP", OTP_PATTERN),
    ("account_number", ACCOUNT_PATTERN),
    ("Aadhaar", AADHAAR_PATTERN),
    ("account_number", LONG_NUMBER_PATTERN),
)


@dataclass(frozen=True)
class PIIResult:
    has_pii: bool
    redacted_text: str
    detected_types: List[str]


class PIIGuard:
    def scan(self, text: str) -> PIIResult:
        spans: List[Tuple[int, int, str, int]] = []
        for priority, (pii_type, pattern) in enumerate(PATTERNS):
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end(), pii_type, priority))

        if not spans:
            return PIIResult(False, text, [])

        spans.sort(key=lambda item: (item[0], item[1], item[3]))
        merged: List[List[object]] = []
        for start, end, pii_type, priority in spans:
            if not merged or start >= int(merged[-1][1]):
                merged.append([start, end, pii_type, priority])
                continue
            if end > int(merged[-1][1]):
                merged[-1][1] = end
            existing_priority = int(merged[-1][3])
            if priority < existing_priority:
                merged[-1][2] = pii_type
                merged[-1][3] = priority

        pieces: List[str] = []
        detected_types: List[str] = []
        cursor = 0
        for start, end, pii_type, _ in merged:
            start = int(start)
            end = int(end)
            pieces.append(text[cursor:start])
            pieces.append(f"[REDACTED_{pii_type.upper()}]")
            if pii_type not in detected_types:
                detected_types.append(str(pii_type))
            cursor = end
        pieces.append(text[cursor:])
        return PIIResult(True, "".join(pieces), detected_types)


def scan_for_pii(text: str) -> PIIResult:
    return PIIGuard().scan(text)
