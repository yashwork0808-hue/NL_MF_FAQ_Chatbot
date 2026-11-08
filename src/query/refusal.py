from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.ingest.registry import Source, get_sources
from src.query.intent_gate import ADVISORY, PERFORMANCE, IntentResult


@dataclass(frozen=True)
class RefusalResult:
    answer: str
    source_url: str
    source_date: str
    category: str
    reason: str


class RefusalHandler:
    def __init__(self) -> None:
        self.sources = list(get_sources())
        self.default_source = self.sources[0]

    def build(
        self,
        intent: IntentResult,
        source_date: str,
        category: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> RefusalResult:
        refusal_category = category or intent.category
        refusal_reason = reason or intent.reason
        source = intent.source or self.default_source
        url = source.url

        if refusal_category == ADVISORY:
            text = (
                "I can’t provide investment advice, recommendations, or portfolio "
                f"guidance. For factual scheme information, please see {url}."
            )
        elif refusal_category == PERFORMANCE:
            text = (
                "I can’t calculate or compare returns. Please refer to the official "
                f"scheme page or factsheet for performance information: {url}."
            )
        elif refusal_reason == "empty_question":
            text = (
                "Please ask a factual question about the covered HDFC Mutual Fund "
                "schemes. You can ask about expense ratio, exit load, minimum SIP, "
                f"lock-in, benchmark, or statements: {url}."
            )
        elif refusal_category == "pii":
            text = (
                "I can’t process requests containing PAN, Aadhaar, account numbers, "
                "OTPs, email addresses, or phone numbers. Please ask a factual "
                f"question without personal details: {url}."
            )
        else:
            text = (
                "I can only answer factual questions about the covered HDFC Mutual "
                "Fund schemes. Please ask about expense ratio, exit load, minimum "
                f"SIP, lock-in, benchmark, or statements: {url}."
            )

        answer = f"{text}\nLast updated from sources: {source_date}"
        return RefusalResult(
            answer=answer,
            source_url=url,
            source_date=source_date,
            category=refusal_category,
            reason=refusal_reason,
        )


def build_refusal(
    intent: IntentResult,
    source_date: str,
    category: Optional[str] = None,
    reason: Optional[str] = None,
) -> RefusalResult:
    return RefusalHandler().build(
        intent,
        source_date,
        category=category,
        reason=reason,
    )
