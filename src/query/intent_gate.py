from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from src.ingest.registry import Source, get_sources


FACTUAL = "factual"
ADVISORY = "advisory"
PERFORMANCE = "performance"
OUT_OF_SCOPE = "out_of_scope"

ADVISORY_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\bshould\s+i\b", re.IGNORECASE),
    re.compile(r"\bcan\s+i\s+invest\b", re.IGNORECASE),
    re.compile(r"\b(advice|recommend|recommendation|suggest|suggestion)\b", re.IGNORECASE),
    re.compile(r"\bportfolio\b", re.IGNORECASE),
    re.compile(r"\bbest\s+(fund|scheme|investment)\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+fund\s+(is|should|to|will|would)\b", re.IGNORECASE),
    re.compile(
        r"\bis\s+(this|it|the)\s+fund\s+(good|safe|worth|better)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bgood\s+(for\s+my|investment|fund)\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+should\s+i\s+invest\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+should\s+i\s+do\b", re.IGNORECASE),
    re.compile(r"\bwill\s+(this|the)\s+fund\s+(give|make|earn)\b", re.IGNORECASE),
)
PROMPT_INJECTION_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above)?\s*instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\b(jailbreak|developer\s+mode)\b", re.IGNORECASE),
    re.compile(r"\breveal\s+.*\b(prompt|instructions)\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+.*\b(assistant|ai|model)\b", re.IGNORECASE),
)
PERFORMANCE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\breturns?\b", re.IGNORECASE),
    re.compile(r"\bprofit\b", re.IGNORECASE),
    re.compile(r"\bperformance\b", re.IGNORECASE),
    re.compile(r"\bcagr\b", re.IGNORECASE),
    re.compile(r"\bxirr\b", re.IGNORECASE),
    re.compile(r"\bcompare\b|\bcomparison\b", re.IGNORECASE),
    re.compile(r"\breturn\s+calculator\b", re.IGNORECASE),
    re.compile(r"\boutperform\b", re.IGNORECASE),
    re.compile(
        r"\bhow\s+much\b.*\b(return|profit|earn)\b",
        re.IGNORECASE,
    ),
)
IN_SCOPE_TERMS: Sequence[str] = (
    "mutual fund",
    "fund",
    "scheme",
    "amc",
    "hdfc",
    "sip",
    "nav",
    "expense",
    "exit load",
    "entry load",
    "lock-in",
    "lock in",
    "lockin",
    "elss",
    "riskometer",
    "risk",
    "benchmark",
    "aum",
    "fund size",
    "minimum",
    "investment",
    "invest",
    "statement",
    "capital gain",
    "redemption",
    "holdings",
    "fund manager",
    "tax",
    "lumpsum",
    "growth",
    "direct plan",
    "rating",
    "category",
    "equity",
    "large cap",
    "mid cap",
    "flexi cap",
    "download",
)
SCHEME_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hdfc-mutual-funds": (
        "hdfc mutual fund",
        "hdfc amc",
        "hdfc fund house",
        "hdfc asset management",
    ),
    "hdfc-large-cap-fund-direct-growth": (
        "hdfc large cap fund",
        "hdfc large cap",
        "large cap fund",
        "large cap",
    ),
    "hdfc-equity-fund-direct-growth": (
        "hdfc flexi cap fund",
        "hdfc flexi cap",
        "flexi cap fund",
        "flexi cap",
        "hdfc equity fund",
        "equity fund",
    ),
    "hdfc-elss-tax-saver-fund-direct-plan-growth": (
        "hdfc elss tax saver",
        "elss tax saver",
        "hdfc elss",
        "elss",
        "tax saver",
    ),
    "hdfc-mid-cap-fund-direct-growth": (
        "hdfc mid cap fund",
        "hdfc mid cap",
        "mid cap fund",
        "mid cap",
    ),
}


@dataclass(frozen=True)
class IntentResult:
    category: str
    reason: str
    source: Optional[Source]
    source_url: Optional[str]
    scheme: Optional[str]


class IntentGate:
    def __init__(self, sources: Optional[Sequence[Source]] = None) -> None:
        self.sources = list(sources) if sources is not None else list(get_sources())
        self._source_by_id = {source.source_id: source for source in self.sources}

    def classify(self, text: str) -> IntentResult:
        cleaned = text.strip()
        if not cleaned:
            return IntentResult(
                category=OUT_OF_SCOPE,
                reason="empty_question",
                source=None,
                source_url=None,
                scheme=None,
            )

        source = self.detect_source(cleaned)
        lowered = cleaned.lower()

        if any(pattern.search(lowered) for pattern in PROMPT_INJECTION_PATTERNS):
            return self._result(OUT_OF_SCOPE, "prompt_injection", source)
        if any(pattern.search(lowered) for pattern in ADVISORY_PATTERNS):
            return self._result(ADVISORY, "advisory_or_portfolio_request", source)
        if any(pattern.search(lowered) for pattern in PERFORMANCE_PATTERNS):
            return self._result(PERFORMANCE, "performance_or_comparison_request", source)
        if source is not None or any(term in lowered for term in IN_SCOPE_TERMS):
            return self._result(FACTUAL, "covered_topic", source)
        return self._result(
            OUT_OF_SCOPE,
            "not_about_covered_mutual_fund_topics",
            None,
        )

    def detect_source(self, text: str) -> Optional[Source]:
        lowered = text.lower()
        matches: list[Tuple[int, str]] = []
        for source in self.sources:
            aliases = SCHEME_ALIASES.get(source.source_id, ())
            for alias in aliases:
                index = lowered.find(alias)
                if index >= 0:
                    matches.append((index, source.source_id))
                    break
        if not matches:
            return None
        matches.sort(key=lambda item: item[0])
        return self._source_by_id[matches[0][1]]

    @staticmethod
    def _result(category: str, reason: str, source: Optional[Source]) -> IntentResult:
        return IntentResult(
            category=category,
            reason=reason,
            source=source,
            source_url=source.url if source else None,
            scheme=source.scheme if source else None,
        )


def classify_intent(text: str) -> IntentResult:
    return IntentGate().classify(text)
