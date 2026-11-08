from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from src.ingest.registry import get_allowed_urls
from src.query.retriever import RetrievalResult


MAX_SENTENCES = 3
URL_PATTERN = re.compile(r"https?://[^\s)\]>,]+", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\((https?://[^)]+)\)",
    re.IGNORECASE,
)
DATE_LINE_PATTERN = re.compile(
    r"(?im)^\s*last updated from sources\s*:.*$",
    re.IGNORECASE,
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
ADVICE_PATTERNS = (
    re.compile(r"\b(recommend|recommendation|advice|advisor)\b", re.IGNORECASE),
    re.compile(r"\bportfolio\b", re.IGNORECASE),
    re.compile(r"\byou\s+(should|must|need\s+to)\b", re.IGNORECASE),
    re.compile(r"\bbest\s+(fund|scheme|investment)\b", re.IGNORECASE),
    re.compile(r"\bguarantee(d|s)?\b", re.IGNORECASE),
    re.compile(r"\bassured\s+returns?\b", re.IGNORECASE),
    re.compile(r"\bwill\s+(give|earn|make|generate)\b", re.IGNORECASE),
    re.compile(r"\bgood\s+investment\b", re.IGNORECASE),
    re.compile(r"\bsafe\s+investment\b", re.IGNORECASE),
)
PERFORMANCE_PATTERNS = (
    re.compile(r"\breturns?\b", re.IGNORECASE),
    re.compile(r"\bprofit\b", re.IGNORECASE),
    re.compile(r"\b(performance|cagr|xirr|outperform)\b", re.IGNORECASE),
    re.compile(r"\b(compare|comparison)\b", re.IGNORECASE),
    re.compile(r"\b(annualized|absolute)\s+return\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    answer: str
    citation_url: Optional[str]
    errors: List[str]


class AnswerValidator:
    def __init__(self, allowed_urls: Optional[Sequence[str]] = None) -> None:
        self.allowed_urls: Set[str] = set(
            allowed_urls if allowed_urls is not None else get_allowed_urls()
        )

    def validate(self, answer: str, retrieval: RetrievalResult) -> ValidationResult:
        errors: List[str] = []
        text = MARKDOWN_LINK_PATTERN.sub(r"\2", answer or "")
        text = DATE_LINE_PATTERN.sub("", text).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)

        if not text:
            return ValidationResult(
                ok=False,
                answer="",
                citation_url=None,
                errors=["empty_answer"],
            )

        urls = [
            url.rstrip(".,;:!?")
            for url in URL_PATTERN.findall(text)
        ]
        unique_urls = list(dict.fromkeys(urls))
        if len(urls) != 1:
            errors.append(f"expected_exactly_one_url_found_{len(urls)}")
        if len(unique_urls) != 1:
            errors.append("expected_one_unique_url")
        citation_url = unique_urls[0] if len(unique_urls) == 1 else None
        if citation_url is not None and citation_url not in self.allowed_urls:
            errors.append("url_not_approved")
        if citation_url is None:
            errors.append("missing_citation_url")

        if retrieval.citation is not None:
            expected_url = str(retrieval.citation.metadata.get("source_url", ""))
            if citation_url is not None and citation_url != expected_url:
                errors.append("url_does_not_match_retrieved_source")

        if any(pattern.search(text) for pattern in ADVICE_PATTERNS):
            errors.append("advice_language")
        if any(pattern.search(text) for pattern in PERFORMANCE_PATTERNS):
            errors.append("performance_language")

        sentence_source = URL_PATTERN.sub("", text)
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_SPLIT_PATTERN.split(sentence_source)
            if sentence.strip()
        ]
        if not sentences:
            errors.append("empty_answer_body")
        elif len(sentences) > MAX_SENTENCES:
            errors.append(f"too_many_sentences_{len(sentences)}")

        if errors:
            return ValidationResult(
                ok=False,
                answer=text,
                citation_url=citation_url,
                errors=errors,
            )

        final_answer = (
            f"{text}\n"
            f"Last updated from sources: {retrieval.source_date}"
        )
        return ValidationResult(
            ok=True,
            answer=final_answer,
            citation_url=citation_url,
            errors=[],
        )


def validate_answer(answer: str, retrieval: RetrievalResult) -> ValidationResult:
    return AnswerValidator().validate(answer, retrieval)
