from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    scheme: str
    page_type: str
    page_label: str


SOURCES: Tuple[Source, ...] = (
    Source(
        source_id="hdfc-mutual-funds",
        url="https://groww.in/mutual-funds/amc/hdfc-mutual-funds",
        scheme="HDFC Mutual Fund",
        page_type="amc",
        page_label="AMC Overview",
    ),
    Source(
        source_id="hdfc-large-cap-fund-direct-growth",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        scheme="HDFC Large Cap Fund",
        page_type="scheme",
        page_label="Large Cap",
    ),
    Source(
        source_id="hdfc-equity-fund-direct-growth",
        url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        scheme="HDFC Flexi Cap Fund",
        page_type="scheme",
        page_label="Flexi Cap",
    ),
    Source(
        source_id="hdfc-elss-tax-saver-fund-direct-plan-growth",
        url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        scheme="HDFC ELSS Tax Saver Fund",
        page_type="scheme",
        page_label="ELSS",
    ),
    Source(
        source_id="hdfc-mid-cap-fund-direct-growth",
        url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        scheme="HDFC Mid Cap Fund",
        page_type="scheme",
        page_label="Mid Cap",
    ),
)

_ALLOWED_URLS: FrozenSet[str] = frozenset(source.url for source in SOURCES)
_SOURCES_BY_URL = {source.url: source for source in SOURCES}


def get_sources() -> Tuple[Source, ...]:
    return SOURCES


def get_allowed_urls() -> FrozenSet[str]:
    return _ALLOWED_URLS


def is_allowed_url(url: str) -> bool:
    return url in _ALLOWED_URLS


def get_source_by_url(url: str) -> Optional[Source]:
    return _SOURCES_BY_URL.get(url)


def validate_source_url(url: str) -> Source:
    source = get_source_by_url(url)
    if source is None:
        raise ValueError(f"URL is not in the approved source allowlist: {url}")
    return source
