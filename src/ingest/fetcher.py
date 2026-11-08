from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

from src.config import Settings, settings as default_settings
from src.ingest.registry import Source, validate_source_url


LOGGER = logging.getLogger(__name__)
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
ALLOWED_HOSTS = frozenset({"groww.in", "www.groww.in"})


class ScrapeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScrapedDocument:
    source_id: str
    source_url: str
    final_url: str
    scheme: str
    page_type: str
    page_label: str
    fetched_at: str
    checksum: str
    status_code: int
    content_type: str
    content_length: int
    file_path: str


@dataclass
class ScrapeReport:
    generated_at: str
    documents: List[ScrapedDocument]
    errors: List[Dict[str, str]]
    manifest_path: str

    @property
    def ok(self) -> bool:
        return not self.errors


class Fetcher:
    def __init__(
        self,
        settings: Settings = default_settings,
        raw_dir: Optional[Path] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.settings = settings
        self.raw_dir = Path(raw_dir) if raw_dir is not None else Path("data/raw")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def close(self) -> None:
        self.session.close()

    def fetch(self, source: Source) -> ScrapedDocument:
        validate_source_url(source.url)
        response = self._get(source.url)
        try:
            content = response.content
            if not content:
                raise ScrapeError(f"Empty response body for {source.url}")
            final_url = response.url
            if not self._is_allowed_host(final_url):
                raise ScrapeError(
                    f"Request for {source.url} redirected outside groww.in to {final_url}"
                )
            fetched_at = datetime.now(timezone.utc).isoformat()
            checksum = hashlib.sha256(content).hexdigest()
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.raw_dir / f"{source.source_id}.html"
            file_path.write_bytes(content)
            return ScrapedDocument(
                source_id=source.source_id,
                source_url=source.url,
                final_url=final_url,
                scheme=source.scheme,
                page_type=source.page_type,
                page_label=source.page_label,
                fetched_at=fetched_at,
                checksum=checksum,
                status_code=response.status_code,
                content_type=response.headers.get("Content-Type", ""),
                content_length=len(content),
                file_path=str(file_path),
            )
        finally:
            response.close()

    def scrape(self, sources: Iterable[Source]) -> ScrapeReport:
        documents: List[ScrapedDocument] = []
        errors: List[Dict[str, str]] = []
        for source in sources:
            try:
                document = self.fetch(source)
            except Exception as exc:
                errors.append(
                    {
                        "source_id": source.source_id,
                        "url": source.url,
                        "error": str(exc),
                    }
                )
                LOGGER.error("Failed to scrape %s: %s", source.url, exc)
            else:
                documents.append(document)
                LOGGER.info("Scraped %s", source.url)
        generated_at = datetime.now(timezone.utc).isoformat()
        manifest_path = self.raw_dir / "manifest.json"
        report = ScrapeReport(
            generated_at=generated_at,
            documents=documents,
            errors=errors,
            manifest_path=str(manifest_path),
        )
        self._write_manifest(report)
        return report

    def _get(self, url: str) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code in RETRY_STATUS_CODES:
                    last_error = ScrapeError(
                        f"Retryable status {response.status_code} for {url}"
                    )
                    response.close()
                elif response.status_code >= 400:
                    status_code = response.status_code
                    response.close()
                    raise ScrapeError(f"HTTP {status_code} for {url}")
                else:
                    return response
            if attempt < self.max_retries:
                time.sleep(self.backoff_seconds * attempt)
        raise ScrapeError(f"Failed to fetch {url}: {last_error}")

    def _write_manifest(self, report: ScrapeReport) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": report.generated_at,
            "document_count": len(report.documents),
            "documents": [asdict(document) for document in report.documents],
            "errors": report.errors,
        }
        manifest_path = Path(report.manifest_path)
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _is_allowed_host(url: str) -> bool:
        return urlparse(url).hostname in ALLOWED_HOSTS
