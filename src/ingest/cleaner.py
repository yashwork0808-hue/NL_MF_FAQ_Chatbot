from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bs4 import BeautifulSoup

from src.ingest.loader import Document, DocumentLoader


LOGGER = logging.getLogger(__name__)
REMOVAL_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "canvas",
    "form",
    "button",
    "nav",
    "footer",
    "aside",
)
REMOVAL_SELECTORS = (
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']",
    "[aria-hidden='true']",
    "[class*='footer']",
    "[class*='header2025']",
    "[class*='nav']",
    "[class*='menu']",
    "[class*='breadcrumb']",
    "[class*='cookie']",
    "[class*='advert']",
)
MAIN_SELECTORS = (
    "main",
    "article",
    "[role='main']",
    "div.pw14MainWrapper",
    "div.amcMainPage",
    "div.layout-main",
    "div[class*='MainWrapper']",
    "div[class*='MainPage']",
    "div[class*='ContentWrapper']",
)
KEY_FACTS = (
    "expense ratio",
    "exit load",
    "minimum sip",
    "min. for sip",
    "lock-in",
    "riskometer",
    "benchmark",
)


class CleanError(RuntimeError):
    pass


@dataclass(frozen=True)
class CleanedDocument:
    source_id: str
    source_url: str
    final_url: str
    scheme: str
    page_type: str
    page_label: str
    fetched_at: str
    checksum: str
    text: str
    character_count: int
    line_count: int
    missing_key_facts: List[str]
    source_file: str
    cleaned_file: str


@dataclass
class CleanReport:
    generated_at: str
    expected_count: int
    documents: List[CleanedDocument]
    skipped: List[Dict[str, str]]
    warnings: List[str]
    report_path: str

    @property
    def ok(self) -> bool:
        return not self.skipped and len(self.documents) == self.expected_count


class DocumentCleaner:
    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        cleaned_dir: Optional[Path] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir) if raw_dir is not None else Path("data/raw")
        self.cleaned_dir = (
            Path(cleaned_dir) if cleaned_dir is not None else Path("data/cleaned")
        )

    def clean(
        self,
        documents: Optional[Sequence[Document]] = None,
    ) -> CleanReport:
        if documents is None:
            documents = DocumentLoader(raw_dir=self.raw_dir).load().documents
        expected_count = len(documents)
        cleaned_documents: List[CleanedDocument] = []
        skipped: List[Dict[str, str]] = []
        warnings: List[str] = []
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        for document in documents:
            try:
                cleaned = self._clean_document(document)
            except Exception as exc:
                skipped.append(
                    {
                        "source_id": document.source_id,
                        "url": document.source_url,
                        "reason": str(exc),
                    }
                )
                LOGGER.error("Failed to clean %s: %s", document.source_url, exc)
                continue
            cleaned_documents.append(cleaned)
            if cleaned.missing_key_facts and document.page_type == "scheme":
                warnings.append(
                    f"{document.source_id} is missing key facts: "
                    + ", ".join(cleaned.missing_key_facts)
                )
            if cleaned.character_count < 1000:
                warnings.append(
                    f"{document.source_id} cleaned text is short: "
                    f"{cleaned.character_count} characters"
                )
            LOGGER.info("Cleaned %s", document.source_url)

        generated_at = datetime.now(timezone.utc).isoformat()
        report_path = self.cleaned_dir / "report.json"
        report = CleanReport(
            generated_at=generated_at,
            expected_count=expected_count,
            documents=cleaned_documents,
            skipped=skipped,
            warnings=warnings,
            report_path=str(report_path),
        )
        self._write_report(report)
        return report

    def _clean_document(self, document: Document) -> CleanedDocument:
        soup = BeautifulSoup(document.raw_html, "html.parser")
        self._remove_chrome(soup)
        root = self._select_main_content(soup)
        text = self._normalize_text(root.get_text("\n"))
        if not text.strip():
            raise CleanError("Cleaned text is empty")

        lowered = text.lower()
        missing_key_facts = [
            fact for fact in KEY_FACTS if fact not in lowered
        ]
        cleaned_file = self.cleaned_dir / f"{document.source_id}.txt"
        cleaned_file.write_text(text + "\n", encoding="utf-8")

        return CleanedDocument(
            source_id=document.source_id,
            source_url=document.source_url,
            final_url=document.final_url,
            scheme=document.scheme,
            page_type=document.page_type,
            page_label=document.page_label,
            fetched_at=document.fetched_at,
            checksum=document.checksum,
            text=text,
            character_count=len(text),
            line_count=text.count("\n") + 1,
            missing_key_facts=missing_key_facts,
            source_file=document.file_path,
            cleaned_file=str(cleaned_file),
        )

    @staticmethod
    def _remove_chrome(soup: BeautifulSoup) -> None:
        for tag_name in REMOVAL_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        for selector in REMOVAL_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()

    @staticmethod
    def _select_main_content(soup: BeautifulSoup) -> BeautifulSoup:
        for selector in MAIN_SELECTORS:
            for candidate in soup.select(selector):
                if len(candidate.get_text(" ", strip=True)) >= 500:
                    return candidate
        return soup.body or soup

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\xa0", " ").replace("\u200b", "").replace("\r", "")
        lines: List[str] = []
        for line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if not line:
                continue
            if lines and line == lines[-1]:
                continue
            lines.append(line)
        return "\n".join(lines)

    def _write_report(self, report: CleanReport) -> None:
        payload = {
            "generated_at": report.generated_at,
            "expected_count": report.expected_count,
            "cleaned_count": len(report.documents),
            "documents": [
                {
                    "source_id": document.source_id,
                    "source_url": document.source_url,
                    "final_url": document.final_url,
                    "scheme": document.scheme,
                    "page_type": document.page_type,
                    "page_label": document.page_label,
                    "fetched_at": document.fetched_at,
                    "checksum": document.checksum,
                    "character_count": document.character_count,
                    "line_count": document.line_count,
                    "missing_key_facts": document.missing_key_facts,
                    "cleaned_file": document.cleaned_file,
                }
                for document in report.documents
            ],
            "skipped": report.skipped,
            "warnings": report.warnings,
        }
        report_path = Path(report.report_path)
        report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def clean_documents(
    raw_dir: Optional[Path] = None,
    cleaned_dir: Optional[Path] = None,
    documents: Optional[Sequence[Document]] = None,
) -> CleanReport:
    return DocumentCleaner(raw_dir=raw_dir, cleaned_dir=cleaned_dir).clean(documents)
