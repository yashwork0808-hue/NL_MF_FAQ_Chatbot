from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

from src.ingest.registry import Source, get_sources, validate_source_url


LOGGER = logging.getLogger(__name__)
ALLOWED_HOSTS = frozenset({"groww.in", "www.groww.in"})


class LoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Document:
    source_id: str
    source_url: str
    final_url: str
    scheme: str
    page_type: str
    page_label: str
    fetched_at: str
    checksum: str
    content_type: str
    content_length: int
    file_path: str
    raw_html: str


@dataclass
class LoadReport:
    generated_at: str
    expected_count: int
    documents: List[Document]
    skipped: List[Dict[str, str]]
    warnings: List[str]
    report_path: str

    @property
    def ok(self) -> bool:
        return not self.skipped and len(self.documents) == self.expected_count


class DocumentLoader:
    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir) if raw_dir is not None else Path("data/raw")
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.raw_dir / "manifest.json"
        )

    def load(self, sources: Optional[Sequence[Source]] = None) -> LoadReport:
        expected_sources = list(sources) if sources is not None else list(get_sources())
        manifest = self._read_manifest()
        entries = self._get_entries(manifest)
        warnings: List[str] = []
        skipped: List[Dict[str, str]] = []
        documents: List[Document] = []

        declared_count = manifest.get("document_count")
        if declared_count != len(entries):
            warnings.append(
                f"Manifest document_count={declared_count!r} does not match "
                f"the number of document entries ({len(entries)})"
            )
        if manifest.get("errors"):
            warnings.append(
                f"Scrape manifest contains {len(manifest['errors'])} scrape error(s)"
            )

        entries_by_url: Dict[str, Dict[str, object]] = {}
        for entry in entries:
            source_url = entry.get("source_url")
            if not isinstance(source_url, str) or not source_url:
                warnings.append("Ignored manifest entry without a source_url")
                continue
            if source_url in entries_by_url:
                warnings.append(f"Duplicate manifest entry ignored: {source_url}")
                continue
            entries_by_url[source_url] = entry

        expected_urls = {source.url for source in expected_sources}
        if len(expected_urls) != len(expected_sources):
            raise LoadError("Registry contains duplicate source URLs")
        if len(expected_sources) != 5:
            warnings.append(
                f"Registry contains {len(expected_sources)} source(s); expected 5"
            )

        for source_url in entries_by_url:
            if source_url not in expected_urls:
                warnings.append(f"Ignored non-allowlisted manifest entry: {source_url}")

        for source in expected_sources:
            entry = entries_by_url.get(source.url)
            if entry is None:
                skipped.append(
                    {
                        "source_id": source.source_id,
                        "url": source.url,
                        "reason": "missing from manifest",
                    }
                )
                continue
            try:
                document = self._load_entry(source, entry)
            except Exception as exc:
                skipped.append(
                    {
                        "source_id": source.source_id,
                        "url": source.url,
                        "reason": str(exc),
                    }
                )
                LOGGER.error("Failed to load %s: %s", source.url, exc)
            else:
                documents.append(document)
                LOGGER.info("Loaded %s", source.url)

        generated_at = datetime.now(timezone.utc).isoformat()
        report_path = self.raw_dir / "load_report.json"
        report = LoadReport(
            generated_at=generated_at,
            expected_count=len(expected_sources),
            documents=documents,
            skipped=skipped,
            warnings=warnings,
            report_path=str(report_path),
        )
        self._write_report(report)
        return report

    def _read_manifest(self) -> Dict[str, object]:
        if not self.manifest_path.exists():
            raise LoadError(f"Scrape manifest not found: {self.manifest_path}")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LoadError(f"Unable to read scrape manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise LoadError("Scrape manifest must be a JSON object")
        return manifest

    @staticmethod
    def _get_entries(manifest: Dict[str, object]) -> List[Dict[str, object]]:
        entries = manifest.get("documents")
        if not isinstance(entries, list):
            raise LoadError("Scrape manifest must contain a documents list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise LoadError("Each manifest document entry must be an object")
        return entries

    def _load_entry(
        self,
        source: Source,
        entry: Dict[str, object],
    ) -> Document:
        validate_source_url(source.url)
        final_url = entry.get("final_url") or source.url
        if not isinstance(final_url, str):
            raise LoadError("final_url must be a string")
        if urlparse(final_url).hostname not in ALLOWED_HOSTS:
            raise LoadError(f"final_url is outside the approved host: {final_url}")

        file_path = self._resolve_document_path(source, entry)
        content = file_path.read_bytes()
        if not content:
            raise LoadError(f"Raw document is empty: {file_path}")
        raw_html = self._decode(content, entry.get("content_type"))
        if not raw_html.strip():
            raise LoadError(f"Raw document contains no text: {file_path}")

        checksum = hashlib.sha256(content).hexdigest()
        expected_checksum = entry.get("checksum")
        if not isinstance(expected_checksum, str) or not expected_checksum:
            raise LoadError("checksum is missing from the manifest entry")
        if checksum != expected_checksum:
            raise LoadError(
                f"Checksum mismatch for {source.url}: "
                f"expected {expected_checksum}, got {checksum}"
            )

        content_type = entry.get("content_type") or ""
        if not isinstance(content_type, str):
            content_type = ""
        fetched_at = entry.get("fetched_at") or ""
        if not isinstance(fetched_at, str):
            fetched_at = ""

        return Document(
            source_id=source.source_id,
            source_url=source.url,
            final_url=final_url,
            scheme=source.scheme,
            page_type=source.page_type,
            page_label=source.page_label,
            fetched_at=fetched_at,
            checksum=checksum,
            content_type=content_type,
            content_length=len(content),
            file_path=str(file_path),
            raw_html=raw_html,
        )

    def _resolve_document_path(
        self,
        source: Source,
        entry: Dict[str, object],
    ) -> Path:
        raw_root = self.raw_dir.resolve()
        manifest_path = entry.get("file_path")
        candidates = []
        if isinstance(manifest_path, str) and manifest_path:
            candidates.append(Path(manifest_path))
        candidates.append(self.raw_dir / f"{source.source_id}.html")

        for candidate in candidates:
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            resolved = candidate.resolve()
            if not resolved.is_relative_to(raw_root):
                raise LoadError(f"Document path escapes raw directory: {resolved}")
            if resolved.exists() and resolved.is_file():
                return resolved
        raise LoadError(f"Raw document not found for {source.url}")

    @staticmethod
    def _decode(content: bytes, content_type: object) -> str:
        charset = "utf-8"
        if isinstance(content_type, str) and "charset=" in content_type.lower():
            charset = (
                content_type.lower().split("charset=", 1)[1].split(";", 1)[0].strip()
            )
            charset = charset.strip("\"'")
        try:
            return content.decode(charset)
        except (LookupError, UnicodeDecodeError):
            return content.decode("utf-8", errors="replace")

    def _write_report(self, report: LoadReport) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": report.generated_at,
            "expected_count": report.expected_count,
            "loaded_count": len(report.documents),
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
                    "content_type": document.content_type,
                    "content_length": document.content_length,
                    "file_path": document.file_path,
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


def load_documents(
    raw_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    sources: Optional[Sequence[Source]] = None,
) -> LoadReport:
    return DocumentLoader(raw_dir=raw_dir, manifest_path=manifest_path).load(sources)
