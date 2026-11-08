from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.config import settings as default_settings
from src.ingest.registry import Source, get_sources


LOGGER = logging.getLogger(__name__)
SECTION_HEADINGS = (
    "overview",
    "fund details",
    "fund overview",
    "fund facts",
    "scheme details",
    "about the fund",
    "about",
    "investment objective",
    "returns",
    "historic returns",
    "fund performance",
    "holdings",
    "top holdings",
    "sector allocation",
    "asset allocation",
    "fund managers",
    "fund manager",
    "expense ratio",
    "exit load",
    "exit load, stamp duty and tax",
    "minimum investment",
    "minimum sip",
    "min. for sip",
    "lock-in",
    "lock in",
    "riskometer",
    "risk",
    "fund benchmark",
    "benchmark",
    "nav",
    "aum",
    "fund size",
    "rating",
    "category",
    "tax",
    "faq",
    "frequently asked questions",
    "documents",
    "fund house",
    "amc",
)
LABEL_LINES = (
    "expense ratio",
    "exit load",
    "entry load",
    "minimum sip",
    "min. for sip",
    "min. for 1st investment",
    "min. for 2nd investment",
    "minimum investment",
    "minimum lumpsum investment",
    "fund size (aum)",
    "aum",
    "nav",
    "rating",
    "benchmark",
    "fund benchmark",
    "lock-in",
    "lock in",
    "riskometer",
    "category",
    "scheme category",
    "fund manager",
    "fund managers",
    "launch date",
    "plan",
    "option",
    "entry load",
    "stamp duty",
    "tax",
)
OVERVIEW_SECTION = "Overview"


class ChunkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    source_url: str
    scheme: str
    page_type: str
    section_heading: str
    chunk_index: int
    scraped_date: str
    text: str
    token_estimate: int


@dataclass(frozen=True)
class _Unit:
    text: str
    tokens: int
    section_heading: str


@dataclass
class ChunkReport:
    generated_at: str
    expected_source_count: int
    source_count: int
    chunks: List[Chunk]
    skipped: List[Dict[str, str]]
    warnings: List[str]
    chunks_path: str
    summary_path: str

    @property
    def ok(self) -> bool:
        return (
            not self.skipped
            and self.source_count == self.expected_source_count
            and all(chunk.text.strip() for chunk in self.chunks)
        )


class DocumentChunker:
    def __init__(
        self,
        cleaned_dir: Optional[Path] = None,
        chunks_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ) -> None:
        self.cleaned_dir = (
            Path(cleaned_dir) if cleaned_dir is not None else Path("data/cleaned")
        )
        self.chunks_dir = (
            Path(chunks_dir) if chunks_dir is not None else Path("data/chunks")
        )
        self.raw_dir = Path(raw_dir) if raw_dir is not None else Path("data/raw")

    def chunk(self, sources: Optional[Sequence[Source]] = None) -> ChunkReport:
        expected_sources = list(sources) if sources is not None else list(get_sources())
        metadata = self._load_metadata()
        chunks: List[Chunk] = []
        skipped: List[Dict[str, str]] = []
        warnings: List[str] = []
        per_source: Dict[str, int] = {}

        for source in expected_sources:
            cleaned_file = self.cleaned_dir / f"{source.source_id}.txt"
            if not cleaned_file.exists():
                skipped.append(
                    {
                        "source_id": source.source_id,
                        "url": source.url,
                        "reason": f"cleaned document not found: {cleaned_file}",
                    }
                )
                continue
            text = cleaned_file.read_text(encoding="utf-8").strip()
            if not text:
                skipped.append(
                    {
                        "source_id": source.source_id,
                        "url": source.url,
                        "reason": "cleaned document is empty",
                    }
                )
                continue

            source_metadata = metadata.get(source.source_id, {})
            if not source_metadata:
                warnings.append(
                    f"No metadata found for {source.source_id}; "
                    "using registry and configured source date"
                )
            source_chunks = self._chunk_source(source, text, source_metadata)
            if not source_chunks:
                skipped.append(
                    {
                        "source_id": source.source_id,
                        "url": source.url,
                        "reason": "no chunks produced",
                    }
                )
                continue
            chunks.extend(source_chunks)
            per_source[source.source_id] = len(source_chunks)
            LOGGER.info(
                "Chunked %s into %d chunks", source.source_id, len(source_chunks)
            )

        generated_at = datetime.now(timezone.utc).isoformat()
        chunks_path = self.chunks_dir / "chunks.jsonl"
        summary_path = self.chunks_dir / "summary.json"
        report = ChunkReport(
            generated_at=generated_at,
            expected_source_count=len(expected_sources),
            source_count=len(per_source),
            chunks=chunks,
            skipped=skipped,
            warnings=warnings,
            chunks_path=str(chunks_path),
            summary_path=str(summary_path),
        )
        self._write_outputs(report, per_source)
        return report

    def _chunk_source(
        self,
        source: Source,
        text: str,
        metadata: Dict[str, object],
    ) -> List[Chunk]:
        units = self._build_units(text)
        chunk_groups = self._pack_units(units)
        scraped_date = self._scraped_date(metadata)
        source_scheme = metadata.get("scheme") or source.scheme
        source_page_type = metadata.get("page_type") or source.page_type

        chunks: List[Chunk] = []
        for index, group in enumerate(chunk_groups):
            chunk_text = "\n".join(unit.text for unit in group).strip()
            if not chunk_text:
                continue
            section_heading = group[0].section_heading or OVERVIEW_SECTION
            chunk_id = hashlib.sha256(
                (
                    f"{source.url}\n{section_heading}\n{index}\n{chunk_text}"
                ).encode("utf-8")
            ).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=source.source_id,
                    source_url=source.url,
                    scheme=str(source_scheme),
                    page_type=str(source_page_type),
                    section_heading=section_heading,
                    chunk_index=index,
                    scraped_date=scraped_date,
                    text=chunk_text,
                    token_estimate=self._token_count(chunk_text),
                )
            )
        return chunks

    def _build_units(self, text: str) -> List[_Unit]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        units: List[_Unit] = []
        current_section = OVERVIEW_SECTION
        index = 0

        while index < len(lines):
            line = lines[index]
            if self._is_heading(line):
                current_section = line.rstrip(":")

            unit_text = line
            next_index = index + 1
            if (
                self._is_label(line)
                and next_index < len(lines)
                and not self._is_heading(lines[next_index])
                and len(lines[next_index]) <= 160
            ):
                unit_text = f"{line}\n{lines[next_index]}"
                next_index += 1

            units.append(
                _Unit(
                    text=unit_text,
                    tokens=self._token_count(unit_text),
                    section_heading=current_section,
                )
            )
            index = next_index
        return units

    def _pack_units(self, units: Sequence[_Unit]) -> List[List[_Unit]]:
        target = max(1, default_settings.chunk_size)
        overlap = max(0, default_settings.chunk_overlap)
        hard_max = max(target + overlap, int(target * 1.25))
        section_flush_min = max(1, int(target * 0.75))
        min_chunk_tokens = max(80, target // 2)
        groups: List[List[_Unit]] = []
        current: List[_Unit] = []
        current_tokens = 0
        current_section: Optional[str] = None
        pending_overlap: List[_Unit] = []

        def flush() -> List[_Unit]:
            nonlocal current, current_tokens, current_section
            if current:
                groups.append(list(current))
            flushed = list(current)
            current = []
            current_tokens = 0
            current_section = None
            return flushed

        for unit in units:
            if (
                current
                and unit.section_heading != current_section
                and current_tokens >= section_flush_min
            ):
                flush()
                pending_overlap = []
            if not current and pending_overlap:
                if pending_overlap[0].section_heading == unit.section_heading:
                    current = list(pending_overlap)
                    current_tokens = sum(item.tokens for item in current)
                    current_section = unit.section_heading
                pending_overlap = []

            if current and current_tokens + unit.tokens > hard_max:
                flushed = flush()
                pending_overlap = self._take_overlap(flushed, overlap)
                if pending_overlap and pending_overlap[0].section_heading == unit.section_heading:
                    current = list(pending_overlap)
                    current_tokens = sum(item.tokens for item in current)
                    current_section = unit.section_heading
                pending_overlap = []

            if not current:
                current_section = unit.section_heading
            current.append(unit)
            current_tokens += unit.tokens

            if current_tokens >= target:
                flushed = flush()
                pending_overlap = self._take_overlap(flushed, overlap)

        if current:
            flush()
        groups = [
            group for group in groups if any(unit.text.strip() for unit in group)
        ]
        return self._merge_small_groups(groups, min_chunk_tokens, hard_max)

    @staticmethod
    def _merge_small_groups(
        groups: List[List[_Unit]],
        min_chunk_tokens: int,
        max_chunk_tokens: int,
    ) -> List[List[_Unit]]:
        merged: List[List[_Unit]] = []
        for group in groups:
            tokens = sum(unit.tokens for unit in group)
            if tokens < min_chunk_tokens and merged:
                previous_tokens = sum(unit.tokens for unit in merged[-1])
                if previous_tokens + tokens <= max_chunk_tokens:
                    merged[-1].extend(group)
                    continue
            merged.append(list(group))

        if len(merged) > 1:
            first_tokens = sum(unit.tokens for unit in merged[0])
            second_tokens = sum(unit.tokens for unit in merged[1])
            if first_tokens < min_chunk_tokens and first_tokens + second_tokens <= max_chunk_tokens:
                merged[1][0:0] = merged[0]
                merged.pop(0)

        if len(merged) > 1:
            last_tokens = sum(unit.tokens for unit in merged[-1])
            previous_tokens = sum(unit.tokens for unit in merged[-2])
            if last_tokens < min_chunk_tokens and last_tokens + previous_tokens <= max_chunk_tokens:
                merged[-2].extend(merged.pop())
        return merged

    @staticmethod
    def _take_overlap(units: Sequence[_Unit], overlap_tokens: int) -> List[_Unit]:
        if overlap_tokens <= 0 or len(units) <= 1:
            return []
        selected: List[_Unit] = []
        token_total = 0
        for unit in reversed(units[1:]):
            if token_total + unit.tokens > overlap_tokens:
                break
            selected.insert(0, unit)
            token_total += unit.tokens
        if selected and selected[0].tokens > overlap_tokens:
            return []
        return selected

    @staticmethod
    def _token_count(text: str) -> int:
        return max(1, len(re.findall(r"\S+", text)))

    @staticmethod
    def _is_heading(line: str) -> bool:
        normalized = line.lower().strip().rstrip(":")
        return normalized in SECTION_HEADINGS

    @staticmethod
    def _is_label(line: str) -> bool:
        normalized = line.lower().strip().rstrip(":")
        if normalized in LABEL_LINES:
            return True
        return normalized.startswith("nav:")

    def _scraped_date(self, metadata: Dict[str, object]) -> str:
        fetched_at = metadata.get("fetched_at")
        if isinstance(fetched_at, str) and len(fetched_at) >= 10:
            return fetched_at[:10]
        return default_settings.source_date

    def _load_metadata(self) -> Dict[str, Dict[str, object]]:
        for path in (
            self.cleaned_dir / "report.json",
            self.raw_dir / "load_report.json",
        ):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Unable to read metadata from %s: %s", path, exc)
                continue
            documents = payload.get("documents") if isinstance(payload, dict) else None
            if not isinstance(documents, list):
                continue
            metadata: Dict[str, Dict[str, object]] = {}
            for document in documents:
                if not isinstance(document, dict):
                    continue
                source_id = document.get("source_id")
                if isinstance(source_id, str):
                    metadata[source_id] = document
            if metadata:
                return metadata
        return {}

    def _write_outputs(
        self,
        report: ChunkReport,
        per_source: Dict[str, int],
    ) -> None:
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = Path(report.chunks_path)
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in report.chunks:
                handle.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_id": chunk.source_id,
                            "source_url": chunk.source_url,
                            "scheme": chunk.scheme,
                            "page_type": chunk.page_type,
                            "section_heading": chunk.section_heading,
                            "chunk_index": chunk.chunk_index,
                            "scraped_date": chunk.scraped_date,
                            "text": chunk.text,
                            "token_estimate": chunk.token_estimate,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        summary = {
            "generated_at": report.generated_at,
            "expected_source_count": report.expected_source_count,
            "source_count": report.source_count,
            "chunk_count": len(report.chunks),
            "per_source": per_source,
            "chunk_size": default_settings.chunk_size,
            "chunk_overlap": default_settings.chunk_overlap,
            "chunks_file": report.chunks_path,
            "skipped": report.skipped,
            "warnings": report.warnings,
        }
        Path(report.summary_path).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def chunk_documents(
    cleaned_dir: Optional[Path] = None,
    chunks_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    sources: Optional[Sequence[Source]] = None,
) -> ChunkReport:
    return DocumentChunker(
        cleaned_dir=cleaned_dir,
        chunks_dir=chunks_dir,
        raw_dir=raw_dir,
    ).chunk(sources)
