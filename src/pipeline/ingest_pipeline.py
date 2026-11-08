from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import settings
from src.ingest.chunker import ChunkError, DocumentChunker
from src.ingest.cleaner import CleanError, DocumentCleaner
from src.ingest.embedder import ChunkEmbedder, EmbedError
from src.ingest.fetcher import Fetcher
from src.ingest.loader import DocumentLoader, LoadError
from src.ingest.registry import get_sources
from src.store.chroma_store import ChromaStore, ChromaStoreError


LOGGER = logging.getLogger(__name__)


def run_scrape(raw_dir: Optional[Path] = None) -> int:
    sources = get_sources()
    fetcher = Fetcher(settings=settings, raw_dir=raw_dir)
    try:
        report = fetcher.scrape(sources)
    finally:
        fetcher.close()

    print(f"Scrape manifest: {report.manifest_path}")
    print(f"Documents scraped: {len(report.documents)}/{len(sources)}")
    for document in report.documents:
        print(
            f"  {document.source_id}: HTTP {document.status_code}, "
            f"{document.content_length} bytes"
        )
    for error in report.errors:
        print(f"  ERROR {error['source_id']}: {error['error']}", file=sys.stderr)

    if report.errors or len(report.documents) != len(sources):
        return 1
    return 0


def run_load(raw_dir: Optional[Path] = None) -> int:
    try:
        report = DocumentLoader(raw_dir=raw_dir).load()
    except LoadError as exc:
        print(f"Load error: {exc}", file=sys.stderr)
        return 1

    print(f"Load report: {report.report_path}")
    print(f"Documents loaded: {len(report.documents)}/{report.expected_count}")
    for document in report.documents:
        print(
            f"  {document.source_id}: {document.content_length} bytes, "
            f"checksum {document.checksum[:12]}"
        )
    for skipped in report.skipped:
        print(f"  SKIPPED {skipped['source_id']}: {skipped['reason']}", file=sys.stderr)
    for warning in report.warnings:
        print(f"  WARNING: {warning}", file=sys.stderr)

    return 0 if report.ok else 1


def run_clean(
    raw_dir: Optional[Path] = None,
    cleaned_dir: Optional[Path] = None,
) -> int:
    try:
        report = DocumentCleaner(raw_dir=raw_dir, cleaned_dir=cleaned_dir).clean()
    except (LoadError, CleanError) as exc:
        print(f"Clean error: {exc}", file=sys.stderr)
        return 1

    print(f"Clean report: {report.report_path}")
    print(f"Documents cleaned: {len(report.documents)}/{report.expected_count}")
    for document in report.documents:
        print(
            f"  {document.source_id}: {document.character_count} characters, "
            f"{document.line_count} lines"
        )
    for skipped in report.skipped:
        print(f"  SKIPPED {skipped['source_id']}: {skipped['reason']}", file=sys.stderr)
    for warning in report.warnings:
        print(f"  WARNING: {warning}", file=sys.stderr)

    return 0 if report.ok else 1


def run_chunk(
    cleaned_dir: Optional[Path] = None,
    chunks_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> int:
    try:
        report = DocumentChunker(
            cleaned_dir=cleaned_dir,
            chunks_dir=chunks_dir,
            raw_dir=raw_dir,
        ).chunk()
    except ChunkError as exc:
        print(f"Chunk error: {exc}", file=sys.stderr)
        return 1

    print(f"Chunks file: {report.chunks_path}")
    print(f"Chunk summary: {report.summary_path}")
    print(
        f"Sources chunked: {report.source_count}/{report.expected_source_count}, "
        f"chunks: {len(report.chunks)}"
    )
    for skipped in report.skipped:
        print(f"  SKIPPED {skipped['source_id']}: {skipped['reason']}", file=sys.stderr)
    for warning in report.warnings:
        print(f"  WARNING: {warning}", file=sys.stderr)

    return 0 if report.ok else 1


def run_embed(
    chunks_dir: Optional[Path] = None,
    embeddings_dir: Optional[Path] = None,
) -> int:
    try:
        report = ChunkEmbedder(
            chunks_dir=chunks_dir,
            embeddings_dir=embeddings_dir,
        ).embed()
    except EmbedError as exc:
        print(f"Embed error: {exc}", file=sys.stderr)
        return 1

    print(f"Embeddings: {report.embeddings_path}")
    print(f"Embedding IDs: {report.ids_path}")
    print(f"Embedding manifest: {report.manifest_path}")
    print(f"Model: {report.model_name}")
    print(f"Chunks embedded: {report.chunk_count}")
    print(f"Dimension: {report.dimension}")

    return 0 if report.ok else 1


@dataclass
class StoreReport:
    collection_name: str
    dimension: int
    chunk_count: int
    collection_count: int
    scraped_date: str
    manifest_path: str
    pruned_count: int

    @property
    def ok(self) -> bool:
        return self.collection_count == self.chunk_count


def run_store(
    chunks_dir: Optional[Path] = None,
    embeddings_dir: Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
) -> int:
    try:
        report = _store_vectors(chunks_dir, embeddings_dir, chroma_dir)
    except (ChromaStoreError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Store error: {exc}", file=sys.stderr)
        return 1

    print(f"Collection: {report.collection_name}")
    print(f"Collection count: {report.collection_count}/{report.chunk_count}")
    print(f"Dimension: {report.dimension}")
    print(f"Scraped date: {report.scraped_date}")
    print(f"Stale vectors removed: {report.pruned_count}")
    print(f"Ingestion manifest: {report.manifest_path}")

    return 0 if report.ok else 1


def _store_vectors(
    chunks_dir: Optional[Path],
    embeddings_dir: Optional[Path],
    chroma_dir: Optional[Path],
) -> StoreReport:
    chunks_path = (
        Path(chunks_dir) if chunks_dir is not None else Path("data/chunks")
    ) / "chunks.jsonl"
    embeddings_root = (
        Path(embeddings_dir) if embeddings_dir is not None else Path("data/embeddings")
    )
    embeddings_path = embeddings_root / "embeddings.npy"
    ids_path = embeddings_root / "embedding_ids.json"
    embedding_manifest_path = embeddings_root / "embedding_manifest.json"
    persist_dir = (
        Path(chroma_dir) if chroma_dir is not None else Path(settings.chroma_dir)
    )

    chunks = _read_chunks_for_store(chunks_path)
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    embeddings, embedding_ids, embedding_manifest = _load_embeddings(
        embeddings_path,
        ids_path,
        embedding_manifest_path,
        chunk_ids,
    )

    if embedding_manifest.get("model") != settings.embedding_model:
        raise ChromaStoreError(
            "Embedding model mismatch: manifest has "
            f"{embedding_manifest.get('model')!r}, config expects "
            f"{settings.embedding_model!r}"
        )

    metadatas = [_chunk_metadata(chunk) for chunk in chunks]
    documents = [str(chunk["text"]) for chunk in chunks]
    store = ChromaStore(persist_dir=persist_dir)
    store.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    pruned_count = store.prune(chunk_ids)
    collection_count = store.count()
    if collection_count != len(chunk_ids):
        raise ChromaStoreError(
            f"Collection count {collection_count} does not match "
            f"chunk count {len(chunk_ids)}"
        )

    stored_metadatas = store.get_metadatas()
    if len(stored_metadatas) != len(chunk_ids):
        raise ChromaStoreError(
            "Stored metadata count does not match chunk count"
        )
    for metadata in stored_metadatas:
        for key in ("chunk_id", "source_url", "scheme"):
            value = metadata.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ChromaStoreError(f"Stored record is missing metadata: {key}")

    scraped_date = max(str(chunk["scraped_date"]) for chunk in chunks)
    generated_at = datetime.now(timezone.utc).isoformat()
    persist_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = persist_dir / "ingestion_manifest.json"
    ingestion_manifest = {
        "collection_name": store.collection_name,
        "embedding_model": embedding_manifest["model"],
        "dimension": int(embeddings.shape[1]),
        "chunk_count": len(chunk_ids),
        "ingestion_timestamp": generated_at,
        "generated_at": generated_at,
        "scraped_date": scraped_date,
        "vector_db": settings.vector_db,
        "persist_dir": str(persist_dir),
        "pruned_count": pruned_count,
        "chunks_file": str(chunks_path),
        "embeddings_file": str(embeddings_path),
    }
    manifest_path.write_text(
        json.dumps(ingestion_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return StoreReport(
        collection_name=store.collection_name,
        dimension=int(embeddings.shape[1]),
        chunk_count=len(chunk_ids),
        collection_count=collection_count,
        scraped_date=scraped_date,
        manifest_path=str(manifest_path),
        pruned_count=pruned_count,
    )


def _read_chunks_for_store(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise ChromaStoreError(f"Chunks file not found: {path}")
    chunks: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChromaStoreError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(chunk, dict):
                raise ChromaStoreError(
                    f"Chunk at line {line_number} must be a JSON object"
                )
            required = (
                "chunk_id",
                "source_url",
                "scheme",
                "page_type",
                "section_heading",
                "chunk_index",
                "scraped_date",
                "text",
            )
            missing = [
                key
                for key in required
                if key not in chunk
                or chunk[key] is None
                or (isinstance(chunk[key], str) and not chunk[key].strip())
            ]
            if missing:
                raise ChromaStoreError(
                    f"Chunk at line {line_number} is missing: {', '.join(missing)}"
                )
            if not isinstance(chunk["chunk_index"], int):
                raise ChromaStoreError(
                    f"Chunk {chunk['chunk_id']} has non-integer chunk_index"
                )
            chunks.append(chunk)
    if not chunks:
        raise ChromaStoreError(f"No chunks found in {path}")
    return chunks


def _load_embeddings(
    embeddings_path: Path,
    ids_path: Path,
    manifest_path: Path,
    chunk_ids: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    for path in (embeddings_path, ids_path, manifest_path):
        if not path.exists():
            raise ChromaStoreError(f"Embedding artifact not found: {path}")

    embeddings = np.load(embeddings_path, allow_pickle=False)
    embedding_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(embedding_ids, list) or not all(
        isinstance(item, str) for item in embedding_ids
    ):
        raise ChromaStoreError("embedding_ids.json must contain a list of strings")
    if not isinstance(manifest, dict):
        raise ChromaStoreError("embedding_manifest.json must contain an object")
    if embeddings.ndim != 2:
        raise ChromaStoreError(
            f"Embeddings must be 2-dimensional, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != len(chunk_ids):
        raise ChromaStoreError(
            f"Embedding count {embeddings.shape[0]} does not match "
            f"chunk count {len(chunk_ids)}"
        )
    if embedding_ids != chunk_ids:
        raise ChromaStoreError("Embedding IDs do not match chunk IDs in order")
    if manifest.get("count") != len(chunk_ids):
        raise ChromaStoreError("Embedding manifest count does not match chunks")
    if manifest.get("dimension") != embeddings.shape[1]:
        raise ChromaStoreError("Embedding manifest dimension does not match vectors")
    if not np.isfinite(embeddings).all():
        raise ChromaStoreError("Embeddings contain NaN or infinite values")
    return embeddings, embedding_ids, manifest


def _chunk_metadata(chunk: Dict[str, object]) -> Dict[str, Any]:
    return {
        "chunk_id": str(chunk["chunk_id"]),
        "source_id": str(chunk.get("source_id", "")),
        "source_url": str(chunk["source_url"]),
        "scheme": str(chunk["scheme"]),
        "page_type": str(chunk["page_type"]),
        "section_heading": str(chunk["section_heading"]),
        "chunk_index": int(chunk["chunk_index"]),
        "scraped_date": str(chunk["scraped_date"]),
        "token_estimate": int(chunk.get("token_estimate", 0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Groww mutual fund FAQ ingest pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["scrape", "load", "clean", "chunk", "embed", "store"],
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--cleaned-dir", type=Path, default=None)
    parser.add_argument("--chunks-dir", type=Path, default=None)
    parser.add_argument("--embeddings-dir", type=Path, default=None)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.stage == "scrape":
        return run_scrape(args.raw_dir)
    if args.stage == "load":
        return run_load(args.raw_dir)
    if args.stage == "clean":
        return run_clean(args.raw_dir, args.cleaned_dir)
    if args.stage == "chunk":
        return run_chunk(args.cleaned_dir, args.chunks_dir, args.raw_dir)
    if args.stage == "embed":
        return run_embed(args.chunks_dir, args.embeddings_dir)
    if args.stage == "store":
        return run_store(args.chunks_dir, args.embeddings_dir, args.chroma_dir)
    LOGGER.error("Unsupported stage: %s", args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
