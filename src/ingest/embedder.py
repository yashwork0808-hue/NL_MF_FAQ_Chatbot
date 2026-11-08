from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Settings, settings as default_settings


LOGGER = logging.getLogger(__name__)
DEFAULT_BATCH_SIZE = 32


class EmbedError(RuntimeError):
    pass


@dataclass
class EmbedReport:
    generated_at: str
    model_name: str
    dimension: int
    chunk_count: int
    embeddings_path: str
    ids_path: str
    manifest_path: str

    @property
    def ok(self) -> bool:
        return self.chunk_count > 0 and self.dimension > 0


class ChunkEmbedder:
    def __init__(
        self,
        chunks_dir: Optional[Path] = None,
        embeddings_dir: Optional[Path] = None,
        settings: Settings = default_settings,
    ) -> None:
        self.chunks_dir = (
            Path(chunks_dir) if chunks_dir is not None else Path("data/chunks")
        )
        self.embeddings_dir = (
            Path(embeddings_dir)
            if embeddings_dir is not None
            else Path("data/embeddings")
        )
        self.settings = settings

    def embed(
        self,
        chunks_path: Optional[Path] = None,
        model: Optional[SentenceTransformer] = None,
    ) -> EmbedReport:
        source_path = (
            Path(chunks_path)
            if chunks_path is not None
            else self.chunks_dir / "chunks.jsonl"
        )
        chunks = self._read_chunks(source_path)
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]

        model_name = self.settings.embedding_model
        if model is None:
            LOGGER.info("Loading embedding model: %s", model_name)
            try:
                model = SentenceTransformer(model_name)
            except Exception as exc:
                raise EmbedError(
                    f"Unable to load embedding model {model_name}: {exc}"
                ) from exc

        LOGGER.info("Embedding %d chunks", len(texts))
        try:
            embeddings = model.encode(
                texts,
                batch_size=DEFAULT_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbedError(f"Unable to embed chunks: {exc}") from exc

        embeddings = np.asarray(embeddings)
        self._validate_embeddings(embeddings, chunk_ids)

        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = self.embeddings_dir / "embeddings.npy"
        ids_path = self.embeddings_dir / "embedding_ids.json"
        manifest_path = self.embeddings_dir / "embedding_manifest.json"

        np.save(embeddings_path, embeddings, allow_pickle=False)
        ids_path.write_text(
            json.dumps(chunk_ids, indent=2) + "\n",
            encoding="utf-8",
        )

        generated_at = datetime.now(timezone.utc).isoformat()
        manifest: Dict[str, object] = {
            "generated_at": generated_at,
            "model": model_name,
            "dimension": int(embeddings.shape[1]),
            "count": len(chunk_ids),
            "normalized": True,
            "batch_size": DEFAULT_BATCH_SIZE,
            "chunks_file": str(source_path),
            "embeddings_file": str(embeddings_path),
            "ids_file": str(ids_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        report = EmbedReport(
            generated_at=generated_at,
            model_name=model_name,
            dimension=int(embeddings.shape[1]),
            chunk_count=len(chunk_ids),
            embeddings_path=str(embeddings_path),
            ids_path=str(ids_path),
            manifest_path=str(manifest_path),
        )
        self._validate_saved_outputs(report)
        return report

    def _read_chunks(self, chunks_path: Path) -> List[Dict[str, object]]:
        if not chunks_path.exists():
            raise EmbedError(f"Chunks file not found: {chunks_path}")

        chunks: List[Dict[str, object]] = []
        seen_ids = set()
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EmbedError(
                        f"Invalid JSON in {chunks_path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(chunk, dict):
                    raise EmbedError(
                        f"Chunk at line {line_number} must be a JSON object"
                    )
                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise EmbedError(f"Chunk at line {line_number} has no chunk_id")
                if not isinstance(text, str) or not text.strip():
                    raise EmbedError(f"Chunk {chunk_id} has empty text")
                if chunk_id in seen_ids:
                    raise EmbedError(f"Duplicate chunk_id: {chunk_id}")
                seen_ids.add(chunk_id)
                chunks.append(chunk)

        if not chunks:
            raise EmbedError(f"No chunks found in {chunks_path}")
        return chunks

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> None:
        if embeddings.ndim != 2:
            raise EmbedError(
                f"Embeddings must be 2-dimensional, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != len(chunk_ids):
            raise EmbedError(
                f"Embedding count {embeddings.shape[0]} does not match "
                f"chunk count {len(chunk_ids)}"
            )
        if embeddings.shape[1] <= 0:
            raise EmbedError("Embedding dimension must be greater than zero")
        if not np.issubdtype(embeddings.dtype, np.floating):
            raise EmbedError(f"Embeddings must be floating point, got {embeddings.dtype}")
        if not np.isfinite(embeddings).all():
            raise EmbedError("Embeddings contain NaN or infinite values")

    def _validate_saved_outputs(self, report: EmbedReport) -> None:
        embeddings = np.load(report.embeddings_path, allow_pickle=False)
        chunk_ids = json.loads(Path(report.ids_path).read_text(encoding="utf-8"))
        manifest = json.loads(
            Path(report.manifest_path).read_text(encoding="utf-8")
        )
        if embeddings.shape[0] != len(chunk_ids):
            raise EmbedError("Saved embedding count does not match saved IDs")
        if embeddings.shape[1] != manifest["dimension"]:
            raise EmbedError("Saved embedding dimension does not match manifest")
        if manifest["count"] != len(chunk_ids):
            raise EmbedError("Saved manifest count does not match saved IDs")


def embed_chunks(
    chunks_dir: Optional[Path] = None,
    embeddings_dir: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
    settings: Settings = default_settings,
) -> EmbedReport:
    return ChunkEmbedder(
        chunks_dir=chunks_dir,
        embeddings_dir=embeddings_dir,
        settings=settings,
    ).embed(chunks_path=chunks_path)
