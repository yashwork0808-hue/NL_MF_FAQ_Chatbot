from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.errors import NotFoundError

from src.config import Settings, settings as default_settings
from src.store.base import VectorStore


LOGGER = logging.getLogger(__name__)


class ChromaStoreError(RuntimeError):
    pass


class ChromaStore(VectorStore):
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        settings: Settings = default_settings,
    ) -> None:
        if settings.vector_db != "chroma":
            raise ChromaStoreError(
                f"VECTOR_DB must be 'chroma', got {settings.vector_db!r}"
            )
        self.settings = settings
        self.persist_dir = (
            Path(persist_dir) if persist_dir is not None else Path(settings.chroma_dir)
        )
        self.collection_name = collection_name or settings.collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        except Exception as exc:
            raise ChromaStoreError(
                f"Unable to open ChromaDB at {self.persist_dir}: {exc}"
            ) from exc
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise ChromaStoreError(
                f"Unable to open collection {self.collection_name!r}: {exc}"
            ) from exc

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        id_list = list(ids)
        embedding_list = [
            [float(value) for value in embedding] for embedding in embeddings
        ]
        document_list = list(documents)
        metadata_list = list(metadatas)

        lengths = {
            len(id_list),
            len(embedding_list),
            len(document_list),
            len(metadata_list),
        }
        if len(lengths) != 1:
            raise ChromaStoreError("IDs, embeddings, documents, and metadata differ in length")
        if not id_list:
            raise ChromaStoreError("Cannot upsert an empty batch")
        if len(set(id_list)) != len(id_list):
            raise ChromaStoreError("Batch contains duplicate IDs")

        dimension = len(embedding_list[0])
        if dimension <= 0:
            raise ChromaStoreError("Embedding dimension must be greater than zero")
        if any(len(embedding) != dimension for embedding in embedding_list):
            raise ChromaStoreError("Batch contains inconsistent embedding dimensions")

        for chunk_id, document, metadata in zip(
            id_list, document_list, metadata_list
        ):
            if not document.strip():
                raise ChromaStoreError(f"Chunk {chunk_id} has empty text")
            for key in ("chunk_id", "source_url", "scheme"):
                value = metadata.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ChromaStoreError(
                        f"Chunk {chunk_id} has missing metadata: {key}"
                    )
            if metadata["chunk_id"] != chunk_id:
                raise ChromaStoreError(
                    f"Metadata chunk_id does not match record ID {chunk_id}"
                )

        self._validate_collection_dimension(dimension)

        try:
            self.collection.upsert(
                ids=id_list,
                embeddings=embedding_list,
                documents=document_list,
                metadatas=metadata_list,
            )
        except Exception as exc:
            raise ChromaStoreError(f"Unable to upsert vectors: {exc}") from exc

    def _validate_collection_dimension(self, dimension: int) -> None:
        if self.count() == 0:
            return
        try:
            existing = self.collection.get(limit=1, include=["embeddings"])
        except Exception as exc:
            raise ChromaStoreError(
                f"Unable to inspect existing collection dimension: {exc}"
            ) from exc
        existing_embeddings = existing.get("embeddings")
        if existing_embeddings is None or len(existing_embeddings) == 0:
            raise ChromaStoreError(
                "Existing collection has records but no readable embeddings"
            )
        existing_dimension = len(existing_embeddings[0])
        if existing_dimension != dimension:
            raise ChromaStoreError(
                f"Embedding dimension mismatch: collection has {existing_dimension}, "
                f"new vectors have {dimension}. Reset the collection before re-ingesting."
            )

    def prune(self, valid_ids: Sequence[str]) -> int:
        valid = set(valid_ids)
        try:
            existing = self.collection.get(include=[])
        except Exception as exc:
            raise ChromaStoreError(f"Unable to list collection IDs: {exc}") from exc
        stale = sorted(set(existing.get("ids", [])) - valid)
        if stale:
            try:
                self.collection.delete(ids=stale)
            except Exception as exc:
                raise ChromaStoreError(f"Unable to delete stale IDs: {exc}") from exc
            LOGGER.info("Deleted %d stale vector(s)", len(stale))
        return len(stale)

    def get_metadatas(self) -> List[Dict[str, Any]]:
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as exc:
            raise ChromaStoreError(f"Unable to read collection metadata: {exc}") from exc
        return result.get("metadatas") or []

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            return self.collection.query(
                query_embeddings=[
                    [float(value) for value in embedding]
                    for embedding in query_embeddings
                ],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ChromaStoreError(f"Unable to query collection: {exc}") from exc

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:
            raise ChromaStoreError(f"Unable to count collection: {exc}") from exc

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except NotFoundError:
            pass
        except Exception as exc:
            raise ChromaStoreError(
                f"Unable to reset collection {self.collection_name!r}: {exc}"
            ) from exc
        self.collection = self._get_or_create_collection()
