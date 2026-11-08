from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sentence_transformers import SentenceTransformer

from src.config import Settings, settings as default_settings
from src.ingest.registry import Source, get_allowed_urls, get_sources
from src.store.chroma_store import ChromaStore


LOGGER = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    score: float


@dataclass
class RetrievalResult:
    question: str
    chunks: List[RetrievedChunk]
    citation: Optional[RetrievedChunk]
    source_date: str
    filtered_by_scheme: bool = False
    warnings: List[str] = field(default_factory=list)


class Retriever:
    def __init__(
        self,
        settings: Settings = default_settings,
        store: Optional[ChromaStore] = None,
        model: Optional[SentenceTransformer] = None,
        sources: Optional[Sequence[Source]] = None,
    ) -> None:
        self.settings = settings
        self._store = store
        self._model = model
        self.sources = list(sources) if sources is not None else list(get_sources())
        self._source_priority = {
            source.url: index for index, source in enumerate(self.sources)
        }
        self._allowed_urls = get_allowed_urls()

    @property
    def store(self) -> ChromaStore:
        if self._store is None:
            self._store = ChromaStore(
                persist_dir=Path(self.settings.chroma_dir),
                collection_name=self.settings.collection_name,
                settings=self.settings,
            )
        return self._store

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            LOGGER.info("Loading query embedding model: %s", self.settings.embedding_model)
            try:
                self._model = SentenceTransformer(self.settings.embedding_model)
            except Exception as exc:
                raise RetrievalError(
                    f"Unable to load embedding model {self.settings.embedding_model}: {exc}"
                ) from exc
        return self._model

    def retrieve(
        self,
        question: str,
        source: Optional[Source] = None,
    ) -> RetrievalResult:
        if not question.strip():
            raise RetrievalError("Question must not be empty")

        try:
            query_vector = self.model.encode(
                [question],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
        except Exception as exc:
            raise RetrievalError(f"Unable to embed question: {exc}") from exc

        warnings: List[str] = []
        filtered = source is not None
        where = {"scheme": {"$eq": source.scheme}} if source else None
        results = self.store.query(
            [query_vector.tolist()],
            n_results=self.settings.top_k,
            where=where,
        )
        chunks = self._parse_results(results)
        if filtered and not chunks:
            warnings.append(
                f"No chunks matched scheme filter {source.scheme!r}; used unfiltered search"
            )
            results = self.store.query(
                [query_vector.tolist()],
                n_results=self.settings.top_k,
            )
            chunks = self._parse_results(results)
            filtered = False

        if not chunks:
            warnings.append("No relevant chunks were retrieved")
            return RetrievalResult(
                question=question,
                chunks=[],
                citation=None,
                source_date=self.settings.source_date,
                filtered_by_scheme=filtered,
                warnings=warnings,
            )

        citation = self._select_citation(chunks, source)
        if citation is None:
            warnings.append("No retrieved chunk had an approved source URL")
        source_date = self._source_date(citation, chunks)
        return RetrievalResult(
            question=question,
            chunks=chunks,
            citation=citation,
            source_date=source_date,
            filtered_by_scheme=filtered,
            warnings=warnings,
        )

    @staticmethod
    def _parse_results(results: Dict[str, Any]) -> List[RetrievedChunk]:
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: List[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            document = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            if not isinstance(metadata, dict):
                metadata = {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(document),
                    metadata=metadata,
                    distance=distance,
                    score=1.0 - distance,
                )
            )
        return chunks

    def _select_citation(
        self,
        chunks: Sequence[RetrievedChunk],
        source: Optional[Source],
    ) -> Optional[RetrievedChunk]:
        def sort_key(chunk: RetrievedChunk) -> tuple:
            source_url = str(chunk.metadata.get("source_url", ""))
            scheme_match = (
                1
                if source is not None and chunk.metadata.get("scheme") == source.scheme
                else 0
            )
            priority = self._source_priority.get(source_url, len(self._source_priority))
            return (-chunk.score, -scheme_match, priority)

        for chunk in sorted(chunks, key=sort_key):
            source_url = chunk.metadata.get("source_url")
            if isinstance(source_url, str) and source_url in self._allowed_urls:
                return chunk
        return None

    def _source_date(
        self,
        citation: Optional[RetrievedChunk],
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        if citation is not None:
            scraped_date = citation.metadata.get("scraped_date")
            if isinstance(scraped_date, str) and scraped_date:
                return scraped_date
        dates = [
            str(chunk.metadata.get("scraped_date"))
            for chunk in chunks
            if chunk.metadata.get("scraped_date")
        ]
        if dates:
            return max(dates)
        return self.settings.source_date


def retrieve_context(
    question: str,
    source: Optional[Source] = None,
    settings: Settings = default_settings,
) -> RetrievalResult:
    return Retriever(settings=settings).retrieve(question, source=source)
