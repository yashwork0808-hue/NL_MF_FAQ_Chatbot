from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.query.retriever import RetrievalResult, RetrievedChunk


SYSTEM_PROMPT = (
    "You are a facts-only mutual fund FAQ assistant for Groww. "
    "Use only the provided context. Do not use outside knowledge. "
    "Answer in at most three sentences. "
    "Include exactly one source URL, copied verbatim from the context. "
    "Do not provide investment advice, recommendations, opinions, or portfolio guidance. "
    "Do not calculate, compare, or claim investment returns or performance. "
    "If the answer is not present in the context, say the information is not available "
    "on the indexed pages. "
    "End with a final line exactly like: Last updated from sources: <date>."
)


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    context_chunks: List[RetrievedChunk]
    citation_url: str
    source_date: str


class PromptBuilder:
    def build(self, retrieval: RetrievalResult, question: str) -> PromptBundle:
        if retrieval.citation is None:
            raise ValueError("Retrieval result has no citation")
        context_parts = []
        for index, chunk in enumerate(retrieval.chunks, start=1):
            metadata = chunk.metadata
            context_parts.append(
                "\n".join(
                    (
                        f"[Context {index}]",
                        f"Source URL: {metadata.get('source_url', '')}",
                        f"Scheme: {metadata.get('scheme', '')}",
                        f"Page type: {metadata.get('page_type', '')}",
                        f"Section: {metadata.get('section_heading', '')}",
                        f"Updated: {metadata.get('scraped_date', retrieval.source_date)}",
                        "Text:",
                        chunk.text,
                    )
                )
            )
        context = "\n\n".join(context_parts)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Allowed source URL: {retrieval.citation.metadata['source_url']}\n"
            f"Source date: {retrieval.source_date}\n\n"
            f"Context:\n{context}\n\n"
            "Answer using only the context above. Include exactly one source URL and "
            f"end with: Last updated from sources: {retrieval.source_date}"
        )
        return PromptBundle(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context_chunks=list(retrieval.chunks),
            citation_url=str(retrieval.citation.metadata["source_url"]),
            source_date=retrieval.source_date,
        )


def build_prompt(retrieval: RetrievalResult, question: str) -> PromptBundle:
    return PromptBuilder().build(retrieval, question)
