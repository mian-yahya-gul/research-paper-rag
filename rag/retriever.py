"""Retrieval helpers built on top of the Chroma vector store."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk paired with its similarity score."""

    document: Document
    score: float

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def page(self) -> int:
        return self.document.metadata.get("page", 0)


def retrieve_relevant_chunks(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
) -> list[RetrievedChunk]:
    """Run similarity search and return scored chunks, best first.

    Chroma's `similarity_search_with_score` returns a distance where
    lower is more similar; results are already sorted ascending by
    that distance, so no additional sorting is required here.
    """
    if not query.strip():
        return []

    results = vectorstore.similarity_search_with_score(query, k=k)
    return [RetrievedChunk(document=doc, score=score) for doc, score in results]
