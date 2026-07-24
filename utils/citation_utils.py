"""Helpers for formatting retrieved chunks into context and citations."""

from __future__ import annotations

from dataclasses import dataclass

from rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    """A de-duplicated (source, page) reference shown to the user."""

    source: str
    page: int
    preview: str


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single context string for the prompt.

    Each excerpt is labeled with its source and page so the LLM can
    reproduce that information in its inline citations.
    """
    blocks = []
    for chunk in chunks:
        header = f"[Source: {chunk.source}, p. {chunk.page}]"
        blocks.append(f"{header}\n{chunk.document.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_citations(chunks: list[RetrievedChunk], preview_chars: int = 160) -> list[Citation]:
    """Collapse retrieved chunks into unique, display-ready citations."""
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []

    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        content = chunk.document.page_content.strip().replace("\n", " ")
        preview = content[:preview_chars] + ("..." if len(content) > preview_chars else "")
        citations.append(Citation(source=chunk.source, page=chunk.page, preview=preview))

    return citations
