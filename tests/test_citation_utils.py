"""Unit tests for utils.citation_utils."""

from langchain_core.documents import Document

from rag.retriever import RetrievedChunk
from utils.citation_utils import build_citations, build_context_block


def _make_chunk(source: str, page: int, text: str, score: float = 0.1) -> RetrievedChunk:
    doc = Document(page_content=text, metadata={"source": source, "page": page})
    return RetrievedChunk(document=doc, score=score)


def test_build_context_block_includes_source_and_page():
    chunks = [_make_chunk("paper.pdf", 3, "Transformers use self-attention.")]
    context = build_context_block(chunks)

    assert "paper.pdf" in context
    assert "p. 3" in context
    assert "self-attention" in context


def test_build_citations_deduplicates_same_source_and_page():
    chunks = [
        _make_chunk("paper.pdf", 1, "First excerpt from page one."),
        _make_chunk("paper.pdf", 1, "Second excerpt, same page."),
        _make_chunk("paper.pdf", 2, "Excerpt from page two."),
    ]

    citations = build_citations(chunks)

    assert len(citations) == 2
    assert (citations[0].source, citations[0].page) == ("paper.pdf", 1)
    assert (citations[1].source, citations[1].page) == ("paper.pdf", 2)


def test_build_citations_empty_input_returns_empty_list():
    assert build_citations([]) == []
