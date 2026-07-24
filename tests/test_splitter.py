"""Unit tests for rag.splitter."""

from langchain_core.documents import Document

from rag.splitter import split_documents


def test_split_documents_preserves_metadata():
    docs = [
        Document(page_content="A" * 500, metadata={"source": "paper.pdf", "page": 1}),
        Document(page_content="B" * 500, metadata={"source": "paper.pdf", "page": 2}),
    ]

    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.metadata["source"] == "paper.pdf"
        assert chunk.metadata["page"] in (1, 2)
        assert "chunk_id" in chunk.metadata


def test_split_documents_chunk_ids_are_unique():
    docs = [Document(page_content="X" * 1000, metadata={"source": "a.pdf", "page": 1})]

    chunks = split_documents(docs, chunk_size=100, chunk_overlap=0)
    ids = [c.metadata["chunk_id"] for c in chunks]

    assert len(ids) == len(set(ids))


def test_split_documents_empty_input_returns_empty_list():
    assert split_documents([]) == []
