"""Chunking utilities built on RecursiveCharacterTextSplitter."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split page-level documents into retrieval-sized chunks.

    Each resulting chunk inherits the source page's metadata (source
    filename, page number) and gains a stable `chunk_id` derived from
    its content and position, which is used later to avoid duplicate
    inserts into the vector store.

    Args:
        documents: Page-level Document objects, typically from `rag.loader`.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A flat list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        chunk.metadata["chunk_id"] = f"{source}-p{page}-{index}-{digest}"

    return chunks
