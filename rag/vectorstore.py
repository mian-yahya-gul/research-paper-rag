"""Chroma vector store management: creation, persistence, and inserts."""

from __future__ import annotations

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from rag.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    """Return the persistent Chroma collection, creating it if needed."""
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_chunks(vectorstore: Chroma, chunks: list[Document]) -> int:
    """Add chunks to the vector store, skipping ones already indexed.

    Uses each chunk's `chunk_id` metadata as the Chroma document id so
    re-uploading the same paper is a no-op rather than a duplicate.

    Returns:
        The number of newly inserted chunks.
    """
    if not chunks:
        return 0

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    existing = vectorstore.get(ids=ids)
    existing_ids = set(existing.get("ids", []))

    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
    new_ids = [c.metadata["chunk_id"] for c in new_chunks]

    if new_chunks:
        vectorstore.add_documents(documents=new_chunks, ids=new_ids)

    return len(new_chunks)


def list_indexed_sources(vectorstore: Chroma) -> list[str]:
    """Return the distinct source filenames currently in the collection."""
    record = vectorstore.get(include=["metadatas"])
    sources = {meta.get("source", "unknown") for meta in record.get("metadatas", [])}
    return sorted(sources)


def clear_vectorstore(vectorstore: Chroma) -> None:
    """Delete every document currently stored in the collection."""
    record = vectorstore.get(include=[])
    ids = record.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
