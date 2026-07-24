"""Embedding model factory."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings client.

    Cached so repeated calls across Streamlit reruns reuse a single
    client instance instead of reinitializing on every interaction.
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
