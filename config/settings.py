"""Centralized application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_int(key: str, default: int) -> int:
    value = os.getenv(key)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved once at import time."""

    openai_api_key: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    chroma_persist_dir: str
    chroma_collection_name: str = "research_papers"
    uploads_dir: str = "data/uploads"

    def validate(self) -> None:
        """Raise a clear error if required configuration is missing."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )


def load_settings() -> Settings:
    """Build a Settings instance from the current environment."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        retrieval_k=_get_int("RETRIEVAL_K", 5),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "data/vectorstore"),
    )


settings = load_settings()
