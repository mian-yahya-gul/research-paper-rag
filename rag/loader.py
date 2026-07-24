"""PDF loading utilities.

Wraps LangChain's PyPDFLoader and enriches each page's metadata so that
downstream nodes can cite the originating file and page number.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """Load a single PDF into one LangChain Document per page.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        A list of Document objects, one per page, with `source` (the
        original filename) and `page` (1-indexed) metadata set.

    Raises:
        FileNotFoundError: If the given path does not exist.
        ValueError: If the file is not a PDF.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    filename = path.name
    for page in pages:
        # PyPDFLoader sets a 0-indexed "page" key; normalize to 1-indexed
        # for human-friendly citations and pin down the display filename.
        raw_page = page.metadata.get("page", 0)
        page.metadata["source"] = filename
        page.metadata["page"] = int(raw_page) + 1

    return pages


def load_pdfs(file_paths: list[str]) -> list[Document]:
    """Load multiple PDFs and return their combined pages.

    Files that fail to load are skipped with the error surfaced to the
    caller via a warning message rather than aborting the whole batch.
    """
    all_documents: list[Document] = []
    for file_path in file_paths:
        try:
            all_documents.extend(load_pdf(file_path))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[loader] Skipping '{file_path}': {exc}")
    return all_documents


def save_uploaded_file(file_bytes: bytes, filename: str, uploads_dir: str) -> str:
    """Persist an in-memory uploaded file to disk and return its path."""
    os.makedirs(uploads_dir, exist_ok=True)
    destination = Path(uploads_dir) / filename
    with open(destination, "wb") as f:
        f.write(file_bytes)
    return str(destination)
