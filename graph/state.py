"""Shared state schema for the RAG LangGraph workflow."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage

from rag.retriever import RetrievedChunk
from utils.citation_utils import Citation


class GraphState(TypedDict):
    """State passed between nodes in the RAG graph.

    Attributes:
        question: The current user question.
        chat_history: Prior conversation turns as LangChain messages.
        retrieved_chunks: Chunks returned by the retrieval node.
        answer: The final generated answer text.
        citations: De-duplicated source citations for display.
    """

    question: str
    chat_history: list[BaseMessage]
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    citations: list[Citation]
