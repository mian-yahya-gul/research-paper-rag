"""Retrieval node: fetches relevant chunks for the current question."""

from __future__ import annotations

from langchain_chroma import Chroma

from graph.state import GraphState
from rag.retriever import retrieve_relevant_chunks


def make_retrieve_node(vectorstore: Chroma, k: int):
    """Build a retrieve node bound to a specific vector store and top-k."""

    def retrieve(state: GraphState) -> GraphState:
        chunks = retrieve_relevant_chunks(vectorstore, state["question"], k=k)
        return {**state, "retrieved_chunks": chunks}

    return retrieve
