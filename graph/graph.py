"""Assembles the RAG workflow as a compiled LangGraph graph.

Flow:
    START -> retrieve -> generate -> END
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes.generate import make_generate_node
from graph.nodes.retrieve import make_retrieve_node
from graph.state import GraphState


def build_rag_graph(
    vectorstore: Chroma,
    llm: ChatOpenAI,
    retrieval_k: int = 5,
) -> CompiledStateGraph:
    """Construct and compile the two-node RAG graph."""
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", make_retrieve_node(vectorstore, k=retrieval_k))
    graph.add_node("generate", make_generate_node(llm))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
