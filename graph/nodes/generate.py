"""Generation node: builds the prompt, calls the LLM, and attaches citations."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from graph.state import GraphState
from prompts.rag_prompt import RAG_PROMPT
from utils.citation_utils import build_citations, build_context_block

NO_CONTEXT_ANSWER = (
    "I couldn't find relevant information in the uploaded documents to answer "
    "that question."
)


def make_generate_node(llm: ChatOpenAI):
    """Build a generate node bound to a specific chat model."""

    chain = RAG_PROMPT | llm | StrOutputParser()

    def generate(state: GraphState) -> GraphState:
        chunks = state["retrieved_chunks"]

        if not chunks:
            return {**state, "answer": NO_CONTEXT_ANSWER, "citations": []}

        context = build_context_block(chunks)
        answer = chain.invoke(
            {
                "context": context,
                "question": state["question"],
                "chat_history": state["chat_history"],
            }
        )
        citations = build_citations(chunks)

        return {**state, "answer": answer, "citations": citations}

    return generate
