"""Streamlit interface for the Research Paper RAG Assistant."""

from __future__ import annotations

import streamlit as st
from langchain_openai import ChatOpenAI

from config.settings import settings
from graph.graph import build_rag_graph
from rag.loader import load_pdf, save_uploaded_file
from rag.splitter import split_documents
from rag.vectorstore import add_chunks, clear_vectorstore, get_vectorstore, list_indexed_sources
from utils.history_utils import to_langchain_messages

st.set_page_config(page_title="Research Paper RAG Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_vectorstore():
    return get_vectorstore()


@st.cache_resource(show_spinner=False)
def _load_llm():
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


@st.cache_resource(show_spinner=False)
def _load_graph():
    return build_rag_graph(
        vectorstore=_load_vectorstore(),
        llm=_load_llm(),
        retrieval_k=settings.retrieval_k,
    )


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "citations"}


def _process_uploaded_files(uploaded_files) -> None:
    vectorstore = _load_vectorstore()
    new_chunks_total = 0

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(
                uploaded_file.getvalue(), uploaded_file.name, settings.uploads_dir
            )
            pages = load_pdf(file_path)
            chunks = split_documents(
                pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
            )
            new_chunks_total += add_chunks(vectorstore, chunks)

    if new_chunks_total:
        st.success(f"Indexed {new_chunks_total} new chunks.")
    else:
        st.info("No new content to index (files already processed).")


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("📄 Documents")

        uploaded_files = st.file_uploader(
            "Upload research papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Process Documents", use_container_width=True):
            _process_uploaded_files(uploaded_files)
            st.rerun()

        st.divider()

        sources = list_indexed_sources(_load_vectorstore())
        st.subheader(f"Indexed Papers ({len(sources)})")
        if sources:
            for source in sources:
                st.markdown(f"- {source}")
        else:
            st.caption("No documents indexed yet.")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Clear Documents", use_container_width=True):
                clear_vectorstore(_load_vectorstore())
                st.rerun()


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            citations = message.get("citations") or []
            if citations:
                with st.expander("Sources"):
                    for citation in citations:
                        st.markdown(f"**{citation.source}, p. {citation.page}**")
                        st.caption(citation.preview)


def _handle_user_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question, "citations": []})
    with st.chat_message("user"):
        st.markdown(question)

    history_turns = [
        (m["role"] if m["role"] == "user" else "assistant", m["content"])
        for m in st.session_state.messages[:-1]
    ]
    chat_history = to_langchain_messages(history_turns)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            graph = _load_graph()
            result = graph.invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "retrieved_chunks": [],
                    "answer": "",
                    "citations": [],
                }
            )
        st.markdown(result["answer"])
        citations = result["citations"]
        if citations:
            with st.expander("Sources"):
                for citation in citations:
                    st.markdown(f"**{citation.source}, p. {citation.page}**")
                    st.caption(citation.preview)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "citations": citations}
    )


def main() -> None:
    try:
        settings.validate()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    _init_session_state()

    st.title("📄 Research Paper RAG Assistant")
    st.caption(
        "Upload research papers and ask questions. Answers are grounded only in "
        "the uploaded documents and always cite the source page."
    )

    _render_sidebar()
    _render_chat_history()

    question = st.chat_input("Ask a question about your uploaded papers...")
    if question:
        _handle_user_question(question)


if __name__ == "__main__":
    main()
