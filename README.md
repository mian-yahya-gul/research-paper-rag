#  Research Paper RAG Assistant

A Retrieval Augmented Generation (RAG) assistant that answers questions about
uploaded research papers using only the content of those papers, with every
claim backed by an inline `[Source: file.pdf, p. X]` citation.

Built with **LangChain**, **LangGraph**, **ChromaDB**, and **Streamlit**.

---

Overview

Upload one or more PDF research papers, then ask questions in a chat
interface. The assistant retrieves the most relevant passages via semantic
search, grounds its answer strictly in those passages, and returns the
supporting page level citations alongside the response.

The project is intentionally scoped to demonstrate a complete, correct RAG
pipeline — loading, chunking, embedding, vector storage, retrieval, prompt
construction, generation, and citation  

## Features

-  Multi file PDF upload with per page text extraction
-  Recursive character chunking with configurable size/overlap
-  OpenAI embeddings persisted in a local Chroma vector store
-  Similarity search retrieval with relevance scores
-  Citation enforcing prompt template (page-level `[Source, p. X]` tags)
-  LangGraph orchestrated pipeline: `retrieve → generate`
-  Multi turn conversation memory (follo up question resolution)
-  Sidebar showing indexed documents, with clear chat / clear-documents controls
-  Idempotent ingestion  re uploading a paper does not duplicate chunks
-  Unit tests for chunking and citation logic

## Architecture

```
                     ┌─────────────────────┐
                     │   Streamlit UI       │
                     │  (app.py)             │
                     └──────────┬───────────┘
                                │
                 ┌──────────────┴───────────────┐
                 │                               │
        ┌────────▼────────┐            ┌─────────▼─────────┐
        │  Ingestion path   │            │   Query path        │
        │  (on upload)      │            │   (on chat input)   │
        └────────┬────────┘            └─────────┬─────────┘
                 │                               │
   ┌─────────────▼─────────────┐        ┌────────▼─────────┐
   │ rag/loader.py               │        │  graph/graph.py    │
   │  PyPDFLoader → pages        │        │  (LangGraph)        │
   └─────────────┬─────────────┘        └────────┬─────────┘
                 │                               │
   ┌─────────────▼─────────────┐                 │
   │ rag/splitter.py              │                 │
   │  RecursiveCharacterSplitter │                 │
   └─────────────┬─────────────┘                 │
                 │                               │
   ┌─────────────▼─────────────┐                 │
   │ rag/embeddings.py            │◄────────────────┤
   │  OpenAIEmbeddings            │                 │
   └─────────────┬─────────────┘                 │
                 │                               │
   ┌─────────────▼───────────────────────────────▼─────────┐
   │              rag/vectorstore.py (Chroma)                 │
   │        persistent collection, dedup by chunk_id           │
   └───────────────────────────────────────────────────────┘
```

### RAG Pipeline (LangGraph)

```
        START
          │
          ▼
 ┌──────────────────┐
 │  retrieve          │   similarity_search_with_score(question, k)
 │  (graph/nodes/     │   → List[RetrievedChunk]
 │   retrieve.py)     │
 └────────┬──────────┘
          │
          ▼
 ┌──────────────────┐
 │  generate          │   build_context_block(chunks)
 │  (graph/nodes/     │   → RAG_PROMPT | ChatOpenAI | StrOutputParser
 │   generate.py)     │   → answer + build_citations(chunks)
 └────────┬──────────┘
          │
          ▼
         END
```

State (`graph/state.py`) is a single `TypedDict` (`GraphState`) carrying the
question, chat history, retrieved chunks, final answer, and citations — no
agent loops, tool calls, or reflection steps, by design.

## Folder Structure

```
research-paper-rag/
├── app.py                     # Streamlit entry point
├── graph/
│   ├── state.py               # GraphState TypedDict
│   ├── graph.py                # Builds & compiles the LangGraph workflow
│   └── nodes/
│       ├── retrieve.py         # Retrieval node
│       └── generate.py         # Generation + citation node
├── rag/
│   ├── loader.py               # PDF loading (PyPDFLoader)
│   ├── splitter.py             # Chunking (RecursiveCharacterTextSplitter)
│   ├── embeddings.py           # OpenAIEmbeddings factory
│   ├── vectorstore.py          # Chroma persistence, dedup, listing
│   └── retriever.py            # Similarity search wrapper
├── prompts/
│   └── rag_prompt.py           # Citation-enforcing ChatPromptTemplate
├── utils/
│   ├── citation_utils.py       # Context formatting + citation dedup
│   └── history_utils.py        # Chat history → LangChain messages
├── config/
│   └── settings.py             # Environment-driven configuration
├── tests/
│   ├── test_splitter.py
│   └── test_citation_utils.py
├── data/
│   ├── uploads/                # Uploaded PDFs (gitignored)
│   └── vectorstore/            # Chroma persistence dir (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

```bash
git clone https://github.com/<your-username>/research-paper-rag.git
cd research-paper-rag

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```



## Running the Project

```bash
streamlit run app.py
```

1. Upload one or more PDF research papers in the sidebar.
2. Click **Process Documents** to chunk, embed, and index them.
3. Ask questions in the chat input.
4. Expand **Sources** under any answer to see the cited pages and excerpts.

## Running Tests

```bash
pytest tests/ -v
```

## Example Questions

- "What problem does this paper try to solve?"
- "Summarize the methodology described in section 3."
- "What datasets were used for evaluation, and what were the results?"
- "What are the stated limitations of this approach?"
- "How does this compare to the baseline mentioned in the paper?"



