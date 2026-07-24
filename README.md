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

The project is scoped to demonstrate a complete, correct RAG
pipeline loading, chunking, embedding, vector storage, retrieval, prompt
construction, generation, and citation  

## Features

-  Multi file PDF upload with per page text extraction
-  Recursive character chunking with configurable size/overlap
-  OpenAI embeddings persisted in a local Chroma vector store
-  Similarity search retrieval with relevance scores
-  Citation enforcing prompt template (page-level `[Source, p. X]` tags)
-  LangGraph orchestrated pipeline: `retrieve → generate`
-  Multi turn conversation memory (follow up question resolution)
-  Sidebar showing indexed documents, with clear chat / clear documents controls
-  Idempotent ingestion  reuploading a paper does not duplicate chunks
-  Unit tests for chunking and citation logic


### RAG Pipeline (LangGraph)

```
        START
          │
          ▼
 ┌──────────────────┐
 │  retrieve          │   
 │  (graph/nodes/     │  
 │   retrieve.py)     │
 └────────┬──────────┘
          │
          ▼
 ┌──────────────────┐
 │  generate          │ 
 │  (graph/nodes/     │   
 │   generate.py)     │ 
 └────────┬──────────┘
          │
          ▼
         END
```





