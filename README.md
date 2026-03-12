# document-qna-rag

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![LangChain](https://img.shields.io/badge/LangChain-1.x-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-purple)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Render](https://img.shields.io/badge/Deployed-Render-brightgreen)

An end-to-end production RAG application for intelligent document question answering. Upload a PDF or DOCX, ask questions in natural language, and get grounded answers with source citations — powered by OpenAI embeddings, Pinecone vector search, Cohere reranking, and GPT-4o-mini.

**Live Demo:** https://document-qna-rag-ui.onrender.com  
**API Docs:** https://document-qna-rag.onrender.com/docs

---

## Demo

> TODO: Add Loom demo GIF here

---

## Architecture

> TODO: Add architecture diagram (Excalidraw) here

```
PDF/DOCX Upload
      ↓
Text Extraction (PyPDF2 / python-docx)
      ↓
Recursive Character Chunking
      ↓
OpenAI Embeddings (text-embedding-3-small)
      ↓
Pinecone Upsert (namespace per document)
      ↓
─────────────────────────────────────────
User Query
      ↓
Query Embedding
      ↓
Pinecone Similarity Search (top_k * 3)
      ↓
Cohere Reranker (rerank-english-v3.0)
      ↓
Prompt Assembly (context + chat history)
      ↓
GPT-4o-mini (streaming)
      ↓
Answer + Source Citations
```

---

## Key Engineering Decisions

**Namespace-per-document isolation**  
Each uploaded document gets its own Pinecone namespace, with a corresponding chat history scoped to that namespace. This means a user can upload multiple documents and switch between them mid-session without context bleed — each document maintains independent retrieval and conversation state.

**Over-fetching before reranking**  
Pinecone retrieves `top_k * 3` candidates before passing to Cohere. The reranker needs a meaningful candidate pool to reorder — fetching only `top_k` gives it nothing to work with. The final `top_k` results returned to the LLM are post-rerank.

**Manual chat history management**  
Chat history is managed as a plain list of `HumanMessage` / `AIMessage` objects with a sliding window of k=5 exchanges — keeping the last 5 turns in context without bloating the prompt as conversations grow longer.

**Chunk metadata for citations**  
Each vector upserted to Pinecone carries `filename`, `page`, and `chunk_index` in its metadata. This allows the app to return source citations (filename + page number) alongside every answer without a separate database lookup.

**Boilerplate chunk pollution**  
PDFs from sources like JSTOR include repeated copyright/boilerplate text that gets indexed as content and scores surprisingly high on semantic similarity, surfacing as low-quality retrieved chunks. See "What I Would Do Next" for the fix.

---

## Evaluation Results

Evaluated on a 10-question testset from a Salesforce financial guidance document. Metrics measured using RAGAS across two retrieval configurations.

| Configuration | Faithfulness | Context Precision | Context Recall | Answer Relevancy |
|---|---|---|---|---|
| Dense only (baseline) | 0.80 | 0.94 | 1.00 | 0.82 |
| Dense + Cohere rerank | 0.90 | 0.97 | 1.00 | 0.83 |

Reranking improved faithfulness by +0.10 — answers are more grounded in retrieved context. Context precision improved by +0.03. Context recall is 1.00 for both configurations — expected on a single-page document where all relevant content is retrievable. Answer relevancy is nearly identical across configs, which makes sense — reranking changes what is retrieved, not how relevant the question is.

---

## Running Locally

**Prerequisites:** Python 3.11, Docker (optional)

```bash
# 1. clone the repo
git clone https://github.com/saikaushik1997/document-qna-rag
cd document-qna-rag

# 2. install dependencies
pip install -r requirements.txt

# 3. set up environment variables
cp .env.example .env
# fill in your API keys in .env

# 4. start the API
uvicorn app.main:app --reload

# 5. start the UI (separate terminal)
streamlit run streamlit_app.py
```

API available at `http://localhost:8000/docs`  
UI available at `http://localhost:8501`

**With Docker:**
```bash
docker compose up
```

---

## What I Would Do Next

**Pre-processing boilerplate filter**  
Add a minimum token count threshold during ingestion to drop low-content chunks (copyright notices, headers, footers) before they get indexed. Chunks under ~20 tokens are almost always noise.

**Hybrid search**  
Add BM25 sparse vectors alongside dense embeddings in Pinecone for hybrid retrieval. Tunable alpha parameter balances keyword vs semantic search — better for queries with specific technical terms that dense search misses.

**RAGAS-driven chunk size tuning**  
Run ablation across chunk sizes (256 / 512 / 1024 tokens) and measure faithfulness and context precision. The optimal chunk size is document-type dependent and worth measuring rather than assuming.

**Multi-document querying**  
Currently queries are scoped to a single document namespace. A cross-namespace query mode would let users ask questions across all uploaded documents simultaneously.

**Persistent chat history**  
Chat history currently lives in memory and resets on server restart. Storing it in Redis or a database would allow conversations to persist across sessions.

**Streaming through FastAPI**  
Streaming is implemented at the generation layer but not exposed through the FastAPI endpoint. Implementing SSE (Server-Sent Events) would allow true token-by-token streaming through the API.