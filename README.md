# Modern Information Retrieval — Final Project Specification
### Dual-Engine Document Search & RAG System

**Author:** Hamed Zare
**Course:** Modern Information Retrieval — Final Project

<!--
  TODO (fill in before submission):
  - [x] GitHub repository URL below
  - [ ] Demo video link in the "Video Demonstration" section
  - [ ] Evaluation results table (optional, see note in that section)
  - [ ] Final review pass after the code-quality/comment pass
-->

**Repository:** https://github.com/Hmd200/mir-search-rag

---

## 1. Overview

This project implements a full-stack document search and question-answering
system that combines three retrieval paradigms behind one API and one web
UI:

- **Classical lexical search** — TF-IDF / Vector Space Model and Okapi BM25,
  both built on a hand-written inverted index (no search libraries).
- **Semantic search** — dense vector retrieval over document embeddings
  using ChromaDB.
- **Retrieval-Augmented Generation (RAG)** — retrieved chunks are passed to
  a local LLM (Ollama) to synthesize a cited, grounded answer.

All three methods query the same underlying document corpus, which is kept
synchronized across a custom inverted index and a vector database as
documents are added, scraped from the web, or deleted.

---

## 2. System Architecture

### 2.1 Data flow

```
Upload (PDF/DOCX) or URL
        |
        v
  Parsing & extraction  (PyMuPDF, python-docx, trafilatura for URLs)
        |
        v
  Chunking  (word-window, page-boundary aware -- see 2.3)
        |
        +--------------------------+
        v                          v
  Inverted index              Embedding model
  (hand-built, TF-IDF/BM25)   (all-MiniLM-L6-v2)
        |                          |
        v                          v
  SQLite (postings, stats)    ChromaDB (vectors)
```

Both indexes are updated together on every add or delete, inside one
synchronization path, with rollback if either index write fails. This is
what keeps lexical and semantic search -- and RAG, which reads from the
vector side -- always consistent with the current corpus.

### 2.2 Retrieval methods

| Method | What it does |
|---|---|
| **TF-IDF (VSM)** | Cosine similarity over log-scaled TF-IDF vectors. Both query and document term weights use `(1 + log(tf)) × idf` (SMART `ltc.ltc` weighting — IDF is applied on both sides, not just the query side), normalized by cosine length. Query expansion via Rocchio pseudo-relevance feedback is available as a toggle. |
| **BM25** | Okapi BM25 with tunable `k1` (default 1.5) and `b` (default 0.75), using the Lucene IDF variant (`ln(1 + (N-df+0.5)/(df+0.5))`) to avoid the negative-weight edge case of the classical Robertson-Sparck Jones formula for very common terms. |
| **Semantic** | Dense vector cosine similarity via ChromaDB, using the same embedding model for documents and queries. |
| **RAG** | Semantic retrieval -> optional cross-encoder reranking -> prompt construction with numbered `[1]...[N]` source chunks -> local LLM generation -> citation validation (fabricated citation markers are detected and stripped) -> answer with linked sources. |

**Inexact top-K optimization:** champion lists (top-*r* postings per term
by term frequency) are maintained alongside the full postings lists and
used by default, with a measured reduction in postings visited versus
exhaustive scoring, and automatic fallback to full postings when a query
term's champion list doesn't yield enough candidates.

### 2.3 Chunking

Documents are split into overlapping word windows of **500 words with a
75-word overlap** (the project spec's 500/50 is given as an example, not a
requirement -- 75 was chosen to preserve more context across the overlap
for this corpus size). Windows snap to token (word) boundaries, not
sentence boundaries -- the spec asks for "logical, overlapping chunks"
without mandating sentence-aware splitting, and a word-window approach was
chosen for its simplicity and predictable, testable offsets.

Chunks are allowed to span page boundaries -- a chunk records both a
`page_start` and `page_end` (equal when the chunk stays on one page). This
was a deliberate late change: the original version reset the chunk window
at every page break, which kept citations pinned to a single exact page but
could split a paragraph or section (like the rubric table on p.5, which
continues from p.4) across two chunks that could never be retrieved
together. Allowing chunks to cross pages improves recall on content that
straddles a page break, at the cost of RAG/search citations sometimes
reading "Pages 4-5" instead of a single page number.

### 2.4 Known limitations

Documented here deliberately rather than left for a grader to discover:

- **Ingestion is O(n) per write, not incremental.** Adding or deleting a
  document currently recomputes cosine norms and champion lists for the
  whole index rather than only the affected terms. This is invisible at
  the scale of a course-project corpus (tens of documents) and was left
  as-is in favor of spending remaining time on features and verification
  rather than an internal performance rewrite of well-tested code.
- **Re-indexing older documents.** Documents indexed before the
  page-spanning chunking change (2.3) were chunked under the old
  one-page-per-chunk rule. Re-upload a document to get the new chunking
  behavior; there is no automatic migration.

---

## 3. Setup & Installation

### 3.1 Prerequisites

- Python 3.12
- Node.js 24
- [Ollama](https://ollama.com), for local LLM generation
- Git

### 3.2 Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .
```

> **Note:** dependencies are declared in `pyproject.toml`, not a
> `requirements.txt` file. `pip install -e .` installs the backend and all
> of its dependencies in one step (the `-e` installs it in "editable" mode,
> so code changes take effect immediately without reinstalling).

> **Windows note:** if `.venv\Scripts\activate` is blocked by PowerShell's
> execution policy, run this once (per user, no admin required):
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

Copy the environment template and adjust if needed (defaults work
out of the box for local development):

```bash
cp .env.example .env
```

Run the test suite to confirm everything is working:

```bash
python -m pytest -q
```

Start the API server:

```bash
uvicorn app.main:app --reload
```

The API is now available at `http://127.0.0.1:8000`. Wait for
`Application startup complete.` before using it -- the first startup in a
session can take up to 30 seconds while the embedding model initializes.

### 3.3 Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dev server proxies API requests to the
backend automatically.

### 3.4 LLM setup (Ollama)

RAG generation runs against a local Ollama model -- no API key required for
the default configuration.

```bash
ollama pull qwen3:8b
```

Confirm Ollama is running (it typically starts automatically after
installation):

```bash
ollama list
```

The backend expects Ollama at `http://127.0.0.1:11434` by default. Both the
base URL and model are configurable via environment variables (see 3.5).

**Recommended:** set `OLLAMA_KEEP_ALIVE=30m` as a system environment
variable so the model stays loaded in memory between requests -- without
this, each request after a period of inactivity pays a ~15-20s model
reload cost in addition to generation time.

The LLM client is built behind a provider interface
(`app/retrieval/llm.py`), so a different backend (e.g. an API-based
provider) can be added later without changing the RAG pipeline itself --
only `MIR_LLM_PROVIDER` and a corresponding client implementation would
need to be added. The current implementation ships with Ollama only.

### 3.5 Configuration reference

All settings below are optional overrides -- the application runs with
sensible defaults if `.env` is absent. Prefix: `MIR_`.

| Variable | Default | Purpose |
|---|---|---|
| `MIR_ENVIRONMENT` | `development` | Environment name |
| `MIR_DEBUG` | `false` | Debug mode |
| `MIR_CORS_ORIGINS` | `["http://localhost:5173", "http://127.0.0.1:5173"]` | Allowed frontend origins |
| `MIR_DATABASE_ECHO` | `false` | Log SQL statements |
| `MIR_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for semantic search and RAG |
| `MIR_EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MIR_EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding generation |
| `MIR_VECTOR_COLLECTION_NAME` | `mir_chunks` | ChromaDB collection name |
| `MIR_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `MIR_OLLAMA_MODEL` | `qwen3:8b` | Ollama model for RAG generation |
| `MIR_MAX_UPLOAD_SIZE_MB` | `25` | Maximum upload file size |
| `MIR_CHUNK_SIZE` | `500` | Words per chunk |
| `MIR_CHUNK_OVERLAP` | `75` | Word overlap between chunks |

Additional tuning (BM25 `k1`/`b`, Rocchio `alpha`/`beta`, reranker model,
champion list size, LLM provider selection) is exposed as request
parameters and/or `Settings` fields in `backend/app/core/config.py`, with
working defaults, but is not yet mirrored into `.env.example`.

---

## 4. Feature Summary

### 4.1 Core

- PDF and DOCX parsing with page/offset-preserving chunking
- Hand-built inverted index: postings lists, forward index, TF-IDF and
  BM25 scoring, champion lists for inexact top-K retrieval
- Synchronized dual-index updates (inverted index + ChromaDB) with
  rollback on partial failure
- Rocchio pseudo-relevance feedback (query expansion), toggleable per
  search, with expansion terms shown in the UI
- Semantic search over dense embeddings
- RAG: retrieval -> generation -> citation validation, with abstention
  when the corpus doesn't contain sufficient evidence
- Admin dashboard: upload, delete, corpus overview, index statistics
- Search UI: method selector, PRF toggle, reranker toggle, graded
  highlighting of matched terms by contribution score, RAG answer view
  with clickable inline citations

### 4.2 Bonus features implemented

- **Web scraping (+5):** URL ingestion via the Admin dashboard. Pages are
  fetched and their main content extracted (`trafilatura`), then flow
  through the same chunking/indexing path as an uploaded file. Includes
  SSRF protection (private/internal network targets are rejected after
  DNS resolution, not just URL-string filtering), a request timeout, and
  a download size cap.
- **Advanced RAG -- cross-encoder reranking (+5):** retrieved candidates
  are optionally reranked with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) before being passed to the LLM,
  narrowing a larger candidate set to a smaller, better-ordered context
  window. Both the original retrieval score and the post-rerank score are
  shown in the UI.
- **Innovative visualization (+5):** result snippets use graded
  highlighting -- matched terms are shaded by their actual contribution to
  the ranking score (normalized per result), rather than a flat highlight
  color, with the exact score shown on hover.

---

## 5. Testing

```bash
cd backend
python -m pytest -q
```

The suite covers, among other things: chunking and offset invariants,
hand-verified TF-IDF and BM25 scoring, champion-list behavior versus exact
scoring, Rocchio expansion correctness, dual-index synchronization and
rollback, RAG citation validation (including rejection of fabricated
citations and stripping of model "thinking" output), reranker behavior,
and web-scraping validation including the SSRF guard.

## 6. Evaluation

<!--
  TODO: optional. If added, this section should report P@5 / MRR / nDCG@10
  (or similar) across TF-IDF, TF-IDF+PRF, BM25, and semantic retrieval on a
  small hand-labeled query set, with a short discussion of what the numbers
  show (e.g. where PRF helps, where BM25 and TF-IDF diverge). Not required
  by the project spec, but strengthens the required VSM/BM25/RAG comparison
  segment of the demo video.
-->

*(Not yet completed -- optional addition.)*

---

## 7. Video Demonstration

<!-- TODO: add video link/embed here once recorded -->

`TODO: link to 5-7 minute demo video`

The video demonstrates, per the project requirements:

1. Uploading a new document and searching it
2. Deleting a document and confirming it no longer appears in search
   results (lexical and semantic)
3. Running the same query across TF-IDF, BM25, and RAG to compare results
   and UI presentation, including PRF on/off for TF-IDF
