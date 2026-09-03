# Dual-Engine Document Search & RAG

**Course:** Modern Information Retrieval — Final Project  
**Instructor:** Dr. Ghahramani  
**Author:** Hamed Zare  
**Repository:** https://github.com/Hmd200/mir-search-rag

A full-stack search engine that keeps a **hand-built inverted index** and a **Chroma vector store** in sync, then lets you switch between classical lexical retrieval (TF-IDF / BM25), dense semantic search, and citation-grounded RAG from one web UI.

---

## 1. Overview

| Surface | What you get |
|---|---|
| **Admin** (`/admin`) | Upload PDF/DOCX, scrape a public URL, list the corpus, delete (both indexes cleaned together) |
| **Search** (`/`) | TF-IDF, BM25 (Default / Tunable / Calibrated), Semantic, or RAG, with PRF, query rewrite, and rerank as toggles |

Lexical scoring is implemented from scratch (postings, TF-IDF cosine, Okapi BM25, champion lists, Rocchio PRF). Semantic search and RAG default to local `sentence-transformers/all-MiniLM-L6-v2` embeddings; **Gemini** embeddings (`gemini-embedding-001` through AvalAI) are optional via `MIR_EMBEDDING_PROVIDER=gemini` and use a separate Chroma collection. Generation defaults to **Ollama** (`qwen3:8b`); **Gemini** is an optional RAG toggle when `MIR_GEMINI_API_KEY` is set.

### A note on the name `finetuned`

BM25's third mode is called **Calibrated** everywhere a user can see it — the mode selector, the results summary, the API documentation at `/docs`, and this README.

Internally, the same mode is still spelled `finetuned`: the API value is `bm25_mode=finetuned` and the settings are `MIR_BM25_FINETUNED_K1` / `MIR_BM25_FINETUNED_B`.

**That name means parameter calibration, not neural fine-tuning. No model is trained anywhere in this project.** The mode selects two BM25 hyperparameters (`k1`, `b`) chosen by an offline grid sweep over a 12-query gold set, scored by macro nDCG@4 with MRR as the tie-breaker. 11 of the 16 grid cells tied, so the standard empirical pair `k1=1.5`, `b=0.75` was kept rather than claiming an improvement the evidence does not support. Section 8 reports the sweep in full.

The internal spelling is retained because it is the established API value and configuration key; renaming it would be a contract and configuration migration with no functional benefit. Each of those declarations carries a comment saying the same thing — see `backend/app/core/config.py`, `backend/app/api/schemas/search.py`, `frontend/src/api/client.ts`, and `.env.example`.

---

## 2. Architecture

```mermaid
flowchart TD
  src["PDF / DOCX upload or URL scrape"]
  parse["Parse and extract text"]
  chunk["Overlapping word chunks<br/>500 words, 75 overlap"]
  lex["Custom inverted index<br/>stemmed postings, TF-IDF, BM25"]
  vec["ChromaDB cosine index<br/>active embedder"]
  ui["Search UI"]
  vsm["TF-IDF + optional PRF"]
  bm25["BM25"]
  sem["Semantic kNN"]
  rag["Rewrite optional → hybrid retrieve (dense+BM25 RRF) → rerank optional → LLM + citations + grounding check"]

  src --> parse --> chunk
  chunk --> lex
  chunk --> vec
  ui --> vsm --> lex
  ui --> bm25 --> lex
  ui --> sem --> vec
  ui --> rag
  rag --> vec
  rag --> lex
```

**Indexing.** An admin upload (PDF/DOCX) or public URL is parsed to text, split into overlapping word chunks (500 words, 75-word overlap), then written to both stores: a custom inverted index (tokenized, stop-word filtered, Porter-stemmed postings with term frequencies) and Chroma (dense vectors from the active embedding provider, plus chunk text and metadata). Local MiniLM and Gemini vectors use **separate** Chroma collections and must not be mixed. Add and delete go through one path. If the vector write fails after the keyword write (or the reverse on delete), the other index is rolled back so the two stores stay aligned.

**Querying.** TF-IDF (optional Rocchio PRF) and BM25 score the inverted index. Semantic search embeds the query with the same encoder and retrieves nearest chunks from Chroma. RAG optionally rewrites the query for the **dense** arm only, then always fuses dense top-20 with BM25 top-20 (calibrated `k1`/`b`, original user wording) by unweighted RRF (`k=60`). Optional cross-encoder rerank, generation, and grounding stay on the original question. Each accepted citation group of at most two factual sentences must end with an in-range citation (a citation covers preceding sentences only); verifier failure causes safe abstention. Grounding verification adds one LLM call to successful RAG generation and is best-effort—not a formal entailment guarantee, and it does not make hallucinations impossible.

**Runtime layout** (created on first API start; gitignored except placeholders):

| Path | Role |
|---|---|
| `data/uploads/` | Stored PDF/DOCX files |
| `data/indexes/` | Persisted inverted-index JSON |
| `data/chroma/` | Persistent Chroma collection |
| `data/database/mir.db` | SQLite document/chunk metadata |
| `data/models/` | Local embedding / reranker cache |

---

## 3. Retrieval methods

| Method | Scoring | Notes |
|---|---|---|
| **TF-IDF (VSM)** | Cosine over `(1 + log tf) × idf` on **both** query and document (SMART `ltc.ltc`) | Default search uses **champion lists** (top-50 postings per term by TF), then full postings if there are too few candidates. **Rocchio PRF** (`α=1`, `β=0.75`) is a UI toggle; expansion terms are shown as chips. |
| **BM25** | Okapi BM25, Lucene IDF `ln(1 + (N−df+0.5)/(df+0.5))` | Three modes: **Default** always uses `k1=1.5`, `b=0.75`; **Tunable** uses request `k1`/`b`; **Calibrated** uses `MIR_BM25_FINETUNED_K1` / `MIR_BM25_FINETUNED_B` (also `1.5` / `0.75` after an in-sample sweep that tied). The UI labels these Default / Tunable / **Calibrated**; the API value for Calibrated is still `bm25_mode=finetuned`, and the env vars keep the `FINETUNED` names. The UI always sends the selected mode. The response reports the mode and the effective `k1`/`b`. |
| **Semantic** | Cosine nearest neighbors in Chroma | Same **active** encoder for documents and queries. Local MiniLM and Gemini indexes are separate collections. |
| **RAG** | Optional rewrite → **hybrid retrieve** (dense top-20 + BM25 top-20, unweighted RRF `k=60`) → optional rerank → generate → same-model grounding verify | Dense rewrite is optional; BM25 and the lexical gate always use the original question. Prompt requires `[1]…[N]` citations and asks for `[1][2]`, not `[1, 2]`. Terminal comma groups (`[1, 2]`, `[1,2,3]`) are normalized to `[1][2]` before validation, so the conventional comma form is accepted from any model without loosening the citation rules; malformed lists (`[1,]`, `[,2]`, `[1,,2]`) are left unparsed and fail. Fabricated markers are stripped. A successful cited draft is rewritten once against the retrieved context; each accepted group of at most two factual sentences must end with an in-range citation (configurable via `MIR_RAG_MAX_SENTENCES_PER_CITATION_GROUP`). The relevance gate admits a chunk if dense cosine ≥ `MIR_RAG_MIN_RETRIEVAL_SCORE` **or** BM25-backed lexical coverage/IDF-coverage (`MIR_RAG_LEXICAL_COVERAGE_MIN` / `MIR_RAG_LEXICAL_IDF_COVERAGE_MIN`) **or** the existing all-terms-present bypass. Verifier failure (or empty/`INSUFFICIENT_EVIDENCE` output) abstains. Model may also abstain with `INSUFFICIENT_EVIDENCE` earlier. Grounding verification adds one LLM call to successful generation and is best-effort, not a formal entailment check. |

Lexical preprocessing: Unicode tokenization, stop-word removal, **Porter stemming**. Ranking is **per chunk**; the UI shows document title, score, and a snippet (with page range when known).

---

## 4. Features (mapped to the spec)

### Rubric map

| Rubric category | Pts | Where it lives |
|---|---:|---|
| Data processing & indexing | 20 | `processing/extractors.py` (PyMuPDF, python-docx, trafilatura), `processing/chunker.py` (500/75 word windows), `storage/keyword_index.py`, `storage/vector_store.py` |
| Index synchronization | 15 | `services/documents.py` — one add path and one delete path, each with isolated per-step rollback |
| Classical retrieval | 20 | `storage/keyword_index.py` — TF-IDF `ltc.ltc` cosine, Okapi BM25, champion lists with exact fallback |
| Query expansion (PRF) | 10 | `storage/keyword_index.py` — Rocchio (`α=1`, `β=0.75`), expansion terms surfaced as UI chips |
| Semantic RAG pipeline | 20 | `services/semantic_search.py`, `services/rag.py`, `retrieval/llm.py` — dense retrieval, generation, citation validation, grounding verification |
| UI & system integration | 15 | `frontend/src/pages/AdminPage.tsx`, `frontend/src/pages/SearchPage.tsx` |
| **Bonus** web scraping | +5 | `processing/extractors.py` — `extract_from_url` with SSRF guard |
| **Bonus** visualization | +5 | `SearchPage.tsx` — term-contribution highlighting, keyword heatmap, document-relation graph |
| **Bonus** advanced RAG | +5 | `services/rag.py`, `retrieval/hybrid.py`, `retrieval/reranker.py` — query rewrite, hybrid RRF, cross-encoder rerank |

### Required

- PDF (PyMuPDF) and DOCX (python-docx) parsing, overlapping chunks, dual indexing
- Custom inverted index + Chroma; synchronized add/delete with rollback
- TF-IDF, BM25, inexact top-k (champion lists), Rocchio PRF
- RAG with citations; Ollama by default, optional Gemini
- Admin upload / table / delete; Search bar, method selector, PRF toggle, ranked snippets or RAG answer + sources

### Bonus (all three)

- **Web scraping (+5).** Admin URL field → `trafilatura` main-text extract → same dual-index path. SSRF guard (DNS + private/loopback IPs, including redirects), 10s timeout, 5 MB cap.
- **Visualization (+5).** Graded term-contribution highlighting; TF-IDF/BM25 **keyword heatmap**; lexical/semantic **document-relation graph** (shared `matched_terms`, or score proximity when terms are absent). Click a node or heatmap row to jump to that result card.
- **Advanced RAG (+5).** Optional **query rewrite** before dense retrieval (`rewritten_query` shown in the UI) and optional **cross-encoder rerank** (`cross-encoder/ms-marco-MiniLM-L-6-v2`). BM25 and the lexical gate always use the original question. Cited chunks show dense cosine, BM25, RRF fusion, and rerank scores (missing arms render as n/a, never 0.0).

---

## 5. Setup

### Prerequisites

- Python **3.12**
- Node.js **24** (or current LTS)
- [Ollama](https://ollama.com) for local RAG (optional if you only use Gemini)
- Git

Create the virtualenv at the **repository root** (the API reads `.env` and `data/` from the root).

### Backend

```bash
# from the repository root
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

cd backend
pip install -e ".[dev]"
```

`pip install -e ".[dev]"` is the install command: it installs the API **and** pytest. There is no `requirements.txt`; dependencies live in `backend/pyproject.toml`.

If PowerShell blocks `Activate.ps1`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Optional config (defaults are enough for local use):

```bash
# from the repository root
cp .env.example .env
```

On Windows: `copy .env.example .env`

Run tests (venv must be active, or use the venv Python explicitly):

```bash
cd backend
python -m pytest -q
```

Windows, without activating, from `backend/`:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Start the API from `backend/`:

```bash
uvicorn app.main:app --reload
```

API: http://127.0.0.1:8000 — OpenAPI docs: http://127.0.0.1:8000/docs  

Wait until `Application startup complete.` The first start in a session can take ~30s while the embedding model loads.

### Frontend

Second terminal, venv not required:

```bash
cd frontend
npm install
npm run dev
```

UI: http://localhost:5173  

Vite proxies `/api` to `http://127.0.0.1:8000`.

| Page | URL |
|---|---|
| Search | http://localhost:5173/ |
| Admin | http://localhost:5173/admin |

### LLM setup (Ollama and optional Gemini)

RAG can use **local Ollama** or **Google Gemini**. Lexical and semantic search need neither.

**Ollama (default, no API key)**

1. Install [Ollama](https://ollama.com).
2. Pull the default generation model and confirm it is listed:

```bash
ollama pull qwen3:8b
ollama list
```

3. Leave the default endpoint `http://127.0.0.1:11434`. Override `MIR_OLLAMA_BASE_URL` or `MIR_OLLAMA_MODEL` in a root `.env` only if you copied `.env.example` and changed those values.
4. Recommended: `OLLAMA_KEEP_ALIVE=30m` so the model is not unloaded between requests.
5. In the Search UI, select RAG → **Ollama (local)**.

**Gemini (optional API key)**

Works with a standard Google AI Studio key, or with a proxy such as **AvalAI** (what this course provides).

1. Copy `.env.example` to `.env` at the repository root (never commit `.env`).
2. Set your key, and the base URL only if you use a proxy:

```env
MIR_GEMINI_API_KEY=your_key_here
MIR_GEMINI_MODEL=gemini-2.5-flash
# Default is Google's endpoint; override only for a proxy:
# MIR_GEMINI_API_BASE=https://api.avalai.ir/v1beta
```

3. Restart the API (settings are cached at startup). In the Search UI, select RAG → **Gemini (API)**.

Google AI Studio keys work with the default base (`https://generativelanguage.googleapis.com/v1beta`). For **AvalAI**, use `https://api.avalai.ir/v1beta` — its **native Gemini** interface. Do **not** use `https://api.avalai.ir/v1`, which is AvalAI's OpenAI-compatible surface; this client posts to `{base}/models/{model}:generateContent` with an `x-goog-api-key` header and will not work against it.

**Gemini embeddings (optional and experimental — not the submitted configuration)**

The submitted and demonstrated configuration is **local MiniLM embeddings** (`MIR_EMBEDDING_PROVIDER=local`) with Chroma collection `mir_chunks`. Gemini embeddings are an **optional experimental provider**, not a recommended default and **not shown to improve retrieval** on this corpus:

- When this was measured (a 3-document corpus at the time), Gemini cosine scores for relevant and irrelevant chunks **overlapped badly** — the score distribution is compressed and high across the board (top-1 ≈ `0.70` on a query whose MiniLM top-1 is ≈ `0.28`). Relevance is not separable by an absolute threshold in the way MiniLM's is.
- `MIR_RAG_MIN_RETRIEVAL_SCORE` (`0.30`) is a **dense-cosine floor calibrated against MiniLM**. It is meaningless against Gemini's compressed distribution and was never recalibrated for it. Running RAG on Gemini embeddings therefore applies a threshold that does not correspond to the score scale in use.
- Document embedding is **serial: one HTTP request per chunk**, with no batching and no retry on `429`/`5xx`. Uploads are correspondingly slower and more fragile than local MiniLM, and add a network dependency to indexing.

It is committed as a working, tested integration, not as an accuracy claim. To enable it:

```env
MIR_EMBEDDING_PROVIDER=gemini
MIR_GEMINI_EMBEDDING_MODEL=gemini-embedding-001
MIR_GEMINI_EMBEDDING_DIMENSIONS=768
# Reuses MIR_GEMINI_API_KEY and MIR_GEMINI_API_BASE=https://api.avalai.ir/v1beta
```

Document chunks are requested with task type `RETRIEVAL_DOCUMENT`, queries with `RETRIEVAL_QUERY`, 768 dimensions, then L2-normalized. Gemini vectors are stored in a separate Chroma collection (`mir_chunks_gemini_001_768`); the existing MiniLM collection is left untouched. There is no silent fallback to MiniLM.

After switching providers, re-embed stored SQLite chunks (does not reparse files or rewrite BM25):

```bash
cd backend
python -m app.reindex_vectors
# only if the target Gemini collection is already nonempty:
python -m app.reindex_vectors --overwrite
```

The command prints counts only, and refuses to overwrite a nonempty target collection unless `--overwrite` is set. It only ever touches the **active** collection — the one selected by `MIR_EMBEDDING_PROVIDER`. Note that `--overwrite` therefore *does* delete and rebuild the MiniLM collection if you run it while `MIR_EMBEDDING_PROVIDER=local`; it is safe only in the sense that it never touches a collection other than the active one.

Do not commit `.env`. Local embedding and reranker weights (`sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) download from Hugging Face on first API start when `MIR_EMBEDDING_PROVIDER=local`. Those models are public; no Hugging Face token is required. Gemini embeddings use AvalAI over HTTP and do not download MiniLM.

### Configuration (`MIR_` prefix)

All optional. `.env` is loaded from the **repository root**.

| Variable | Default | Purpose |
|---|---|---|
| `MIR_CORS_ORIGINS` | localhost / 127.0.0.1 ports 5173 and 4173 | Frontend origins |
| `MIR_EMBEDDING_PROVIDER` | `local` | Embedding backend: `local` (MiniLM) or `gemini` (AvalAI native `embedContent`). No silent fallback. |
| `MIR_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local document and query embeddings when provider is `local` |
| `MIR_EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MIR_VECTOR_COLLECTION_NAME` | `mir_chunks` | Chroma collection for the local MiniLM index |
| `MIR_GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model id (AvalAI native) |
| `MIR_GEMINI_EMBEDDING_DIMENSIONS` | `768` | Requested output dimensionality; vectors are L2-normalized |
| `MIR_GEMINI_EMBEDDING_TIMEOUT_SECONDS` | `30` | Timeout for one Gemini embedding HTTP call |
| `MIR_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `MIR_OLLAMA_MODEL` | `qwen3:8b` | RAG generation model (Ollama) |
| `MIR_LLM_PROVIDER` | `ollama` | Default generator if the UI omits a choice |
| `MIR_GEMINI_API_KEY` | empty | Required for Gemini RAG generation and for `MIR_EMBEDDING_PROVIDER=gemini` |
| `MIR_GEMINI_API_BASE` | `https://generativelanguage.googleapis.com/v1beta` | Gemini `generateContent` base. Override for a proxy, e.g. AvalAI's native endpoint `https://api.avalai.ir/v1beta` (not `/v1`, which is OpenAI-compatible). |
| `MIR_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model id |
| `MIR_MAX_UPLOAD_SIZE_MB` | `25` | Upload cap |
| `MIR_CHUNK_SIZE` | `500` | Words per chunk |
| `MIR_CHUNK_OVERLAP` | `75` | Overlap in words |
| `MIR_RAG_MIN_RETRIEVAL_SCORE` | `0.30` | Minimum **dense cosine** for the hybrid RAG relevance gate. A context set also passes if a BM25-backed chunk meets the lexical coverage floors, or if the all-terms-present bypass fires. Do not treat this as an RRF/fusion threshold. |
| `MIR_RAG_MAX_SENTENCES_PER_CITATION_GROUP` | `2` | Maximum factual sentences a single terminal citation group may cover. A citation supports only the preceding sentences in its group, never following ones. `1` restores strict per-sentence citations. |
| `MIR_RAG_LEXICAL_COVERAGE_MIN` | `0.60` | Hybrid RAG lexical gate: minimum unique-term overlap `\|Q ∩ C\| / \|Q\|` for a BM25-retrieved chunk. |
| `MIR_RAG_LEXICAL_IDF_COVERAGE_MIN` | `0.40` | Hybrid RAG lexical gate: minimum IDF-weighted overlap of query terms present in the chunk. Out-of-vocabulary query terms stay in the denominator. |
| `MIR_BM25_FINETUNED_K1` | `1.5` | `k1` for the Calibrated mode (`bm25_mode=finetuned` on the wire) and for the RAG hybrid BM25 arm. Request `k1` is ignored in Calibrated mode. Selected by an in-sample nDCG@4 sweep on the 12-query eval gold set; the grid tied, so the standard empirical baseline was retained. |
| `MIR_BM25_FINETUNED_B` | `0.75` | `b` for the Calibrated mode (`bm25_mode=finetuned` on the wire) and for the RAG hybrid BM25 arm. Request `b` is ignored in Calibrated mode. Same in-sample calibration as `MIR_BM25_FINETUNED_K1`. |

BM25 **default** mode is the standard empirical pair `k1=1.5`, `b=0.75` and ignores request `k1`/`b`. **Tunable** mode uses the request fields (query-param defaults remain `1.5`/`0.75`). Omitting `bm25_mode` preserves the previous API: request `k1`/`b` are used. PRF `α`/`β` and rerank remain request/UI parameters (or fields in `backend/app/core/config.py`).

---

## 6. Using the system

1. Open Admin, drop a PDF or DOCX (or paste a public URL).
2. Wait until both **Keyword** and **Vector** badges say indexed.
3. Search the same query as TF-IDF, BM25, and RAG to compare.
4. On TF-IDF, toggle **Pseudo-Relevance Feedback** — expansion terms appear above the list; rankings can change.
5. On RAG, optionally enable **Rewrite query** and **Rerank results**. Citations in the answer jump to source cards.
6. Delete a document in Admin; it should disappear from lexical and semantic/RAG results.

---

## 7. Testing

```bash
cd backend
python -m pytest -q
```

Coverage includes chunking and offsets, TF-IDF/BM25 scoring, champion lists vs exact search, Rocchio expansion, dual-index sync and rollback, RAG citations (including fake markers and `<think>` stripping), query rewrite, reranker behavior, and URL scraping / SSRF rejection.

Frontend production build:

```bash
cd frontend
npm run build
```

---

## 8. Evaluation

Offline eval in `backend/evaluation/`: four short corpus files, a 12-query gold set, a throwaway index (does **not** touch live SQLite/Chroma).

```bash
cd backend
python evaluation/run_evaluation.py
python evaluation/run_evaluation.py --sweep-bm25
```

Writes `backend/evaluation/results.md`. The sweep appends a BM25 `k1`/`b` section without rewriting the method comparison. It builds **one** throwaway inverted index from `backend/evaluation/corpus/`, skips embeddings/Chroma, and scores the 12-query gold set with exact BM25 (`use_champions=False`).

The sweep is **in-sample calibration** on those 12 queries, not an independent test. Primary metric: macro nDCG@4; tie-breaker: MRR; P@4 is reported only (it is structurally flat at `top_k=4` with four documents). 11 of 16 cells tied on nDCG@4 and MRR, so **`k1=1.5`, `b=0.75` was retained** as `MIR_BM25_FINETUNED_K1` / `MIR_BM25_FINETUNED_B`. That pair is not a claim of generalization beyond this gold set.

**P@4 is `|relevant| / 4` on this corpus** (`top_k=4`, four documents). Macro P@4 / MRR / nDCG therefore barely move between methods. Empty-relevant queries are reported separately (true-negative check: P@4 = 0 for every method).

| Split | Queries | P@4 | MRR | nDCG@4 |
|---|---:|---|---|---|
| Exact vocabulary match | 5 | 0.300 all methods | 1.000 all | 1.000 all |
| Vocabulary mismatch | 6 | 0.333 all methods | 1.000 all | 1.000 lexical; **0.987 Semantic** |
| No relevant document | 1 | 0.000 all | 0.000 all | 0.000 all |

Methods still **disagree on rank-1** on some queries (full rankings in `results.md`):

- *“comparing sparse term weighting with dense neural encodings of meaning”* — TF-IDF/BM25 put `semantic_search.txt` first; Semantic puts `classical_ir.txt` first.
- *“…strongest postings per term, or … pairwise model … before the language model…”* — TF-IDF/BM25 put `rag_llm.txt` first; **TF-IDF+PRF** flips to `classical_ir.txt`.

That is the scale at which PRF and semantic search are visible here. A larger, more confusable corpus would be needed for aggregate P@k to separate the methods.

---

## 9. Known limitations

- **Index maintenance is not incremental.** Each add/delete recomputes cosine norms and champion lists for the whole inverted index. Fine for a course-sized corpus.
- **Older uploads** indexed before page-spanning chunks keep the old one-page-per-chunk split until re-uploaded.
- **LLM:** Ollama is default; Gemini generation needs `MIR_GEMINI_API_KEY`. Gemini embeddings are a separate switch (`MIR_EMBEDDING_PROVIDER`) and a separate Chroma collection; they are not a claim of better retrieval than MiniLM.
- **Gemini embeddings are experimental and uncalibrated.** Their cosine scores for relevant and irrelevant chunks overlap badly on this corpus, and `MIR_RAG_MIN_RETRIEVAL_SCORE` (`0.30`) is a MiniLM-calibrated floor that does not transfer to their compressed score scale. Document embedding is serial (one HTTP request per chunk, no batching, no `429`/`5xx` retry). The submitted configuration uses local MiniLM; see section 5.
- **RAG generation is non-deterministic.** Repeated identical RAG requests do not always produce the same outcome, since generation is sampled. Abstention is the safe outcome, not a crash. After the comma-citation fix (section 3), both providers answered the two tested BM25 parameter questions 4/4; Gemini correctly abstained on the tested Mars and CNN negatives, and Ollama on the tested Mars negative. Before that fix Gemini frequently abstained on answerable questions because it writes `[1, 2]`. These are small live samples on a course-sized corpus, not a benchmark. Long natural-language questions can still fail the retrieval gate with `low_relevance` at this corpus size, and a model that replies with a markdown bullet list is rejected by the plain-prose rule and retried.
- **RAG grounding:** Successful answers go through a same-model grounding rewrite; each accepted citation group of at most two factual sentences must end with an in-range citation, and verifier failure abstains. Same-model verification is best-effort and is not a formal entailment guarantee—hallucinations remain possible. Grounding verification adds one LLM call to successful RAG generation.
- **RAG answer-relevance.** Grounding checks claim support, not whether the answer addresses the asked property. A supported but off-target answer (for example a related count instead of the specific one requested) can still be returned. A single-call structured verdict was attempted and did not close this gap without adding a second LLM call.
- **RAG lexical false positives.** Hybrid retrieval admits some dense-weak, BM25-strong questions that have no answer in the corpus. Safe downstream abstention (`model_abstained`, `citation_failure`, or `grounding_failure`) is acceptable; an unsupported answer is not. Calibration of the coverage / IDF-coverage floors was on a small corpus, so phrasing changes can straddle the gate.

---

## 10. Video demonstration

**Watch:** https://drive.google.com/file/d/1nxe5bdAecuFlUmirmeJ0r-Kq4iskhpf_/view

Recorded 29 August 2026 against commit `91b4151`. Shared as *Anyone with the link — Viewer*; no Google account is required.

Shots covered:

1. Upload a document, then search it.
2. Delete it, then show it gone from lexical **and** semantic/RAG.
3. Same query on TF-IDF (PRF off/on), BM25, and RAG — compare lists vs cited answer.
4. Bonus: URL ingest; heatmap + relation graph; RAG rewrite and rerank.
