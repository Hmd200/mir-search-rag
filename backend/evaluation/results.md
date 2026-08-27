# Offline retrieval evaluation

Built from `backend/evaluation/` against four held-out corpus files.
This run does not use the live SQLite database, production Chroma
store, or uploaded documents.

The labeled metric **P@5** is computed as **P@4**: the collection
contains only four documents, `top_k=4`, and missing ranks are not
padded.

## Per-query P@4 (all queries)

### Okapi BM25 ranking function inverted index postings lists term frequency

Relevant: `classical_ir.txt`. exact vocabulary match on BM25, inverted index, and postings lists

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.250 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.250 | 1.000 | 1.000 |
| BM25 | 0.250 | 1.000 | 1.000 |
| Semantic | 0.250 | 1.000 | 1.000 |

### dense vector embeddings cosine similarity vocabulary mismatch

Relevant: `semantic_search.txt`. exact vocabulary match on embeddings, cosine similarity, and vocabulary mismatch

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.250 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.250 | 1.000 | 1.000 |
| BM25 | 0.250 | 1.000 | 1.000 |
| Semantic | 0.250 | 1.000 | 1.000 |

### retrieval-augmented generation citations grounding hallucination

Relevant: `rag_llm.txt`. exact vocabulary match on RAG, citations, grounding, and hallucination

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.250 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.250 | 1.000 | 1.000 |
| BM25 | 0.250 | 1.000 | 1.000 |
| Semantic | 0.250 | 1.000 | 1.000 |

### compost mulch tomato pruning raised beds basil harvest

Relevant: `unrelated_topic.txt`. exact vocabulary match on the gardening document

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.250 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.250 | 1.000 | 1.000 |
| BM25 | 0.250 | 1.000 | 1.000 |
| Semantic | 0.250 | 1.000 | 1.000 |

### How do search engines turn words into scores using collection statistics?

Relevant: `classical_ir.txt`. vocabulary mismatch: asks about term statistics without naming TF-IDF or BM25

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.250 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.250 | 1.000 | 1.000 |
| BM25 | 0.250 | 1.000 | 1.000 |
| Semantic | 0.250 | 1.000 | 1.000 |

### Finding similar passages by meaning when the author uses different wording

Relevant: `semantic_search.txt`. vocabulary mismatch: describes semantic matching without saying embeddings or cosine

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.250 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.250 | 1.000 | 1.000 |
| BM25 | 0.250 | 1.000 | 1.000 |
| Semantic | 0.250 | 1.000 | 1.000 |

### Stopping a language model from inventing facts by attaching quoted source passages

Relevant: `rag_llm.txt`. vocabulary mismatch: describes grounding and hallucination without those labels

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.250 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.250 | 1.000 | 1.000 |
| BM25 | 0.250 | 1.000 | 1.000 |
| Semantic | 0.250 | 1.000 | 1.000 |

### When should I harvest basil and how do I dry the leaves for winter pesto?

Relevant: `unrelated_topic.txt`. vocabulary mismatch toward gardening: pesto is not in the source, harvest and drying are

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.250 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.250 | 1.000 | 1.000 |
| BM25 | 0.250 | 1.000 | 1.000 |
| Semantic | 0.250 | 1.000 | 1.000 |

### comparing sparse term weighting with dense neural encodings of meaning

Relevant: `classical_ir.txt, semantic_search.txt`. vocabulary mismatch spanning two topics: sparse weighting vs dense encodings

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.500 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.500 | 1.000 | 1.000 |
| BM25 | 0.500 | 1.000 | 1.000 |
| Semantic | 0.500 | 1.000 | 1.000 |

### Okapi BM25 inverted index postings lists TF-IDF cosine and dense vector embeddings

Relevant: `classical_ir.txt, semantic_search.txt`. exact vocabulary match spanning classical sparse retrieval and dense embeddings

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.500 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.500 | 1.000 | 1.000 |
| BM25 | 0.500 | 1.000 | 1.000 |
| Semantic | 0.500 | 1.000 | 1.000 |

### Should a smaller candidate set come from the strongest postings per term, or from a slower pairwise model that reorders chunks right before the language model is prompted?

Relevant: `classical_ir.txt, rag_llm.txt`. vocabulary mismatch spanning two topics: champion lists vs cross-encoder reranking

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.500 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.500 | 1.000 | 1.000 |
| BM25 | 0.500 | 1.000 | 1.000 |
| Semantic | 0.500 | 1.000 | 0.920 |

### What torque specification should I use when replacing a bicycle bottom bracket?

Relevant: `(none)`. no relevant document, tests correct low scoring across all methods

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.000 | 0.000 | 0.000 |
| TF-IDF+PRF | 0.000 | 0.000 | 0.000 |
| BM25 | 0.000 | 0.000 | 0.000 |
| Semantic | 0.000 | 0.000 | 0.000 |

## Raw P@4 lists (before averaging)

- **TF-IDF:** [0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.500, 0.500, 0.500, 0.000]
- **TF-IDF+PRF:** [0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.500, 0.500, 0.500, 0.000]
- **BM25:** [0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.500, 0.500, 0.500, 0.000]
- **Semantic:** [0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.500, 0.500, 0.500, 0.000]

Including the empty-relevant query (P@4 = 0) in the same average as the two-relevant query (P@4 = 0.500) cancelled the lift from the latter and flattened every method to 0.250. True-negative queries are now reported separately.

## True-negative check

These queries have `relevant_files: []` and are **excluded** from the macro-average tables. P@4 = 0 is required for every method because nothing in the corpus is labeled relevant.

**Query:** What torque specification should I use when replacing a bicycle bottom bracket?

| Method | P@4 | RR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.000 | 0.000 | 0.000 |
| TF-IDF+PRF | 0.000 | 0.000 | 0.000 |
| BM25 | 0.000 | 0.000 | 0.000 |
| Semantic | 0.000 | 0.000 | 0.000 |

## Exact-match macro-average

Macro-mean over 5 exact-vocabulary queries (empty-relevant queries excluded).

| Method | P@4 | MRR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.300 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.300 | 1.000 | 1.000 |
| BM25 | 0.300 | 1.000 | 1.000 |
| Semantic | 0.300 | 1.000 | 1.000 |

## Vocabulary-mismatch macro-average

Macro-mean over 6 vocabulary-mismatch queries (empty-relevant queries excluded).

| Method | P@4 | MRR | nDCG@4 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.333 | 1.000 | 1.000 |
| TF-IDF+PRF | 0.333 | 1.000 | 1.000 |
| BM25 | 0.333 | 1.000 | 1.000 |
| Semantic | 0.333 | 1.000 | 0.987 |

nDCG@4 uses binary relevance. Empty-relevant-set queries are omitted from both macro-averages.

## BM25 parameter sweep (in-sample calibration)

This section calibrates Okapi BM25 `k1` and `b` on the **same**
12-query gold set used above. It is **in-sample calibration**, not an
independent test: the same queries both select and report the
parameters. The selected pair is not evidence of generalization
beyond this set.

**Optimized metric:** macro nDCG@4 (mean over all 12 gold queries,
including the empty-relevant true-negative query).
**Tie-breaker:** MRR (same 12-query mean).
**Reported, non-selecting:** P@4. With four corpus documents and
`top_k=4`, P@4 moves only when BM25 returns fewer than four
candidates. Reordering alone never changes it, so it cannot rank
parameter pairs.

Grid: `k1 ∈ {0.9, 1.2, 1.5, 2.0}` × `b ∈ {0.3, 0.5, 0.75, 1.0}`
(16 cells, evaluated in that order). One throwaway inverted index
was built from `backend/evaluation/corpus/` and reused for every
cell. No embeddings and no Chroma collection were built. Scoring
used exact BM25 (`use_champions=False`). Temporary stores were
deleted on exit.

Labeled relevant occurrences counted below: **14**
across **12** queries (the empty-relevant query
contributes none). Rank columns are **counts**, not rates.

### Full grid

| k1 | b | nDCG@4 | MRR | P@4 | rank 1 | rank 2 | rank 3 | rank 4 | missed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 0.30 | 0.886 | 0.875 | 0.292 | 10 | 4 | 0 | 0 | 0 |
| 0.9 | 0.50 | 0.886 | 0.875 | 0.292 | 10 | 4 | 0 | 0 | 0 |
| 0.9 | 0.75 | 0.886 | 0.875 | 0.292 | 10 | 4 | 0 | 0 | 0 |
| 0.9 | 1.00 | 0.886 | 0.875 | 0.292 | 10 | 4 | 0 | 0 | 0 |
| 1.2 | 0.30 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 1.2 | 0.50 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 1.2 | 0.75 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 1.2 | 1.00 | 0.886 | 0.875 | 0.292 | 10 | 4 | 0 | 0 | 0 |
| 1.5 | 0.30 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 1.5 | 0.50 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 1.5 | 0.75 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 1.5 | 1.00 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 2.0 | 0.30 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 2.0 | 0.50 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 2.0 | 0.75 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |
| 2.0 | 1.00 | 0.917 | 0.917 | 0.292 | 11 | 3 | 0 | 0 | 0 |

### Selection

Distinct nDCG@4 values in the grid: 2. Distinct MRR values: 2. Distinct P@4 values: 1.

P@4 is 0.292 in every cell, so it cannot rank parameter pairs.

11 of 16 cells share the winning macro nDCG@4 (0.917) and MRR (0.917). 5 cells are strictly worse on both metrics.

**Selected:** `k1=1.5`, `b=0.75` (tied macro nDCG@4 and MRR; retained standard empirical baseline k1=1.5, b=0.75).

These values are the calibrated defaults (`MIR_BM25_FINETUNED_K1` / `MIR_BM25_FINETUNED_B`).
