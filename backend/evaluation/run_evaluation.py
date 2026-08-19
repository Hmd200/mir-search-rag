"""Offline, reproducible retrieval evaluation independent of live app data.

This script is a README and demo artifact. It builds a throwaway inverted
index and a throwaway Chroma collection from four local corpus files, runs
TF-IDF, TF-IDF+PRF, BM25, and semantic search against a hand-labeled gold
set, and prints P@4 (reported in place of P@5; the corpus has only four
documents, so results are never padded), MRR, and nDCG@4.

It does not read the application SQLite database, the production Chroma
directory, or uploaded files under data/uploads. Temporary stores are
deleted on exit.

Run from the backend package root:

    python evaluation/run_evaluation.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from app.storage.keyword_index import KeywordIndex, KeywordSearchHit
from app.storage.vector_store import ChromaVectorStore, VectorChunk, VectorSearchHit

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLD_PATH = EVAL_DIR / "gold_set.json"
RESULTS_PATH = EVAL_DIR / "results.md"
TOP_K = 4
METHODS = ("TF-IDF", "TF-IDF+PRF", "BM25", "Semantic")


@dataclass(frozen=True, slots=True)
class GoldQuery:
    query: str
    relevant_files: frozenset[str]
    note: str


@dataclass(frozen=True, slots=True)
class MethodScores:
    ranked_ids: tuple[str, ...]
    scores: tuple[float, ...]
    p_at_4: float
    reciprocal_rank: float
    ndcg_at_4: float


def _load_corpus() -> list[tuple[str, str]]:
    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No corpus files found in {CORPUS_DIR}")
    documents: list[tuple[str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Corpus file is empty: {path.name}")
        documents.append((path.name, text))
    return documents


def _load_gold() -> list[GoldQuery]:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    queries: list[GoldQuery] = []
    for item in payload:
        queries.append(
            GoldQuery(
                query=item["query"],
                relevant_files=frozenset(item["relevant_files"]),
                note=item["note"],
            )
        )
    if len(queries) < 8:
        raise ValueError("gold_set.json must contain at least 8 queries.")
    return queries


def _precision_at_k(
    ranked_ids: Sequence[str],
    relevant: frozenset[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    return sum(1 for chunk_id in top if chunk_id in relevant) / k


def _reciprocal_rank(ranked_ids: Sequence[str], relevant: frozenset[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(index + 1) for index, gain in enumerate(gains, start=1))


def _ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant: frozenset[str],
    k: int,
) -> float:
    gains = [1.0 if chunk_id in relevant else 0.0 for chunk_id in ranked_ids[:k]]
    ideal_count = min(len(relevant), k)
    ideal = [1.0] * ideal_count + [0.0] * (k - ideal_count)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def _keyword_ids(hits: Sequence[KeywordSearchHit]) -> tuple[str, ...]:
    return tuple(hit.chunk_id for hit in hits)


def _keyword_scores(hits: Sequence[KeywordSearchHit]) -> tuple[float, ...]:
    return tuple(hit.score for hit in hits)


def _vector_ids(hits: Sequence[VectorSearchHit]) -> tuple[str, ...]:
    return tuple(hit.chunk_id for hit in hits)


def _vector_scores(hits: Sequence[VectorSearchHit]) -> tuple[float, ...]:
    return tuple(hit.score for hit in hits)


def _score_method(
    ranked_ids: Sequence[str],
    scores: Sequence[float],
    relevant: frozenset[str],
) -> MethodScores:
    return MethodScores(
        ranked_ids=tuple(ranked_ids),
        scores=tuple(scores),
        p_at_4=_precision_at_k(ranked_ids, relevant, TOP_K),
        reciprocal_rank=_reciprocal_rank(ranked_ids, relevant),
        ndcg_at_4=_ndcg_at_k(ranked_ids, relevant, TOP_K),
    )


def _build_indexes(
    documents: Sequence[tuple[str, str]],
    workdir: Path,
    settings: Settings,
) -> tuple[KeywordIndex, ChromaVectorStore, SentenceTransformerEmbeddingProvider]:
    keyword_index = KeywordIndex(workdir / "keyword-index.json")
    vector_store = ChromaVectorStore(workdir / "chroma", "eval_chunks")
    embeddings = SentenceTransformerEmbeddingProvider(
        settings.embedding_model,
        cache_dir=settings.model_dir,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    for filename, text in documents:
        keyword_index.upsert_document(filename, ((filename, text),))
        vector_store.upsert_document(
            filename,
            (
                VectorChunk(
                    chunk_id=filename,
                    document_id=filename,
                    text=text,
                    position=0,
                ),
            ),
            embeddings,
        )
    return keyword_index, vector_store, embeddings


def _run_methods(
    query: str,
    keyword_index: KeywordIndex,
    vector_store: ChromaVectorStore,
    embeddings: SentenceTransformerEmbeddingProvider,
    relevant: frozenset[str],
) -> dict[str, MethodScores]:
    tfidf_hits = keyword_index.search(
        query,
        top_k=TOP_K,
        use_champions=False,
    )
    prf_hits = list(
        keyword_index.search_with_prf(
            query,
            top_k=TOP_K,
            feedback_docs=2,
            max_expansion_terms=8,
            scoring_mode="tfidf",
            use_champions=False,
        ).hits
    )
    bm25_hits = keyword_index.search_bm25(
        query,
        top_k=TOP_K,
        use_champions=False,
    )
    semantic_hits = vector_store.search(
        query,
        embeddings,
        top_k=TOP_K,
    )
    return {
        "TF-IDF": _score_method(_keyword_ids(tfidf_hits), _keyword_scores(tfidf_hits), relevant),
        "TF-IDF+PRF": _score_method(_keyword_ids(prf_hits), _keyword_scores(prf_hits), relevant),
        "BM25": _score_method(_keyword_ids(bm25_hits), _keyword_scores(bm25_hits), relevant),
        "Semantic": _score_method(
            _vector_ids(semantic_hits),
            _vector_scores(semantic_hits),
            relevant,
        ),
    }


def _format_ranking(result: MethodScores) -> str:
    if not result.ranked_ids:
        return "(no hits)"
    parts = [
        f"{rank}.{chunk_id}:{score:.4f}"
        for rank, (chunk_id, score) in enumerate(
            zip(result.ranked_ids, result.scores, strict=True),
            start=1,
        )
    ]
    return "  ".join(parts)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_true_negative(gold: GoldQuery) -> bool:
    return not gold.relevant_files


def _is_exact_match(gold: GoldQuery) -> bool:
    return gold.note.casefold().startswith("exact vocabulary match")


def _is_vocab_mismatch(gold: GoldQuery) -> bool:
    # Notes such as "exact vocabulary match on ... vocabulary mismatch"
    # name the IR topic; they are exact-match queries, not mismatch ones.
    return (
        not _is_true_negative(gold)
        and not _is_exact_match(gold)
        and "vocabulary mismatch" in gold.note.casefold()
    )


def _select(
    queries: Sequence[GoldQuery],
    per_query: Sequence[dict[str, MethodScores]],
    predicate,
) -> list[dict[str, MethodScores]]:
    return [
        results
        for gold, results in zip(queries, per_query, strict=True)
        if predicate(gold)
    ]


def _macro_rows(
    subset: Sequence[dict[str, MethodScores]],
) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    for method in METHODS:
        rows.append(
            (
                method,
                _mean([row[method].p_at_4 for row in subset]),
                _mean([row[method].reciprocal_rank for row in subset]),
                _mean([row[method].ndcg_at_4 for row in subset]),
            )
        )
    return rows


def _append_macro_table(
    lines: list[str],
    title: str,
    query_count: int,
    rows: Sequence[tuple[str, float, float, float]],
) -> None:
    lines.append(title)
    lines.append(f"Queries in this average: {query_count}")
    lines.append(f"{'Method':<12} {'P@4':>8} {'MRR':>8} {'nDCG@4':>8}")
    lines.append("-" * 40)
    for method, p_mean, mrr, ndcg_mean in rows:
        lines.append(f"{method:<12} {p_mean:8.3f} {mrr:8.3f} {ndcg_mean:8.3f}")
    lines.append("")


def _markdown_table(
    rows: Sequence[tuple[str, float, float, float]],
) -> list[str]:
    lines = [
        "| Method | P@4 | MRR | nDCG@4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, p_mean, mrr, ndcg_mean in rows:
        lines.append(f"| {method} | {p_mean:.3f} | {mrr:.3f} | {ndcg_mean:.3f} |")
    return lines


def _print_and_collect(
    queries: Sequence[GoldQuery],
    per_query: Sequence[dict[str, MethodScores]],
) -> tuple[str, str]:
    lines: list[str] = [
        "Offline retrieval evaluation",
        "Corpus: 4 documents, one chunk each. top_k=4.",
        "P@5 is reported as P@4: there are only four documents, so the",
        "run never pads to five results and never divides by five.",
        "",
    ]

    for gold, results in zip(queries, per_query, strict=True):
        relevant = ", ".join(sorted(gold.relevant_files)) or "(none)"
        lines.append("=" * 88)
        lines.append(f"Query: {gold.query}")
        lines.append(f"Relevant: {relevant}")
        lines.append(f"Note: {gold.note}")
        lines.append(
            f"{'Method':<12} {'P@4':>8} {'RR':>8} {'nDCG@4':>8}  Ranking"
        )
        lines.append("-" * 88)
        for method in METHODS:
            scores = results[method]
            lines.append(
                f"{method:<12} {scores.p_at_4:8.3f} {scores.reciprocal_rank:8.3f} "
                f"{scores.ndcg_at_4:8.3f}  {_format_ranking(scores)}"
            )
        lines.append("")

    lines.append("=" * 88)
    lines.append("Raw P@4 values across all queries (before averaging)")
    for method in METHODS:
        raw = [row[method].p_at_4 for row in per_query]
        formatted = ", ".join(f"{value:.3f}" for value in raw)
        lines.append(f"{method:<12} [{formatted}]")
    lines.append("")
    lines.append(
        "The previous all-query macro P@4 of 0.250 was not an averaging-code "
        "error. Eight single-relevant queries contribute 0.250, the two-"
        "relevant query contributes 0.500, and the empty-relevant query "
        "contributes 0.000: (8*0.250 + 0.500 + 0.000)/10 = 0.250. The 0.500 "
        "and 0.000 cancel, which is why the published mean looked identical "
        "to the typical single-relevant P@4. Empty-relevant queries are "
        "excluded from the macro tables below and reported as a true-negative "
        "check instead."
    )
    lines.append("")

    true_negatives = [
        (gold, results)
        for gold, results in zip(queries, per_query, strict=True)
        if _is_true_negative(gold)
    ]
    lines.append("=" * 88)
    lines.append("True-negative check (empty relevant set; not in macro-averages)")
    if not true_negatives:
        lines.append("No empty-relevant-set queries in the gold set.")
        lines.append("")
    else:
        for gold, results in true_negatives:
            lines.append(f"Query: {gold.query}")
            lines.append(
                f"{'Method':<12} {'P@4':>8} {'RR':>8} {'nDCG@4':>8}"
            )
            lines.append("-" * 40)
            for method in METHODS:
                scores = results[method]
                lines.append(
                    f"{method:<12} {scores.p_at_4:8.3f} "
                    f"{scores.reciprocal_rank:8.3f} {scores.ndcg_at_4:8.3f}"
                )
            all_zero = all(results[method].p_at_4 == 0.0 for method in METHODS)
            if all_zero:
                lines.append(
                    "P@4 = 0 for every method: no labeled-relevant document "
                    "exists, so no method can surface a relevant hit."
                )
            else:
                lines.append(
                    "P@4 was not 0 for every method; inspect the ranking above."
                )
            lines.append("")

    exact_subset = _select(queries, per_query, _is_exact_match)
    mismatch_subset = _select(queries, per_query, _is_vocab_mismatch)
    exact_rows = _macro_rows(exact_subset)
    mismatch_rows = _macro_rows(mismatch_subset)

    lines.append("=" * 88)
    _append_macro_table(
        lines,
        "Macro-average over exact-match queries",
        len(exact_subset),
        exact_rows,
    )
    _append_macro_table(
        lines,
        "Macro-average over vocabulary-mismatch queries",
        len(mismatch_subset),
        mismatch_rows,
    )
    lines.append(
        "nDCG@4 uses binary relevance. Empty-relevant-set queries are omitted "
        "from both macro-averages above."
    )
    lines.append("")

    markdown = [
        "# Offline retrieval evaluation",
        "",
        "Built from `backend/evaluation/` against four held-out corpus files.",
        "This run does not use the live SQLite database, production Chroma",
        "store, or uploaded documents.",
        "",
        "The labeled metric **P@5** is computed as **P@4**: the collection",
        "contains only four documents, `top_k=4`, and missing ranks are not",
        "padded.",
        "",
        "## Per-query P@4 (all queries)",
        "",
    ]
    for gold, results in zip(queries, per_query, strict=True):
        relevant = ", ".join(sorted(gold.relevant_files)) or "(none)"
        markdown.append(f"### {gold.query}")
        markdown.append("")
        markdown.append(f"Relevant: `{relevant}`. {gold.note}")
        markdown.append("")
        markdown.append("| Method | P@4 | RR | nDCG@4 |")
        markdown.append("| --- | ---: | ---: | ---: |")
        for method in METHODS:
            scores = results[method]
            markdown.append(
                f"| {method} | {scores.p_at_4:.3f} | "
                f"{scores.reciprocal_rank:.3f} | {scores.ndcg_at_4:.3f} |"
            )
        markdown.append("")

    markdown.append("## Raw P@4 lists (before averaging)")
    markdown.append("")
    for method in METHODS:
        raw = [row[method].p_at_4 for row in per_query]
        formatted = ", ".join(f"{value:.3f}" for value in raw)
        markdown.append(f"- **{method}:** [{formatted}]")
    markdown.append("")
    markdown.append(
        "Including the empty-relevant query (P@4 = 0) in the same average as "
        "the two-relevant query (P@4 = 0.500) cancelled the lift from the "
        "latter and flattened every method to 0.250. True-negative queries "
        "are now reported separately."
    )
    markdown.append("")
    markdown.append("## True-negative check")
    markdown.append("")
    if not true_negatives:
        markdown.append("No empty-relevant-set queries in the gold set.")
        markdown.append("")
    else:
        markdown.append(
            "These queries have `relevant_files: []` and are **excluded** "
            "from the macro-average tables. P@4 = 0 is required for every "
            "method because nothing in the corpus is labeled relevant."
        )
        markdown.append("")
        for gold, results in true_negatives:
            markdown.append(f"**Query:** {gold.query}")
            markdown.append("")
            markdown.append("| Method | P@4 | RR | nDCG@4 |")
            markdown.append("| --- | ---: | ---: | ---: |")
            for method in METHODS:
                scores = results[method]
                markdown.append(
                    f"| {method} | {scores.p_at_4:.3f} | "
                    f"{scores.reciprocal_rank:.3f} | {scores.ndcg_at_4:.3f} |"
                )
            markdown.append("")

    markdown.append("## Exact-match macro-average")
    markdown.append("")
    markdown.append(
        f"Macro-mean over {len(exact_subset)} exact-vocabulary queries "
        "(empty-relevant queries excluded)."
    )
    markdown.append("")
    markdown.extend(_markdown_table(exact_rows))
    markdown.append("")
    markdown.append("## Vocabulary-mismatch macro-average")
    markdown.append("")
    markdown.append(
        f"Macro-mean over {len(mismatch_subset)} vocabulary-mismatch queries "
        "(empty-relevant queries excluded)."
    )
    markdown.append("")
    markdown.extend(_markdown_table(mismatch_rows))
    markdown.append("")
    markdown.append(
        "nDCG@4 uses binary relevance. Empty-relevant-set queries are omitted "
        "from both macro-averages."
    )
    markdown.append("")
    return "\n".join(lines) + "\n", "\n".join(markdown) + "\n"


def main() -> int:
    documents = _load_corpus()
    queries = _load_gold()
    settings = Settings()
    workdir = Path(tempfile.mkdtemp(prefix="mir-eval-"))
    try:
        keyword_index, vector_store, embeddings = _build_indexes(
            documents,
            workdir,
            settings,
        )
        per_query = [
            _run_methods(
                gold.query,
                keyword_index,
                vector_store,
                embeddings,
                gold.relevant_files,
            )
            for gold in queries
        ]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    console, markdown = _print_and_collect(queries, per_query)
    print(console, end="")
    RESULTS_PATH.write_text(markdown, encoding="utf-8", newline="\r\n")
    print(f"Wrote {RESULTS_PATH.relative_to(BACKEND_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
