"""Unweighted RRF fusion and lexical-gate helpers for hybrid RAG."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

RRF_K = 60
HYBRID_RETRIEVAL_K = 20
RetrievalSource = Literal["dense", "bm25"]


def first_ranks(chunk_ids: Sequence[str]) -> dict[str, int]:
    """Return 1-based rank of each chunk's first appearance."""

    ranks: dict[str, int] = {}
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id not in ranks:
            ranks[chunk_id] = rank
    return ranks


def reciprocal_rank_fusion(
    dense_ids: Sequence[str],
    bm25_ids: Sequence[str],
    *,
    k: int = RRF_K,
) -> dict[str, float]:
    """Fuse two rank lists with unweighted RRF.

    An arm that did not retrieve a chunk contributes no term. A chunk
    retrieved by only one arm still receives that arm's RRF score.
    """

    if k <= 0:
        raise ValueError("RRF k must be greater than zero.")

    dense_ranks = first_ranks(dense_ids)
    bm25_ranks = first_ranks(bm25_ids)
    scores: dict[str, float] = {}
    for chunk_id in {**dense_ranks, **bm25_ranks}:
        score = 0.0
        dense_rank = dense_ranks.get(chunk_id)
        if dense_rank is not None:
            score += 1.0 / (k + dense_rank)
        bm25_rank = bm25_ranks.get(chunk_id)
        if bm25_rank is not None:
            score += 1.0 / (k + bm25_rank)
        scores[chunk_id] = score
    return scores


def fused_chunk_order(
    scores: Mapping[str, float],
    *,
    top_k: int = HYBRID_RETRIEVAL_K,
) -> list[str]:
    """Sort by fusion score descending, then chunk_id, and keep top_k."""

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    return ordered[:top_k]


def retrieval_sources(
    *,
    has_dense: bool,
    has_bm25: bool,
) -> tuple[RetrievalSource, ...]:
    sources: list[RetrievalSource] = []
    if has_dense:
        sources.append("dense")
    if has_bm25:
        sources.append("bm25")
    return tuple(sources)


def lexical_coverages(
    query_terms: frozenset[str],
    chunk_terms: frozenset[str],
    idf: Mapping[str, float],
) -> tuple[float, float] | None:
    """Return (coverage, idf_coverage), or None when the gate must reject.

    OOV query terms stay in the IDF denominator. An empty query or a zero
    IDF sum is a guaranteed reject, not a division.
    """

    if not query_terms:
        return None
    denominator = sum(idf[term] for term in query_terms)
    if denominator == 0.0:
        return None
    overlap = query_terms & chunk_terms
    coverage = len(overlap) / len(query_terms)
    idf_coverage = sum(idf[term] for term in overlap) / denominator
    return coverage, idf_coverage


def is_lexically_strong(
    *,
    bm25_score: float | None,
    coverage: float | None,
    idf_coverage: float | None,
    coverage_min: float,
    idf_coverage_min: float,
) -> bool:
    return (
        bm25_score is not None
        and coverage is not None
        and idf_coverage is not None
        and coverage >= coverage_min
        and idf_coverage >= idf_coverage_min
    )


def pinning_relative_bm25(
    candidate_score: float | None,
    lexical_scores: Sequence[float | None],
) -> float | None:
    """candidate / max BM25 from the lexical arm, or None if undefined."""

    values = [score for score in lexical_scores if score is not None]
    if not values or candidate_score is None:
        return None
    peak = max(values)
    if peak == 0.0:
        return None
    return candidate_score / peak


def pinning_sort_key(
    *,
    idf_coverage: float,
    relative_bm25: float | None,
    fusion_score: float | None,
    chunk_id: str,
) -> tuple[float, float, float, str]:
    """Descending idf_coverage, relative BM25, fusion; then chunk_id.

    Undefined relative BM25 or fusion sorts as lowest priority.
    """

    relative = relative_bm25 if relative_bm25 is not None else float("-inf")
    fusion = fusion_score if fusion_score is not None else float("-inf")
    return (-idf_coverage, -relative, -fusion, chunk_id)
