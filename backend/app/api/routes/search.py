"""Classical keyword-search endpoints."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_keyword_index
from app.api.schemas.search import (
    Bm25Mode,
    BM25SearchResponse,
    KeywordIndexStatsResponse,
    KeywordSearchResponse,
    KeywordSearchResult,
    PrfAddedTermResponse,
    PrfExpansionResponse,
)
from app.core.config import Settings, get_settings
from app.services.search import KeywordSearchService
from app.storage.database import get_database_session
from app.storage.keyword_index import KeywordIndex, PrfExpansion

router = APIRouter(prefix="/search")

_BM25_STANDARD_K1 = 1.5
_BM25_STANDARD_B = 0.75


def _resolve_bm25_parameters(
    mode: Bm25Mode | None,
    request_k1: float,
    request_b: float,
    settings: Settings,
) -> tuple[Bm25Mode | None, float, float]:
    """Return the selected mode and the k1/b that will actually score."""

    if mode is None:
        return None, request_k1, request_b
    if mode == "default":
        return "default", _BM25_STANDARD_K1, _BM25_STANDARD_B
    if mode == "tunable":
        return "tunable", request_k1, request_b
    return "finetuned", settings.bm25_finetuned_k1, settings.bm25_finetuned_b


def _expansion_payload(
    expansion: PrfExpansion | None,
) -> PrfExpansionResponse | None:
    """Convert Rocchio expansion terms into the API's chip payload."""

    if expansion is None:
        return None
    return PrfExpansionResponse(
        added_terms=[
            PrfAddedTermResponse(term=item.term, weight=item.weight)
            for item in expansion.added_terms
        ],
        feedback_chunk_ids=list(expansion.feedback_chunk_ids),
    )


def _result_payload(records: list) -> list[KeywordSearchResult]:
    """Convert one ranked chunk into its search-result response model."""

    return [
        KeywordSearchResult(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            document_title=record.document_title,
            score=record.score,
            text=record.text,
            page_start=record.page_start,
            page_end=record.page_end,
            section_title=record.section_title,
            matched_terms=record.matched_terms,
            term_contributions=record.term_contributions,
            retrieval_score=record.retrieval_score,
            rerank_score=record.rerank_score,
        )
        for record in records
    ]


@router.get("/keyword", response_model=KeywordSearchResponse)
def keyword_search(
    query: Annotated[str, Query(alias="q", min_length=1, max_length=500)],
    session: Annotated[Session, Depends(get_database_session)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
    settings: Annotated[Settings, Depends(get_settings)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
    use_prf: Annotated[bool, Query()] = False,
    feedback_docs: Annotated[int | None, Query(ge=1, le=50)] = None,
    max_expansion_terms: Annotated[int | None, Query(ge=0, le=100)] = None,
    expansion_terms: Annotated[int | None, Query(ge=0, le=100)] = None,
    alpha: Annotated[float | None, Query()] = None,
    beta: Annotated[float | None, Query()] = None,
    use_reranker: Annotated[bool | None, Query()] = None,
) -> KeywordSearchResponse:
    """Search with champion-list inexact top-K and TF-IDF cosine ranking."""

    started = perf_counter()
    resolved_expansion = (
        max_expansion_terms if max_expansion_terms is not None else expansion_terms
    )
    rerank_enabled = (
        settings.rerank_enabled_default if use_reranker is None else use_reranker
    )
    outcome = KeywordSearchService(session, keyword_index).search(
        query,
        top_k=top_k,
        use_prf=use_prf,
        feedback_docs=(
            feedback_docs if feedback_docs is not None else settings.prf_feedback_docs
        ),
        max_expansion_terms=(
            resolved_expansion
            if resolved_expansion is not None
            else settings.prf_max_expansion_terms
        ),
        alpha=settings.prf_alpha if alpha is None else alpha,
        beta=settings.prf_beta if beta is None else beta,
        use_reranker=rerank_enabled,
    )
    elapsed_ms = (perf_counter() - started) * 1000
    results = _result_payload(outcome.records)
    return KeywordSearchResponse(
        query=query,
        result_count=len(results),
        elapsed_ms=elapsed_ms,
        results=results,
        expansion=_expansion_payload(outcome.expansion),
        reranked=outcome.reranked,
    )


@router.get("/bm25", response_model=BM25SearchResponse)
def bm25_search(
    query: Annotated[str, Query(alias="q", min_length=1, max_length=500)],
    session: Annotated[Session, Depends(get_database_session)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
    settings: Annotated[Settings, Depends(get_settings)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
    bm25_mode: Annotated[Bm25Mode | None, Query()] = None,
    k1: Annotated[float, Query(gt=0.0, le=10.0)] = _BM25_STANDARD_K1,
    b: Annotated[float, Query(ge=0.0, le=1.0)] = _BM25_STANDARD_B,
    use_prf: Annotated[bool, Query()] = False,
    feedback_docs: Annotated[int | None, Query(ge=1, le=50)] = None,
    max_expansion_terms: Annotated[int | None, Query(ge=0, le=100)] = None,
    expansion_terms: Annotated[int | None, Query(ge=0, le=100)] = None,
    alpha: Annotated[float | None, Query()] = None,
    beta: Annotated[float | None, Query()] = None,
    use_reranker: Annotated[bool | None, Query()] = None,
) -> BM25SearchResponse:
    """Search indexed chunks using Okapi BM25 in default, tunable, or calibrated mode."""

    started = perf_counter()
    resolved_mode, effective_k1, effective_b = _resolve_bm25_parameters(
        bm25_mode,
        k1,
        b,
        settings,
    )
    resolved_expansion = (
        max_expansion_terms if max_expansion_terms is not None else expansion_terms
    )
    rerank_enabled = (
        settings.rerank_enabled_default if use_reranker is None else use_reranker
    )
    outcome = KeywordSearchService(session, keyword_index).search_bm25(
        query,
        top_k=top_k,
        k1=effective_k1,
        b=effective_b,
        use_prf=use_prf,
        feedback_docs=(
            feedback_docs if feedback_docs is not None else settings.prf_feedback_docs
        ),
        max_expansion_terms=(
            resolved_expansion
            if resolved_expansion is not None
            else settings.prf_max_expansion_terms
        ),
        alpha=settings.prf_alpha if alpha is None else alpha,
        beta=settings.prf_beta if beta is None else beta,
        use_reranker=rerank_enabled,
    )
    elapsed_ms = (perf_counter() - started) * 1000
    results = _result_payload(outcome.records)
    return BM25SearchResponse(
        query=query,
        bm25_mode=resolved_mode,
        k1=effective_k1,
        b=effective_b,
        result_count=len(results),
        elapsed_ms=elapsed_ms,
        results=results,
        expansion=_expansion_payload(outcome.expansion),
        reranked=outcome.reranked,
    )


@router.get("/keyword/stats", response_model=KeywordIndexStatsResponse)
def keyword_index_stats(
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
) -> KeywordIndexStatsResponse:
    """Expose collection statistics for the future visualization dashboard."""

    stats = keyword_index.stats()
    return KeywordIndexStatsResponse(
        document_count=stats.document_count,
        chunk_count=stats.chunk_count,
        vocabulary_size=stats.vocabulary_size,
        posting_count=stats.posting_count,
    )
