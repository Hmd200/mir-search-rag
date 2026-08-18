"""Classical keyword-search endpoints."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_keyword_index
from app.api.schemas.search import (
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


def _expansion_payload(
    expansion: PrfExpansion | None,
) -> PrfExpansionResponse | None:
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
    return [
        KeywordSearchResult(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            document_title=record.document_title,
            score=record.score,
            text=record.text,
            page_number=record.page_number,
            section_title=record.section_title,
            matched_terms=record.matched_terms,
            term_contributions=record.term_contributions,
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
    candidate_limit: Annotated[int, Query(ge=1, le=5000)] = 200,
    use_prf: Annotated[bool, Query()] = False,
    feedback_docs: Annotated[int | None, Query(ge=1, le=50)] = None,
    max_expansion_terms: Annotated[int | None, Query(ge=0, le=100)] = None,
    expansion_terms: Annotated[int | None, Query(ge=0, le=100)] = None,
    alpha: Annotated[float | None, Query()] = None,
    beta: Annotated[float | None, Query()] = None,
) -> KeywordSearchResponse:
    """Search indexed chunks using inexact top-K and TF-IDF cosine ranking."""

    started = perf_counter()
    resolved_expansion = (
        max_expansion_terms
        if max_expansion_terms is not None
        else expansion_terms
    )
    outcome = KeywordSearchService(session, keyword_index).search(
        query,
        top_k=top_k,
        candidate_limit=candidate_limit,
        use_prf=use_prf,
        feedback_docs=(
            feedback_docs
            if feedback_docs is not None
            else settings.prf_feedback_docs
        ),
        max_expansion_terms=(
            resolved_expansion
            if resolved_expansion is not None
            else settings.prf_max_expansion_terms
        ),
        alpha=settings.prf_alpha if alpha is None else alpha,
        beta=settings.prf_beta if beta is None else beta,
    )
    elapsed_ms = (perf_counter() - started) * 1000
    results = _result_payload(outcome.records)
    return KeywordSearchResponse(
        query=query,
        result_count=len(results),
        elapsed_ms=elapsed_ms,
        results=results,
        expansion=_expansion_payload(outcome.expansion),
    )


@router.get("/bm25", response_model=BM25SearchResponse)
def bm25_search(
    query: Annotated[str, Query(alias="q", min_length=1, max_length=500)],
    session: Annotated[Session, Depends(get_database_session)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
    settings: Annotated[Settings, Depends(get_settings)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
    candidate_limit: Annotated[int, Query(ge=1, le=5000)] = 200,
    k1: Annotated[float, Query(gt=0.0, le=10.0)] = 1.5,
    b: Annotated[float, Query(ge=0.0, le=1.0)] = 0.75,
    use_prf: Annotated[bool, Query()] = False,
    feedback_docs: Annotated[int | None, Query(ge=1, le=50)] = None,
    max_expansion_terms: Annotated[int | None, Query(ge=0, le=100)] = None,
    expansion_terms: Annotated[int | None, Query(ge=0, le=100)] = None,
    alpha: Annotated[float | None, Query()] = None,
    beta: Annotated[float | None, Query()] = None,
) -> BM25SearchResponse:
    """Search indexed chunks using tunable Okapi BM25 ranking."""

    started = perf_counter()
    resolved_expansion = (
        max_expansion_terms
        if max_expansion_terms is not None
        else expansion_terms
    )
    outcome = KeywordSearchService(session, keyword_index).search_bm25(
        query,
        top_k=top_k,
        candidate_limit=candidate_limit,
        k1=k1,
        b=b,
        use_prf=use_prf,
        feedback_docs=(
            feedback_docs
            if feedback_docs is not None
            else settings.prf_feedback_docs
        ),
        max_expansion_terms=(
            resolved_expansion
            if resolved_expansion is not None
            else settings.prf_max_expansion_terms
        ),
        alpha=settings.prf_alpha if alpha is None else alpha,
        beta=settings.prf_beta if beta is None else beta,
    )
    elapsed_ms = (perf_counter() - started) * 1000
    results = _result_payload(outcome.records)
    return BM25SearchResponse(
        query=query,
        k1=k1,
        b=b,
        result_count=len(results),
        elapsed_ms=elapsed_ms,
        results=results,
        expansion=_expansion_payload(outcome.expansion),
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
