"""Classical keyword-search endpoints."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_keyword_index
from app.api.schemas.search import (
    KeywordIndexStatsResponse,
    KeywordSearchResponse,
    KeywordSearchResult,
)
from app.services.search import KeywordSearchService
from app.storage.database import get_database_session
from app.storage.keyword_index import KeywordIndex

router = APIRouter(prefix="/search")


@router.get("/keyword", response_model=KeywordSearchResponse)
def keyword_search(
    query: Annotated[str, Query(alias="q", min_length=1, max_length=500)],
    session: Annotated[Session, Depends(get_database_session)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
    candidate_limit: Annotated[int, Query(ge=1, le=5000)] = 200,
) -> KeywordSearchResponse:
    """Search indexed chunks using inexact top-K and TF-IDF cosine ranking."""

    started = perf_counter()
    records = KeywordSearchService(session, keyword_index).search(
        query,
        top_k=top_k,
        candidate_limit=candidate_limit,
    )
    elapsed_ms = (perf_counter() - started) * 1000
    results = [
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
    return KeywordSearchResponse(
        query=query,
        result_count=len(results),
        elapsed_ms=elapsed_ms,
        results=results,
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
