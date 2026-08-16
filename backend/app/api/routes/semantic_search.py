"""Semantic vector-search endpoints."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_embedding_provider, get_vector_store
from app.api.schemas.semantic_search import (
    SemanticSearchResponse,
    SemanticSearchResult,
    VectorStoreStatsResponse,
)
from app.retrieval.embeddings import EmbeddingError, EmbeddingProvider
from app.services.semantic_search import SemanticSearchService
from app.storage.database import get_database_session
from app.storage.vector_store import ChromaVectorStore, VectorStoreError

router = APIRouter(prefix="/search")


@router.get("/semantic", response_model=SemanticSearchResponse)
def semantic_search(
    query: Annotated[str, Query(alias="q", min_length=1, max_length=500)],
    session: Annotated[Session, Depends(get_database_session)],
    vector_store: Annotated[ChromaVectorStore, Depends(get_vector_store)],
    embeddings: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> SemanticSearchResponse:
    """Search Chroma's HNSW index using local sentence embeddings."""

    started = perf_counter()
    try:
        records = SemanticSearchService(
            session,
            vector_store,
            embeddings,
        ).search(query, top_k=top_k)
    except (EmbeddingError, VectorStoreError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search is temporarily unavailable.",
        ) from error

    elapsed_ms = (perf_counter() - started) * 1000
    results = [
        SemanticSearchResult(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            document_title=record.document_title,
            score=record.score,
            distance=record.distance,
            text=record.text,
            page_number=record.page_number,
            section_title=record.section_title,
        )
        for record in records
    ]
    return SemanticSearchResponse(
        query=query,
        result_count=len(results),
        elapsed_ms=elapsed_ms,
        results=results,
    )


@router.get("/semantic/stats", response_model=VectorStoreStatsResponse)
def vector_store_stats(
    vector_store: Annotated[ChromaVectorStore, Depends(get_vector_store)],
) -> VectorStoreStatsResponse:
    """Expose the current number of vectors in Chroma."""

    stats = vector_store.stats()
    return VectorStoreStatsResponse(chunk_count=stats.chunk_count)
