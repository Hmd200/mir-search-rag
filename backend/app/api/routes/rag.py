"""Retrieval-augmented generation endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_embedding_provider, get_vector_store
from app.api.schemas.rag import RagCitedChunk, RagRequest, RagResponse
from app.core.config import Settings, get_settings
from app.retrieval.embeddings import EmbeddingError, EmbeddingProvider
from app.retrieval.llm import LLMError, create_llm_client, resolve_llm_provider
from app.services.rag import RagService
from app.services.semantic_search import SemanticSearchService
from app.storage.database import get_database_session
from app.storage.vector_store import ChromaVectorStore, VectorStoreError

router = APIRouter(prefix="/search")


def get_rag_service(
    session: Annotated[Session, Depends(get_database_session)],
    vector_store: Annotated[ChromaVectorStore, Depends(get_vector_store)],
    embeddings: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RagService:
    """Build a RAG service from request-scoped search and LLM clients."""

    return RagService(
        SemanticSearchService(
            session,
            vector_store,
            embeddings,
        ),
        create_llm_client(settings),
        min_retrieval_score=settings.rag_min_retrieval_score,
    )


@router.post("/rag", response_model=RagResponse)
def rag_search(
    payload: RagRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RagResponse:
    """Retrieve semantic context and generate a cited answer."""

    try:
        provider = resolve_llm_provider(settings, payload.llm_provider)
        if payload.llm_provider is not None:
            service = service.with_llm(
                create_llm_client(settings, provider=provider),
            )
        outcome = service.generate(
            payload.query,
            top_k=payload.top_k,
            use_reranker=payload.use_reranker,
            use_query_rewrite=payload.use_query_rewrite,
        )
    except LLMError as error:
        detail = str(error)
        if "not configured" in detail.lower() or "unsupported" in detail.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail,
            ) from error
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is unreachable.",
        ) from error
    except (EmbeddingError, VectorStoreError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search is temporarily unavailable.",
        ) from error

    return RagResponse(
        query=outcome.query,
        answer=outcome.answer,
        cited_chunks=[
            RagCitedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                score=chunk.score,
                retrieval_score=chunk.retrieval_score,
                rerank_score=chunk.rerank_score,
            )
            for chunk in outcome.cited_chunks
        ],
        invalid_citations=list(outcome.invalid_citations),
        abstained=outcome.abstained,
        elapsed_ms=outcome.elapsed_ms,
        rewritten_query=outcome.rewritten_query,
        llm_provider=provider,
        citation_enforced=outcome.citation_enforced,
        abstention_reason=outcome.abstention_reason,
        context_chunks=[
            RagCitedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                score=chunk.score,
                retrieval_score=chunk.retrieval_score,
                rerank_score=chunk.rerank_score,
                prompt_index=index,
            )
            for index, chunk in enumerate(outcome.context_chunks, start=1)
        ],
    )
