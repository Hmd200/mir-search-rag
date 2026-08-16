"""Retrieval-augmented generation endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_embedding_provider, get_vector_store
from app.api.schemas.rag import RagCitedChunk, RagRequest, RagResponse
from app.core.config import Settings, get_settings
from app.retrieval.embeddings import EmbeddingError, EmbeddingProvider
from app.retrieval.llm import LLMError, create_llm_client
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
    )


@router.post("/rag", response_model=RagResponse)
def rag_search(
    payload: RagRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> RagResponse:
    """Retrieve semantic context and generate a cited answer."""

    try:
        outcome = service.generate(
            payload.query,
            top_k=payload.top_k,
        )
    except LLMError as error:
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
                page_number=chunk.page_number,
                text=chunk.text,
                score=chunk.score,
            )
            for chunk in outcome.cited_chunks
        ],
        invalid_citations=list(outcome.invalid_citations),
        abstained=outcome.abstained,
        elapsed_ms=outcome.elapsed_ms,
    )
