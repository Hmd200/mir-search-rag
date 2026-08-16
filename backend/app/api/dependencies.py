"""Reusable FastAPI dependency providers."""

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.retrieval.embeddings import (
    EmbeddingProvider,
    open_embedding_provider,
)
from app.storage.keyword_index import KeywordIndex, open_keyword_index
from app.storage.vector_store import (
    ChromaVectorStore,
    open_vector_store,
)


def get_keyword_index(
    settings: Annotated[Settings, Depends(get_settings)],
) -> KeywordIndex:
    """Return the persistent custom index for the configured data directory."""

    index_path = (settings.index_dir / "keyword-index.json").resolve()
    return open_keyword_index(str(index_path))


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider:
    """Return the configured lazy local sentence-transformer."""

    return open_embedding_provider(
        settings.embedding_model,
        str(settings.model_dir.resolve()),
        settings.embedding_device,
        settings.embedding_batch_size,
    )


def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChromaVectorStore:
    """Return the persistent Chroma collection for searchable chunks."""

    return open_vector_store(
        str(settings.chroma_dir.resolve()),
        settings.vector_collection_name,
    )
