"""Re-embed stored SQLite chunks into the active Chroma collection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.retrieval.embeddings import EmbeddingProvider
from app.storage.vector_store import ChromaVectorStore, VectorChunk


class VectorReindexError(RuntimeError):
    """Raised when vector reindex cannot complete safely."""


@dataclass(frozen=True, slots=True)
class VectorReindexReport:
    """Count-only summary of a vector reindex run."""

    provider: str
    collection_name: str
    document_count: int
    sqlite_chunk_count: int
    chroma_chunk_count_before: int
    chroma_chunk_count_after: int


def reindex_vectors(
    session: Session,
    vector_store: ChromaVectorStore,
    embeddings: EmbeddingProvider,
    *,
    provider: str,
    overwrite: bool = False,
) -> VectorReindexReport:
    """Write current SQLite chunk text into the active Chroma collection.

    Does not reparse files, and does not modify SQLite or BM25. Only the
    active collection (the one selected by MIR_EMBEDDING_PROVIDER) is ever
    touched; no other collection is read or deleted. With overwrite=True the
    active collection is dropped and rebuilt, so running this under
    MIR_EMBEDDING_PROVIDER=local does rebuild the MiniLM collection.
    """

    sqlite_chunk_count = int(session.scalar(select(func.count()).select_from(Chunk)) or 0)
    chroma_before = vector_store.stats().chunk_count
    if chroma_before > 0 and not overwrite:
        raise VectorReindexError(
            "Active Chroma collection is not empty. Re-run with --overwrite "
            "to replace it. Other collections are left untouched."
        )
    if chroma_before > 0:
        vector_store.reset_collection()

    chunks = list(
        session.scalars(select(Chunk).order_by(Chunk.document_id, Chunk.position))
    )
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)

    for document_id, document_chunks in grouped.items():
        vector_store.upsert_document(
            document_id,
            [
                VectorChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    position=chunk.position,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                )
                for chunk in document_chunks
            ],
            embeddings,
        )

    chroma_after = vector_store.stats().chunk_count
    if chroma_after != sqlite_chunk_count:
        raise VectorReindexError(
            "Chroma chunk count does not match SQLite. Only the active "
            "collection was touched; no other collection was modified."
        )

    return VectorReindexReport(
        provider=provider,
        collection_name=vector_store.collection_name,
        document_count=len(grouped),
        sqlite_chunk_count=sqlite_chunk_count,
        chroma_chunk_count_before=chroma_before,
        chroma_chunk_count_after=chroma_after,
    )
