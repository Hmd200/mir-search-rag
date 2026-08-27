"""Semantic retrieval service joining Chroma hits with stored metadata."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.retrieval.embeddings import EmbeddingProvider
from app.storage.vector_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class SemanticSearchRecord:
    """A vector hit enriched with source and citation metadata."""

    chunk_id: str
    document_id: str
    document_title: str
    score: float
    distance: float
    text: str
    page_start: int | None
    page_end: int | None
    section_title: str | None


class SemanticSearchService:
    """Run Chroma HNSW retrieval and hydrate ranked chunks from SQLite."""

    def __init__(
        self,
        session: Session,
        vector_store: ChromaVectorStore,
        embeddings: EmbeddingProvider,
    ) -> None:
        self.session = session
        self.vector_store = vector_store
        self.embeddings = embeddings

    def search(self, query: str, *, top_k: int) -> list[SemanticSearchRecord]:
        """Embed the query and return the nearest stored chunks.

        Uses the same active embedding provider that indexed the documents, so
        query and document vectors always share one space.
        """

        hits = self.vector_store.search(
            query,
            self.embeddings,
            top_k=top_k,
        )
        if not hits:
            return []

        chunks = {
            chunk.id: chunk
            for chunk in self.session.scalars(
                select(Chunk).where(Chunk.id.in_([hit.chunk_id for hit in hits]))
            )
        }
        results: list[SemanticSearchRecord] = []
        for hit in hits:
            chunk = chunks.get(hit.chunk_id)
            if chunk is None:
                continue
            results.append(
                SemanticSearchRecord(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=chunk.document.title,
                    score=hit.score,
                    distance=hit.distance,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                )
            )
        return results

    def records_for_ids(
        self,
        chunk_ids: list[str],
    ) -> dict[str, SemanticSearchRecord]:
        """Hydrate stored chunks without inventing a dense retrieval score."""

        if not chunk_ids:
            return {}

        chunks = {
            chunk.id: chunk
            for chunk in self.session.scalars(
                select(Chunk).where(Chunk.id.in_(chunk_ids))
            )
        }
        records: dict[str, SemanticSearchRecord] = {}
        for chunk_id in chunk_ids:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            records[chunk_id] = SemanticSearchRecord(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=chunk.document.title,
                # Placeholder only; callers must not treat this as a dense hit.
                score=0.0,
                distance=1.0,
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
            )
        return records
