"""Persistent Chroma adapter for chunk embeddings and semantic retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from app.retrieval.embeddings import EmbeddingProvider


class VectorStoreError(RuntimeError):
    """Raised when Chroma cannot update or query the vector collection."""


@dataclass(frozen=True, slots=True)
class VectorChunk:
    """A chunk and its citation metadata ready for vector indexing."""

    chunk_id: str
    document_id: str
    text: str
    position: int
    page_number: int | None = None
    section_title: str | None = None


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """A Chroma cosine-distance result normalized to a similarity score."""

    chunk_id: str
    score: float
    distance: float


@dataclass(frozen=True, slots=True)
class VectorStoreStats:
    """Collection statistics exposed to the administration interface."""

    chunk_count: int


class ChromaVectorStore:
    """Thread-safe persistent Chroma collection using supplied embeddings."""

    def __init__(self, path: str | Path, collection_name: str) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        try:
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection: Collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise VectorStoreError("Could not initialize the Chroma store.") from error

    @staticmethod
    def _metadata(chunk: VectorChunk) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "document_id": chunk.document_id,
            "position": chunk.position,
        }
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        if chunk.section_title:
            metadata["section_title"] = chunk.section_title
        return metadata

    def upsert_document(
        self,
        document_id: str,
        chunks: Iterable[VectorChunk],
        embeddings: EmbeddingProvider,
    ) -> None:
        """Embed and upsert all chunks, then remove stale IDs for the document."""

        chunk_values = list(chunks)
        if not chunk_values:
            raise VectorStoreError("Cannot vector-index a document without chunks.")
        if any(chunk.document_id != document_id for chunk in chunk_values):
            raise VectorStoreError("Every vector chunk must belong to the document.")
        chunk_ids = [chunk.chunk_id for chunk in chunk_values]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise VectorStoreError("Vector chunk IDs must be unique.")

        try:
            vectors = embeddings.embed_documents([chunk.text for chunk in chunk_values])
            if len(vectors) != len(chunk_values):
                raise VectorStoreError(
                    "Embedding count does not match the chunk count."
                )

            with self._lock:
                existing = self._collection.get(
                    where={"document_id": document_id},
                    include=[],
                )
                existing_ids = set(existing.get("ids", []))
                self._collection.upsert(
                    ids=chunk_ids,
                    embeddings=np.asarray(vectors, dtype=np.float32),
                    documents=[chunk.text for chunk in chunk_values],
                    metadatas=[self._metadata(chunk) for chunk in chunk_values],
                )
                stale_ids = sorted(existing_ids.difference(chunk_ids))
                if stale_ids:
                    self._collection.delete(ids=stale_ids)
        except VectorStoreError:
            raise
        except Exception as error:
            raise VectorStoreError(
                f"Could not vector-index document: {document_id}"
            ) from error

    def delete_document(self, document_id: str) -> bool:
        """Remove all vectors for a source document."""

        try:
            with self._lock:
                existing = self._collection.get(
                    where={"document_id": document_id},
                    include=[],
                )
                ids = existing.get("ids", [])
                if not ids:
                    return False
                self._collection.delete(ids=ids)
                return True
        except Exception as error:
            raise VectorStoreError(
                f"Could not remove vectors for document: {document_id}"
            ) from error

    def search(
        self,
        query: str,
        embeddings: EmbeddingProvider,
        *,
        top_k: int = 10,
    ) -> list[VectorSearchHit]:
        """Return nearest chunks using cosine distance in Chroma's HNSW index."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        try:
            with self._lock:
                collection_size = self._collection.count()
                if collection_size == 0:
                    return []
                query_vector = embeddings.embed_query(query)
                response = self._collection.query(
                    query_embeddings=np.asarray([query_vector], dtype=np.float32),
                    n_results=min(top_k, collection_size),
                    include=["distances"],
                )

            ids = response.get("ids") or [[]]
            distances = response.get("distances") or [[]]
            hits = [
                VectorSearchHit(
                    chunk_id=chunk_id,
                    distance=float(distance),
                    score=max(0.0, min(1.0, 1.0 - float(distance))),
                )
                for chunk_id, distance in zip(ids[0], distances[0], strict=True)
            ]
            hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
            return hits
        except ValueError:
            raise
        except Exception as error:
            raise VectorStoreError("Could not run semantic search.") from error

    def stats(self) -> VectorStoreStats:
        """Return the current number of embedded chunks."""

        try:
            with self._lock:
                return VectorStoreStats(chunk_count=self._collection.count())
        except Exception as error:
            raise VectorStoreError("Could not read Chroma statistics.") from error


@lru_cache(maxsize=32)
def open_vector_store(path: str, collection_name: str) -> ChromaVectorStore:
    """Reuse one Chroma client per configured path and collection."""

    return ChromaVectorStore(Path(path), collection_name)
