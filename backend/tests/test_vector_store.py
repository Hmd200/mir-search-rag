"""Tests for persistent Chroma vector storage and semantic retrieval."""

import math
from pathlib import Path

from app.retrieval.embeddings import EmbeddingProvider
from app.storage.vector_store import ChromaVectorStore, VectorChunk


class KeywordVectorizer(EmbeddingProvider):
    """Small deterministic embedder for local, network-free tests."""

    vocabulary = ("retrieval", "database", "neural")

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        lowered = text.casefold()
        vector = [float(lowered.count(term)) for term in cls.vocabulary]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return [0.0 for _ in vector]
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


def test_vector_store_upserts_searches_and_persists(tmp_path: Path) -> None:
    store_path = tmp_path / "chroma"
    embedder = KeywordVectorizer()
    store = ChromaVectorStore(store_path, "test_collection")
    store.upsert_document(
        "document-a",
        [
            VectorChunk(
                chunk_id="chunk-a",
                document_id="document-a",
                text="neural neural retrieval",
                position=0,
                page_start=2,
                page_end=2,
            )
        ],
        embedder,
    )
    store.upsert_document(
        "document-b",
        [
            VectorChunk(
                chunk_id="chunk-b",
                document_id="document-b",
                text="database storage",
                position=0,
            )
        ],
        embedder,
    )

    hits = store.search("neural retrieval", embedder, top_k=2)

    assert hits[0].chunk_id == "chunk-a"
    assert hits[0].score > hits[1].score
    assert store.stats().chunk_count == 2

    reloaded = ChromaVectorStore(store_path, "test_collection")
    assert reloaded.search("database", embedder, top_k=1)[0].chunk_id == "chunk-b"


def test_vector_upsert_replaces_stale_chunks_and_delete_is_scoped(
    tmp_path: Path,
) -> None:
    embedder = KeywordVectorizer()
    store = ChromaVectorStore(tmp_path / "chroma", "replacement")
    store.upsert_document(
        "document-a",
        [
            VectorChunk("old-a", "document-a", "neural", 0),
            VectorChunk("old-b", "document-a", "retrieval", 1),
        ],
        embedder,
    )
    store.upsert_document(
        "document-b",
        [VectorChunk("other", "document-b", "database", 0)],
        embedder,
    )

    store.upsert_document(
        "document-a",
        [VectorChunk("new-a", "document-a", "neural retrieval", 0)],
        embedder,
    )

    assert store.stats().chunk_count == 2
    assert {hit.chunk_id for hit in store.search("neural", embedder)} == {
        "new-a",
        "other",
    }
    assert store.delete_document("document-a") is True
    assert store.delete_document("document-a") is False
    assert store.stats().chunk_count == 1
    assert store.search("database", embedder)[0].chunk_id == "other"
