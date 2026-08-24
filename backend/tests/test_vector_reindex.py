"""Tests for safe vector reindex and Gemini/MiniLM collection isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.models import Chunk, Document, SourceType
from app.services.vector_reindex import VectorReindexError, reindex_vectors
from app.storage.vector_store import (
    ChromaVectorStore,
    VectorChunk,
    VectorStoreStats,
)
from tests.test_database import make_test_session
from tests.test_vector_store import KeywordVectorizer


class _CountMismatchStore:
    """Writes vectors but reports a count that cannot match SQLite."""

    def __init__(self, inner: ChromaVectorStore) -> None:
        self._inner = inner
        self.collection_name = inner.collection_name
        self.reset_calls = 0

    def stats(self) -> VectorStoreStats:
        return VectorStoreStats(chunk_count=0)

    def upsert_document(
        self,
        document_id: str,
        chunks: object,
        embeddings: object,
    ) -> None:
        self._inner.upsert_document(document_id, chunks, embeddings)  # type: ignore[arg-type]

    def reset_collection(self) -> None:
        self.reset_calls += 1
        self._inner.reset_collection()


def _add_chunked_document(
    session: Session,
    *,
    document_id: str,
    chunk_id: str,
    text: str,
    position: int = 0,
    page_start: int | None = 2,
    page_end: int | None = 3,
    section_title: str | None = "Scoring",
) -> None:
    document = Document(
        id=document_id,
        title="IR notes",
        source_type=SourceType.UPLOAD,
        chunk_count=1,
        vector_indexed=True,
        keyword_indexed=True,
    )
    document.chunks.append(
        Chunk(
            id=chunk_id,
            position=position,
            text=text,
            token_count=len(text.split()),
            page_start=page_start,
            page_end=page_end,
            section_title=section_title,
            char_start=0,
            char_end=len(text),
        )
    )
    session.add(document)
    session.commit()


def test_gemini_and_local_collections_are_isolated(tmp_path: Path) -> None:
    embedder = KeywordVectorizer()
    local = ChromaVectorStore(tmp_path / "chroma", "mir_chunks")
    gemini = ChromaVectorStore(tmp_path / "chroma", "mir_chunks_gemini_001_768")
    local.upsert_document(
        "document-a",
        [VectorChunk("chunk-local", "document-a", "neural retrieval", 0)],
        embedder,
    )

    assert local.stats().chunk_count == 1
    assert gemini.stats().chunk_count == 0

    gemini.upsert_document(
        "document-a",
        [VectorChunk("chunk-gemini", "document-a", "neural retrieval", 0)],
        embedder,
    )
    gemini.reset_collection()

    assert local.stats().chunk_count == 1
    assert local.listed_chunk_ids() == ["chunk-local"]
    assert gemini.stats().chunk_count == 0


def test_reindex_preserves_ids_and_metadata(tmp_path: Path) -> None:
    session, engine = make_test_session(tmp_path / "reindex.db")
    _add_chunked_document(
        session,
        document_id="document-a",
        chunk_id="chunk-a",
        text="neural retrieval database",
        position=4,
        page_start=2,
        page_end=3,
        section_title="BM25",
    )
    sqlite_count = 1
    local = ChromaVectorStore(tmp_path / "chroma", "mir_chunks")
    gemini = ChromaVectorStore(tmp_path / "chroma", "mir_chunks_gemini_001_768")
    embedder = KeywordVectorizer()
    local.upsert_document(
        "document-a",
        [
            VectorChunk(
                "chunk-a",
                "document-a",
                "neural retrieval database",
                4,
                2,
                3,
                "BM25",
            )
        ],
        embedder,
    )

    report = reindex_vectors(
        session,
        gemini,
        embedder,
        provider="gemini",
        overwrite=False,
    )

    assert report.sqlite_chunk_count == sqlite_count
    assert report.chroma_chunk_count_after == sqlite_count
    assert gemini.listed_chunk_ids() == ["chunk-a"]
    metadata = gemini.get_metadatas(["chunk-a"])["chunk-a"]
    assert metadata["document_id"] == "document-a"
    assert metadata["position"] == 4
    assert metadata["page_start"] == 2
    assert metadata["page_end"] == 3
    assert metadata["section_title"] == "BM25"
    assert local.listed_chunk_ids() == ["chunk-a"]
    session.close()
    engine.dispose()


def test_reindex_count_mismatch_is_fail_safe(tmp_path: Path) -> None:
    session, engine = make_test_session(tmp_path / "mismatch.db")
    _add_chunked_document(
        session,
        document_id="document-a",
        chunk_id="chunk-a",
        text="neural retrieval",
    )
    embedder = KeywordVectorizer()
    local = ChromaVectorStore(tmp_path / "chroma", "mir_chunks")
    gemini = ChromaVectorStore(tmp_path / "chroma", "mir_chunks_gemini_001_768")
    local.upsert_document(
        "document-a",
        [VectorChunk("chunk-a", "document-a", "neural retrieval", 0, 2, 3, "Scoring")],
        embedder,
    )
    lying = _CountMismatchStore(gemini)

    with pytest.raises(VectorReindexError, match="does not match SQLite") as exc:
        reindex_vectors(
            session,
            lying,  # type: ignore[arg-type]
            embedder,
            provider="gemini",
            overwrite=False,
        )

    assert "chunk-a" not in str(exc.value)
    assert "neural retrieval" not in str(exc.value)
    assert local.stats().chunk_count == 1
    assert local.listed_chunk_ids() == ["chunk-a"]
    session.close()
    engine.dispose()


def test_reindex_refuses_nonempty_target_without_overwrite(tmp_path: Path) -> None:
    session, engine = make_test_session(tmp_path / "nonempty.db")
    _add_chunked_document(
        session,
        document_id="document-a",
        chunk_id="chunk-a",
        text="neural retrieval",
    )
    embedder = KeywordVectorizer()
    local = ChromaVectorStore(tmp_path / "chroma", "mir_chunks")
    gemini = ChromaVectorStore(tmp_path / "chroma", "mir_chunks_gemini_001_768")
    local.upsert_document(
        "document-a",
        [VectorChunk("chunk-a", "document-a", "neural retrieval", 0)],
        embedder,
    )
    gemini.upsert_document(
        "document-a",
        [VectorChunk("stale", "document-a", "neural retrieval", 0)],
        embedder,
    )

    with pytest.raises(VectorReindexError, match="not empty"):
        reindex_vectors(
            session,
            gemini,
            embedder,
            provider="gemini",
            overwrite=False,
        )

    assert local.listed_chunk_ids() == ["chunk-a"]
    assert gemini.listed_chunk_ids() == ["stale"]

    report = reindex_vectors(
        session,
        gemini,
        embedder,
        provider="gemini",
        overwrite=True,
    )
    assert report.chroma_chunk_count_after == 1
    assert gemini.listed_chunk_ids() == ["chunk-a"]
    assert local.listed_chunk_ids() == ["chunk-a"]
    session.close()
    engine.dispose()
