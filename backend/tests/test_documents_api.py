"""Integration tests for document administration endpoints."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies import get_embedding_provider
from app.core.config import Settings, get_settings
from app.main import create_app
from app.models import Base, Chunk, Document
from app.retrieval.embeddings import EmbeddingProvider
from app.storage.database import create_database_engine, get_database_session


@dataclass
class DocumentApiContext:
    client: TestClient
    session_factory: sessionmaker[Session]
    settings: Settings


class DeterministicTestEmbedder(EmbeddingProvider):
    """Network-free normalized embeddings for API integration tests."""

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * 8
        for token in text.casefold().split():
            vector[sum(ord(character) for character in token) % len(vector)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


def make_pdf_bytes(text: str = "Searchable information retrieval content.") -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    content = pdf.tobytes()
    pdf.close()
    return content


@pytest.fixture
def document_api(tmp_path: Path) -> Iterator[DocumentApiContext]:
    database_path = tmp_path / "api.db"
    database_engine = create_database_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(database_engine)
    session_factory = sessionmaker(
        bind=database_engine,
        autoflush=False,
        expire_on_commit=False,
    )
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        index_dir=tmp_path / "indexes",
        chroma_dir=tmp_path / "chroma",
        database_dir=tmp_path / "database",
        database_url=f"sqlite:///{database_path.as_posix()}",
        chunk_size=20,
        chunk_overlap=5,
    )
    settings.ensure_data_directories()

    def override_database_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    application = create_app()
    application.dependency_overrides[get_database_session] = override_database_session
    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[get_embedding_provider] = lambda: (
        DeterministicTestEmbedder()
    )
    client = TestClient(application)

    yield DocumentApiContext(client, session_factory, settings)

    client.close()
    application.dependency_overrides.clear()
    database_engine.dispose()


def test_upload_list_get_and_delete_document(
    document_api: DocumentApiContext,
) -> None:
    upload_response = document_api.client.post(
        "/api/v1/documents",
        files={
            "file": (
                "lecture.pdf",
                make_pdf_bytes(),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    uploaded = upload_response.json()
    assert uploaded["title"] == "lecture"
    assert uploaded["status"] == "indexed"
    assert uploaded["chunk_count"] == 1
    assert uploaded["keyword_indexed"] is True
    assert uploaded["vector_indexed"] is True

    list_response = document_api.client.get("/api/v1/documents")
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1
    assert list_response.json()["items"][0]["id"] == uploaded["id"]

    detail_response = document_api.client.get(f"/api/v1/documents/{uploaded['id']}")
    assert detail_response.status_code == 200

    with document_api.session_factory() as session:
        stored = session.get(Document, uploaded["id"])
        assert stored is not None
        assert stored.stored_filename is not None
        stored_path = document_api.settings.upload_dir / stored.stored_filename
        assert stored_path.exists()
        assert session.scalar(select(func.count()).select_from(Chunk)) == 1

    delete_response = document_api.client.delete(f"/api/v1/documents/{uploaded['id']}")
    assert delete_response.status_code == 204
    assert not stored_path.exists()

    with document_api.session_factory() as session:
        assert session.get(Document, uploaded["id"]) is None
        assert session.scalar(select(func.count()).select_from(Chunk)) == 0


def test_duplicate_upload_is_rejected(document_api: DocumentApiContext) -> None:
    content = make_pdf_bytes("The same bytes should only be stored once.")

    first_response = document_api.client.post(
        "/api/v1/documents",
        files={"file": ("first.pdf", content, "application/pdf")},
    )
    second_response = document_api.client.post(
        "/api/v1/documents",
        files={"file": ("second.pdf", content, "application/pdf")},
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert (
        second_response.json()["detail"]["existing_document_id"]
        == first_response.json()["id"]
    )
    assert len(list(document_api.settings.upload_dir.glob("*.pdf"))) == 1
    assert not list(document_api.settings.upload_dir.glob("*.part"))


def test_invalid_type_and_missing_document_return_clear_errors(
    document_api: DocumentApiContext,
) -> None:
    invalid_response = document_api.client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    missing_response = document_api.client.get("/api/v1/documents/missing-id")

    assert invalid_response.status_code == 415
    assert missing_response.status_code == 404
    assert not list(document_api.settings.upload_dir.iterdir())


def test_upload_size_limit_is_enforced(document_api: DocumentApiContext) -> None:
    document_api.settings.max_upload_size_mb = 0

    response = document_api.client.post(
        "/api/v1/documents",
        files={"file": ("large.pdf", make_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 413
    assert not list(document_api.settings.upload_dir.iterdir())


def test_uploaded_document_is_searchable_and_removed_from_index(
    document_api: DocumentApiContext,
) -> None:
    upload_response = document_api.client.post(
        "/api/v1/documents",
        files={
            "file": (
                "vector-model.pdf",
                make_pdf_bytes("Cosine similarity ranks vector retrieval results."),
                "application/pdf",
            )
        },
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["id"]

    search_response = document_api.client.get(
        "/api/v1/search/keyword",
        params={"q": "cosine retrieval", "top_k": 5},
    )
    bm25_response = document_api.client.get(
        "/api/v1/search/bm25",
        params={"q": "cosine retrieval", "top_k": 5, "k1": 1.2, "b": 0.6},
    )
    stats_response = document_api.client.get("/api/v1/search/keyword/stats")
    semantic_response = document_api.client.get(
        "/api/v1/search/semantic",
        params={"q": "cosine retrieval", "top_k": 5},
    )
    vector_stats_response = document_api.client.get("/api/v1/search/semantic/stats")

    assert search_response.status_code == 200
    assert search_response.json()["mode"] == "tfidf"
    assert search_response.json()["results"][0]["document_id"] == document_id
    assert search_response.json()["results"][0]["page_start"] == 1
    assert search_response.json()["results"][0]["page_end"] == 1
    assert bm25_response.status_code == 200
    assert bm25_response.json()["mode"] == "bm25"
    assert bm25_response.json()["k1"] == 1.2
    assert bm25_response.json()["b"] == 0.6
    assert bm25_response.json()["results"][0]["document_id"] == document_id
    assert bm25_response.json()["results"][0]["page_start"] == 1
    assert bm25_response.json()["results"][0]["page_end"] == 1
    assert stats_response.json()["document_count"] == 1
    assert stats_response.json()["chunk_count"] == 1
    assert semantic_response.status_code == 200
    assert semantic_response.json()["mode"] == "semantic"
    assert semantic_response.json()["results"][0]["document_id"] == document_id
    assert vector_stats_response.json()["chunk_count"] == 1

    assert (
        document_api.client.delete(f"/api/v1/documents/{document_id}").status_code
        == 204
    )
    after_delete = document_api.client.get(
        "/api/v1/search/keyword",
        params={"q": "cosine retrieval"},
    )
    assert after_delete.json()["results"] == []
    bm25_after_delete = document_api.client.get(
        "/api/v1/search/bm25",
        params={"q": "cosine retrieval"},
    )
    assert bm25_after_delete.json()["results"] == []
    semantic_after_delete = document_api.client.get(
        "/api/v1/search/semantic",
        params={"q": "cosine retrieval"},
    )
    assert semantic_after_delete.json()["results"] == []
