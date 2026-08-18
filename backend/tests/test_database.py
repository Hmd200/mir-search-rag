"""Persistence tests for document metadata and chunks."""

from pathlib import Path

from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Chunk, Document, DocumentStatus, SourceType
from app.storage.database import create_database_engine


def make_test_session(database_path: Path) -> tuple[Session, Engine]:
    database_engine = create_database_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(database_engine)
    session_factory = sessionmaker(
        bind=database_engine,
        autoflush=False,
        expire_on_commit=False,
    )
    return session_factory(), database_engine


def test_document_and_chunk_round_trip(tmp_path: Path) -> None:
    session, database_engine = make_test_session(tmp_path / "round-trip.db")
    document = Document(
        title="Information Retrieval Notes",
        original_filename="notes.pdf",
        stored_filename="document-id.pdf",
        source_type=SourceType.UPLOAD,
        mime_type="application/pdf",
        file_size_bytes=2048,
        sha256="a" * 64,
    )
    document.chunks.append(
        Chunk(
            position=0,
            text="A searchable passage with stable citation metadata.",
            token_count=8,
            page_start=3,
            page_end=3,
            char_start=0,
            char_end=51,
        )
    )

    session.add(document)
    session.commit()
    session.expire_all()

    stored_document = session.scalar(select(Document))
    assert stored_document is not None
    assert stored_document.status is DocumentStatus.PENDING
    assert stored_document.source_type is SourceType.UPLOAD
    assert stored_document.keyword_indexed is False
    assert stored_document.vector_indexed is False
    assert stored_document.chunks[0].page_start == 3
    assert stored_document.chunks[0].page_end == 3
    assert len(stored_document.chunks[0].id) == 36

    session.close()
    database_engine.dispose()


def test_database_cascade_deletes_chunks(tmp_path: Path) -> None:
    session, database_engine = make_test_session(tmp_path / "cascade.db")
    document = Document(title="Delete Me", source_type=SourceType.WEB)
    document.chunks.append(
        Chunk(
            position=0,
            text="This chunk must be removed with its parent.",
            token_count=9,
            char_start=0,
            char_end=43,
        )
    )
    session.add(document)
    session.commit()

    session.execute(delete(Document).where(Document.id == document.id))
    session.commit()

    remaining_chunks = session.scalar(select(func.count()).select_from(Chunk))
    assert remaining_chunks == 0

    session.close()
    database_engine.dispose()
