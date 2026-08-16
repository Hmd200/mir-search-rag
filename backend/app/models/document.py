"""Document and chunk persistence models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


def new_uuid() -> str:
    """Return a portable UUID string for database and index identifiers."""

    return str(uuid4())


class SourceType(str, Enum):
    """Ways a document can enter the collection."""

    UPLOAD = "upload"
    WEB = "web"


class DocumentStatus(str, Enum):
    """Document processing lifecycle states."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(Base):
    """A source document tracked across both retrieval indexes."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "file_size_bytes IS NULL OR file_size_bytes >= 0",
            name="ck_documents_file_size_nonnegative",
        ),
        CheckConstraint(
            "chunk_count >= 0",
            name="ck_documents_chunk_count_nonnegative",
        ),
        Index("ix_documents_status_created", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    title: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str | None] = mapped_column(String(500))
    stored_filename: Mapped[str | None] = mapped_column(String(500), unique=True)
    source_type: Mapped[SourceType] = mapped_column(
        SqlEnum(
            SourceType,
            values_callable=lambda enum: [item.value for item in enum],
            native_enum=False,
            validate_strings=True,
        ),
        default=SourceType.UPLOAD,
        index=True,
    )
    source_url: Mapped[str | None] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[DocumentStatus] = mapped_column(
        SqlEnum(
            DocumentStatus,
            values_callable=lambda enum: [item.value for item in enum],
            native_enum=False,
            validate_strings=True,
        ),
        default=DocumentStatus.PENDING,
        index=True,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    keyword_indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    vector_indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.position",
    )


class Chunk(Base):
    """A stable citation and retrieval unit derived from one document."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "position", name="uq_chunks_document_position"),
        CheckConstraint("position >= 0", name="ck_chunks_position_nonnegative"),
        CheckConstraint("token_count >= 0", name="ck_chunks_token_count_nonnegative"),
        CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="ck_chunks_page_number_positive",
        ),
        CheckConstraint("char_start >= 0", name="ck_chunks_char_start_nonnegative"),
        CheckConstraint("char_end >= char_start", name="ck_chunks_char_range"),
        CheckConstraint("length(text) > 0", name="ck_chunks_text_not_empty"),
        Index("ix_chunks_document_page", "document_id", "page_number"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(String(500))
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
