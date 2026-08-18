"""Document upload, persistence, listing, and deletion workflows."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Chunk, Document, DocumentStatus, SourceType
from app.processing import (
    DocumentChunker,
    DocumentProcessingError,
    ExtractedDocument,
    extract_document,
)
from app.processing.extractors import extract_from_url
from app.retrieval.embeddings import EmbeddingProvider
from app.storage.keyword_index import KeywordIndex, KeywordIndexError
from app.storage.vector_store import ChromaVectorStore, VectorChunk, VectorStoreError

_SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
_COPY_BUFFER_SIZE = 1024 * 1024


class DocumentServiceError(RuntimeError):
    """Base error for document-management operations."""


class UnsupportedUploadError(DocumentServiceError):
    """Raised when an upload is not PDF or DOCX."""


class EmptyUploadError(DocumentServiceError):
    """Raised when an uploaded file contains no bytes."""


class UploadTooLargeError(DocumentServiceError):
    """Raised when an upload exceeds the configured byte limit."""

    def __init__(self, max_size_mb: int) -> None:
        self.max_size_mb = max_size_mb
        super().__init__(f"File exceeds the {max_size_mb} MB upload limit.")


class DuplicateDocumentError(DocumentServiceError):
    """Raised when the collection already contains identical file bytes."""

    def __init__(self, existing_document_id: str) -> None:
        self.existing_document_id = existing_document_id
        super().__init__(
            f"An identical document already exists: {existing_document_id}"
        )


class DocumentNotFoundError(DocumentServiceError):
    """Raised when a requested document ID does not exist."""


class DocumentService:
    """Coordinate filesystem storage, parsing, chunking, and SQLite metadata."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        keyword_index: KeywordIndex,
        vector_store: ChromaVectorStore,
        embeddings: EmbeddingProvider,
    ) -> None:
        self.session = session
        self.settings = settings
        self.keyword_index = keyword_index
        self.vector_store = vector_store
        self.embeddings = embeddings

    def ingest_upload(self, upload: UploadFile) -> Document:
        """Validate, store, parse, chunk, and index one uploaded document."""

        original_filename = Path(upload.filename or "").name
        suffix = Path(original_filename).suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            upload.file.close()
            raise UnsupportedUploadError("Only PDF and DOCX uploads are supported.")

        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        document_id = str(uuid4())
        stored_filename = f"{document_id}{suffix}"
        final_path = self.settings.upload_dir / stored_filename
        temporary_path = self.settings.upload_dir / f".{stored_filename}.part"

        try:
            file_size, sha256 = self._write_temporary_upload(
                upload,
                temporary_path,
                self.settings.max_upload_size_mb * 1024 * 1024,
            )
            duplicate = self.session.scalar(
                select(Document).where(Document.sha256 == sha256)
            )
            if duplicate is not None:
                raise DuplicateDocumentError(duplicate.id)

            temporary_path.replace(final_path)
            extracted = extract_document(final_path)
            display_title = extracted.title
            if display_title == final_path.stem:
                display_title = Path(original_filename).stem
            return self._index_extracted(
                extracted,
                document_id=document_id,
                title=display_title,
                original_filename=original_filename,
                stored_filename=stored_filename,
                source_type=SourceType.UPLOAD,
                source_url=None,
                mime_type=upload.content_type,
                file_size_bytes=file_size,
                sha256=sha256,
            )
        except (DocumentServiceError, DocumentProcessingError):
            self.session.rollback()
            temporary_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise
        except (
            OSError,
            SQLAlchemyError,
            KeywordIndexError,
            VectorStoreError,
        ) as error:
            self.session.rollback()
            temporary_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise DocumentServiceError(
                "Could not store and index the uploaded document."
            ) from error

    def ingest_from_url(self, url: str) -> Document:
        """Scrape a URL, then index it through the same dual-index path as uploads."""

        extracted = extract_from_url(url)
        document_id = str(uuid4())
        sha256 = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
        download_bytes = extracted.metadata.get("download_bytes")
        file_size = download_bytes if isinstance(download_bytes, int) else None
        return self._index_extracted(
            extracted,
            document_id=document_id,
            title=extracted.title,
            original_filename=None,
            stored_filename=None,
            source_type=SourceType.WEB,
            source_url=url.strip(),
            mime_type="text/html",
            file_size_bytes=file_size,
            sha256=sha256,
        )

    def _index_extracted(
        self,
        extracted: ExtractedDocument,
        *,
        document_id: str,
        title: str,
        original_filename: str | None,
        stored_filename: str | None,
        source_type: SourceType,
        source_url: str | None,
        mime_type: str | None,
        file_size_bytes: int | None,
        sha256: str | None,
    ) -> Document:
        """Chunk, persist, and write both indexes for one extracted document.

        Upload and URL ingest share this path so adding or deleting a
        source always keeps the inverted index and vector store aligned.
        """

        keyword_index_written = False
        vector_index_attempted = False
        try:
            chunk_drafts = DocumentChunker(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            ).split(extracted)

            if sha256 is not None:
                duplicate = self.session.scalar(
                    select(Document).where(Document.sha256 == sha256)
                )
                if duplicate is not None:
                    raise DuplicateDocumentError(duplicate.id)

            document = Document(
                id=document_id,
                title=title,
                original_filename=original_filename,
                stored_filename=stored_filename,
                source_type=source_type,
                source_url=source_url,
                mime_type=mime_type,
                file_size_bytes=file_size_bytes,
                sha256=sha256,
                status=DocumentStatus.PENDING,
                chunk_count=len(chunk_drafts),
                extra_metadata={
                    **extracted.metadata,
                    "source_format": extracted.source_format,
                },
            )
            document.chunks = [
                Chunk(
                    position=chunk.position,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    extra_metadata=chunk.metadata,
                )
                for chunk in chunk_drafts
            ]
            self.session.add(document)
            self.session.flush()

            self.keyword_index.upsert_document(
                document.id,
                ((chunk.id, chunk.text) for chunk in document.chunks),
            )
            keyword_index_written = True
            document.keyword_indexed = True

            vector_index_attempted = True
            self.vector_store.upsert_document(
                document.id,
                (
                    VectorChunk(
                        chunk_id=chunk.id,
                        document_id=document.id,
                        text=chunk.text,
                        position=chunk.position,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section_title=chunk.section_title,
                    )
                    for chunk in document.chunks
                ),
                self.embeddings,
            )
            document.vector_indexed = True
            document.status = DocumentStatus.INDEXED
            document.indexed_at = datetime.now(UTC)
            self.session.commit()
            return document
        except (DocumentServiceError, DocumentProcessingError):
            self.session.rollback()
            if vector_index_attempted:
                self.vector_store.delete_document(document_id)
            if keyword_index_written:
                self.keyword_index.delete_document(document_id)
            raise
        except (
            OSError,
            SQLAlchemyError,
            KeywordIndexError,
            VectorStoreError,
        ) as error:
            self.session.rollback()
            if vector_index_attempted:
                self.vector_store.delete_document(document_id)
            if keyword_index_written:
                self.keyword_index.delete_document(document_id)
            raise DocumentServiceError(
                "Could not store and index the uploaded document."
            ) from error

    @staticmethod
    def _write_temporary_upload(
        upload: UploadFile,
        destination: Path,
        max_size_bytes: int,
    ) -> tuple[int, str]:
        digest = hashlib.sha256()
        total_size = 0

        try:
            with destination.open("xb") as output:
                while data := upload.file.read(_COPY_BUFFER_SIZE):
                    total_size += len(data)
                    if total_size > max_size_bytes:
                        raise UploadTooLargeError(max_size_bytes // (1024 * 1024))
                    digest.update(data)
                    output.write(data)
        finally:
            upload.file.close()

        if total_size == 0:
            raise EmptyUploadError("The uploaded file is empty.")
        return total_size, digest.hexdigest()

    def list_documents(
        self,
        *,
        offset: int = 0,
        limit: int = 20,
        source_type: SourceType | None = None,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        """Return a filtered, newest-first document page and total count."""

        filters = []
        if source_type is not None:
            filters.append(Document.source_type == source_type)
        if status is not None:
            filters.append(Document.status == status)

        total = self.session.scalar(
            select(func.count()).select_from(Document).where(*filters)
        )
        documents = list(
            self.session.scalars(
                select(Document)
                .where(*filters)
                .order_by(Document.created_at.desc(), Document.id.desc())
                .offset(offset)
                .limit(limit)
            )
        )
        return documents, int(total or 0)

    def get_document(self, document_id: str) -> Document:
        """Return one document or raise a domain-specific not-found error."""

        document = self.session.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        return document

    def delete_document(self, document_id: str) -> None:
        """Delete a document from both indexes, SQLite, and the filesystem."""

        document = self.get_document(document_id)
        source_path: Path | None = None
        tombstone_path: Path | None = None
        keyword_chunks = [(chunk.id, chunk.text) for chunk in document.chunks]
        vector_chunks = [
            VectorChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                text=chunk.text,
                position=chunk.position,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
            )
            for chunk in document.chunks
        ]
        keyword_index_removed = False
        vector_index_removed = False

        if document.stored_filename:
            upload_root = self.settings.upload_dir.resolve()
            source_path = (upload_root / document.stored_filename).resolve()
            if not source_path.is_relative_to(upload_root):
                raise DocumentServiceError("Refusing to delete an unsafe file path.")
            if source_path.exists():
                tombstone_path = source_path.with_name(
                    f".{source_path.name}.{uuid4().hex}.deleting"
                )
                source_path.replace(tombstone_path)

        try:
            if document.keyword_indexed:
                keyword_index_removed = self.keyword_index.delete_document(document.id)
            if document.vector_indexed:
                vector_index_removed = self.vector_store.delete_document(document.id)
            self.session.delete(document)
            self.session.commit()
        except (
            OSError,
            SQLAlchemyError,
            KeywordIndexError,
            VectorStoreError,
        ) as error:
            self.session.rollback()
            try:
                if keyword_index_removed:
                    self.keyword_index.upsert_document(
                        document.id,
                        keyword_chunks,
                    )
                if vector_index_removed:
                    self.vector_store.upsert_document(
                        document.id,
                        vector_chunks,
                        self.embeddings,
                    )
            finally:
                if (
                    source_path is not None
                    and tombstone_path is not None
                    and tombstone_path.exists()
                ):
                    tombstone_path.replace(source_path)
            raise DocumentServiceError("Could not delete the document.") from error

        if tombstone_path is not None:
            try:
                tombstone_path.unlink(missing_ok=True)
            except OSError as error:
                raise DocumentServiceError(
                    "Document metadata was deleted, but file cleanup failed."
                ) from error
