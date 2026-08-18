"""Document API response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models import DocumentStatus, SourceType


class DocumentResponse(BaseModel):
    """Admin-facing metadata for one collection document."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    original_filename: str | None
    source_type: SourceType
    source_url: str | None
    mime_type: str | None
    file_size_bytes: int | None
    status: DocumentStatus
    chunk_count: int
    keyword_indexed: bool
    vector_indexed: bool
    extra_metadata: dict[str, Any]
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None


class DocumentListResponse(BaseModel):
    """Paginated document collection response."""

    items: list[DocumentResponse]
    total: int
    offset: int
    limit: int


class DocumentFromUrlRequest(BaseModel):
    """Admin request to scrape a web page into the corpus."""

    url: str = Field(min_length=1, max_length=2000)
