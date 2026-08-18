"""Typed values shared by document extractors and chunkers."""

from dataclasses import dataclass, field
from typing import Any, Literal


class DocumentProcessingError(ValueError):
    """Base error for files that cannot be converted into searchable text."""


class UnsupportedDocumentError(DocumentProcessingError):
    """Raised when a file format is outside the supported collection types."""


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a valid file contains no searchable text."""


@dataclass(frozen=True, slots=True)
class ExtractedSegment:
    """A page or logical section with an offset in the normalized document."""

    text: str
    char_start: int
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_end(self) -> int:
        """Return the exclusive end offset in the normalized document."""

        return self.char_start + len(self.text)


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Normalized text and citation structure extracted from a source file."""

    title: str
    source_format: Literal["pdf", "docx", "html"]
    text: str
    segments: tuple[ExtractedSegment, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk ready to receive a persistent ID and enter both indexes."""

    position: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
