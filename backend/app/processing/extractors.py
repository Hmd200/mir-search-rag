"""Local PDF and DOCX text extraction with citation metadata."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from docx import Document as open_docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.processing.schemas import (
    DocumentProcessingError,
    EmptyDocumentError,
    ExtractedDocument,
    ExtractedSegment,
    UnsupportedDocumentError,
)


@dataclass(slots=True)
class _RawSegment:
    text: str
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    """Remove extraction noise while retaining paragraph boundaries."""

    cleaned_lines: list[str] = []
    previous_blank = False
    for raw_line in text.replace("\x00", "").splitlines():
        line = " ".join(raw_line.split())
        if line:
            cleaned_lines.append(line)
            previous_blank = False
        elif cleaned_lines and not previous_blank:
            cleaned_lines.append("")
            previous_blank = True
    return "\n".join(cleaned_lines).strip()


def _assemble_document(
    *,
    title: str,
    source_format: str,
    raw_segments: list[_RawSegment],
    metadata: dict[str, Any],
) -> ExtractedDocument:
    text_parts: list[str] = []
    segments: list[ExtractedSegment] = []
    cursor = 0

    for raw_segment in raw_segments:
        normalized = _normalize_text(raw_segment.text)
        if not normalized:
            continue
        if text_parts:
            cursor += 2
        segments.append(
            ExtractedSegment(
                text=normalized,
                char_start=cursor,
                page_number=raw_segment.page_number,
                section_title=raw_segment.section_title,
                metadata=raw_segment.metadata,
            )
        )
        text_parts.append(normalized)
        cursor += len(normalized)

    if not segments:
        raise EmptyDocumentError("The document contains no extractable text.")

    normalized_format = source_format.lower()
    if normalized_format not in {"pdf", "docx"}:
        raise UnsupportedDocumentError(f"Unsupported document format: {source_format}")

    return ExtractedDocument(
        title=title.strip() or "Untitled document",
        source_format=normalized_format,  # type: ignore[arg-type]
        text="\n\n".join(text_parts),
        segments=tuple(segments),
        metadata=metadata,
    )


def _extract_pdf(path: Path) -> ExtractedDocument:
    try:
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise DocumentProcessingError(
                    "Password-protected PDF files are not supported."
                )

            pdf_metadata = pdf.metadata or {}
            raw_segments = [
                _RawSegment(
                    text=page.get_text("text", sort=True),
                    page_number=page_index + 1,
                    metadata={"page_index": page_index},
                )
                for page_index, page in enumerate(pdf)
            ]
            metadata = {
                "page_count": len(pdf),
                "author": pdf_metadata.get("author") or None,
                "subject": pdf_metadata.get("subject") or None,
                "keywords": pdf_metadata.get("keywords") or None,
            }
            title = pdf_metadata.get("title") or path.stem
    except DocumentProcessingError:
        raise
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise DocumentProcessingError(f"Could not read PDF: {path.name}") from error

    return _assemble_document(
        title=title,
        source_format="pdf",
        raw_segments=raw_segments,
        metadata=metadata,
    )


def _table_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [_normalize_text(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_docx(path: Path) -> ExtractedDocument:
    try:
        document = open_docx(str(path))
    except Exception as error:
        raise DocumentProcessingError(f"Could not read DOCX: {path.name}") from error

    raw_segments: list[_RawSegment] = []
    section_lines: list[str] = []
    section_title: str | None = None
    section_start = 0
    block_index = 0
    table_count = 0

    def flush_section(end_index: int) -> None:
        nonlocal section_lines
        if not section_lines:
            return
        raw_segments.append(
            _RawSegment(
                text="\n\n".join(section_lines),
                section_title=section_title,
                metadata={
                    "block_start": section_start,
                    "block_end": end_index,
                },
            )
        )
        section_lines = []

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text = _normalize_text(block.text)
            style_name = block.style.name if block.style is not None else ""
            if text and style_name.lower().startswith("heading"):
                flush_section(block_index - 1)
                section_title = text
                section_start = block_index
                section_lines = [text]
            elif text:
                if not section_lines:
                    section_start = block_index
                section_lines.append(text)
        elif isinstance(block, Table):
            table_text = _table_text(block)
            if table_text:
                if not section_lines:
                    section_start = block_index
                section_lines.append(table_text)
                table_count += 1
        block_index += 1

    flush_section(block_index - 1)
    properties = document.core_properties
    metadata = {
        "author": properties.author or None,
        "subject": properties.subject or None,
        "keywords": properties.keywords or None,
        "table_count": table_count,
        "section_count": len(raw_segments),
    }

    return _assemble_document(
        title=properties.title or path.stem,
        source_format="docx",
        raw_segments=raw_segments,
        metadata=metadata,
    )


def extract_document(path: str | Path) -> ExtractedDocument:
    """Extract a supported local file into normalized, citable segments."""

    source_path = Path(path)
    if not source_path.is_file():
        raise DocumentProcessingError(f"Document does not exist: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(source_path)
    if suffix == ".docx":
        return _extract_docx(source_path)
    raise UnsupportedDocumentError(
        f"Unsupported file type '{suffix or 'unknown'}'. Only PDF and DOCX are allowed."
    )
