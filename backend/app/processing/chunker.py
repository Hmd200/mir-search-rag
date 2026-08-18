"""Deterministic page-aware text chunking."""

import re
from dataclasses import dataclass

from app.processing.schemas import (
    ChunkDraft,
    EmptyDocumentError,
    ExtractedDocument,
    ExtractedSegment,
)

_WORD_PATTERN = re.compile(r"\S+")


def _covering_segments(
    segments: tuple[ExtractedSegment, ...],
    char_start: int,
    char_end: int,
) -> tuple[ExtractedSegment, ...]:
    """Return segments whose text overlaps the chunk character range."""

    return tuple(
        segment
        for segment in segments
        if segment.char_start < char_end and segment.char_end > char_start
    )


@dataclass(frozen=True, slots=True)
class DocumentChunker:
    """Split extracted text into overlapping word windows."""

    chunk_size: int = 500
    chunk_overlap: int = 75

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

    def split(self, document: ExtractedDocument) -> list[ChunkDraft]:
        """Create overlapping word windows that may span page boundaries.

        Word windows still snap to existing token boundaries; they no longer
        reset at each page edge. Cross-page windows let a phrase split by a
        page break co-occur in one chunk, which improves recall. The tradeoff
        is coarser citations: a hit may be labeled "Pages 5-6" instead of a
        single exact page.
        """

        chunks: list[ChunkDraft] = []
        next_position = 0
        word_matches = list(_WORD_PATTERN.finditer(document.text))
        start_word = 0

        while start_word < len(word_matches):
            end_word = min(start_word + self.chunk_size, len(word_matches))
            first_match = word_matches[start_word]
            last_match = word_matches[end_word - 1]
            char_start = first_match.start()
            char_end = last_match.end()
            text = document.text[char_start:char_end]
            covering = _covering_segments(document.segments, char_start, char_end)
            first_segment = covering[0] if covering else None
            last_segment = covering[-1] if covering else None
            metadata = dict(first_segment.metadata) if first_segment else {}
            metadata["source_format"] = document.source_format
            chunks.append(
                ChunkDraft(
                    position=next_position,
                    text=text,
                    token_count=end_word - start_word,
                    char_start=char_start,
                    char_end=char_end,
                    page_start=(first_segment.page_number if first_segment else None),
                    page_end=(last_segment.page_number if last_segment else None),
                    section_title=(
                        first_segment.section_title if first_segment else None
                    ),
                    metadata=metadata,
                )
            )
            next_position += 1

            if end_word == len(word_matches):
                break
            start_word = end_word - self.chunk_overlap

        if not chunks:
            raise EmptyDocumentError("The document contains no words to chunk.")
        return chunks
