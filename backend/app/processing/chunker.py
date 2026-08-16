"""Deterministic page-aware text chunking."""

import re
from dataclasses import dataclass

from app.processing.schemas import ChunkDraft, EmptyDocumentError, ExtractedDocument

_WORD_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class DocumentChunker:
    """Split each citation segment into overlapping word windows."""

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
        """Create stable chunks that never cross a page or section boundary."""

        chunks: list[ChunkDraft] = []
        next_position = 0

        for segment in document.segments:
            word_matches = list(_WORD_PATTERN.finditer(segment.text))
            start_word = 0

            while start_word < len(word_matches):
                end_word = min(start_word + self.chunk_size, len(word_matches))
                first_match = word_matches[start_word]
                last_match = word_matches[end_word - 1]
                local_start = first_match.start()
                local_end = last_match.end()
                text = segment.text[local_start:local_end]

                metadata = dict(segment.metadata)
                metadata["source_format"] = document.source_format
                chunks.append(
                    ChunkDraft(
                        position=next_position,
                        text=text,
                        token_count=end_word - start_word,
                        char_start=segment.char_start + local_start,
                        char_end=segment.char_start + local_end,
                        page_number=segment.page_number,
                        section_title=segment.section_title,
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
