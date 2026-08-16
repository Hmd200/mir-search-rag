"""Retrieval-augmented generation over semantic search hits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

from app.retrieval.llm import LLMClient, strip_think_blocks
from app.services.semantic_search import (
    SemanticSearchRecord,
    SemanticSearchService,
)

_RETRIEVAL_K = 20
_CITATION = re.compile(r"\[(\d+)\]")
_ABSTAIN_TEXT = "INSUFFICIENT_EVIDENCE"

_SYSTEM_PROMPT = """\
You are a citation-grounded question answering assistant.
Answer ONLY from the provided context chunks.
Cite every claim with bracket markers such as [1] that match the chunk labels.
If the context does not contain enough information to answer the question, \
reply exactly:
INSUFFICIENT_EVIDENCE
Do not use outside knowledge or invent citations.\
"""


@dataclass(frozen=True, slots=True)
class RagCitedChunk:
    """A context chunk referenced by a generated answer."""

    chunk_id: str
    document_id: str
    document_title: str
    page_number: int | None
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class RagOutcome:
    """Grounded generation result with citation diagnostics."""

    query: str
    answer: str
    cited_chunks: tuple[RagCitedChunk, ...]
    invalid_citations: tuple[str, ...]
    abstained: bool
    elapsed_ms: float


def _format_context_chunk(
    index: int,
    record: SemanticSearchRecord,
) -> str:
    page = (
        f", page {record.page_number}"
        if record.page_number is not None
        else ""
    )
    return f"[{index}] {record.document_title}{page}\n{record.text}"


def _build_context(records: list[SemanticSearchRecord]) -> str:
    return "\n\n".join(
        _format_context_chunk(index, record)
        for index, record in enumerate(records, start=1)
    )


def _build_user_prompt(query: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )


def validate_citations(
    answer: str,
    chunk_count: int,
) -> tuple[str, tuple[str, ...]]:
    """Drop citation markers that do not refer to a context chunk."""

    invalid: list[str] = []
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        marker = match.group(0)
        if 1 <= number <= chunk_count:
            return marker
        if marker not in seen:
            seen.add(marker)
            invalid.append(marker)
        return ""

    cleaned = _CITATION.sub(replace, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), tuple(invalid)


def _cited_chunks(
    answer: str,
    records: list[SemanticSearchRecord],
) -> tuple[RagCitedChunk, ...]:
    seen: set[int] = set()
    cited: list[RagCitedChunk] = []
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if number in seen or not 1 <= number <= len(records):
            continue
        seen.add(number)
        record = records[number - 1]
        cited.append(
            RagCitedChunk(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                document_title=record.document_title,
                page_number=record.page_number,
                text=record.text,
                score=record.score,
            )
        )
    return tuple(cited)


class RagService:
    """Ground an LLM answer in semantically retrieved chunks."""

    def __init__(
        self,
        search: SemanticSearchService,
        llm: LLMClient,
    ) -> None:
        self._search = search
        self._llm = llm

    def generate(self, query: str, *, top_k: int = 5) -> RagOutcome:
        started = perf_counter()
        retrieved = self._search.search(
            query,
            top_k=_RETRIEVAL_K,
        )
        context_records = retrieved[:top_k]

        if not context_records:
            return RagOutcome(
                query=query,
                answer=_ABSTAIN_TEXT,
                cited_chunks=(),
                invalid_citations=(),
                abstained=True,
                elapsed_ms=(perf_counter() - started) * 1000,
            )

        raw_answer = self._llm.generate(
            _SYSTEM_PROMPT,
            _build_user_prompt(
                query,
                _build_context(context_records),
            ),
        )
        answer = strip_think_blocks(raw_answer)
        abstained = answer == _ABSTAIN_TEXT

        if abstained:
            return RagOutcome(
                query=query,
                answer=answer,
                cited_chunks=(),
                invalid_citations=(),
                abstained=True,
                elapsed_ms=(perf_counter() - started) * 1000,
            )

        cleaned, invalid = validate_citations(
            answer,
            len(context_records),
        )
        return RagOutcome(
            query=query,
            answer=cleaned,
            cited_chunks=_cited_chunks(
                cleaned,
                context_records,
            ),
            invalid_citations=invalid,
            abstained=False,
            elapsed_ms=(perf_counter() - started) * 1000,
        )
