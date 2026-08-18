"""Retrieval-augmented generation over semantic search hits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

from app.retrieval.llm import LLMClient, strip_think_blocks
from app.retrieval.reranker import CrossEncoderReranker, reranker_from_settings
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
    page_start: int | None
    page_end: int | None
    text: str
    score: float
    retrieval_score: float
    rerank_score: float | None


@dataclass(frozen=True, slots=True)
class RagContextChunk:
    """A retrieved chunk after the top-20 to top-N context cut."""

    chunk_id: str
    document_id: str
    document_title: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    text: str
    retrieval_score: float
    rerank_score: float | None

    @property
    def score(self) -> float:
        return self.retrieval_score


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
    record: RagContextChunk,
) -> str:
    page = ""
    if record.page_start is not None and record.page_end is not None:
        if record.page_start == record.page_end:
            page = f", page {record.page_start}"
        else:
            page = f", pages {record.page_start}-{record.page_end}"
    return f"[{index}] {record.document_title}{page}\n{record.text}"


def _build_context(records: list[RagContextChunk]) -> str:
    return "\n\n".join(
        _format_context_chunk(index, record)
        for index, record in enumerate(records, start=1)
    )


def _build_user_prompt(query: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {query}"


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
    records: list[RagContextChunk],
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
                page_start=record.page_start,
                page_end=record.page_end,
                text=record.text,
                score=record.retrieval_score,
                retrieval_score=record.retrieval_score,
                rerank_score=record.rerank_score,
            )
        )
    return tuple(cited)


def _to_context_chunk(
    record: SemanticSearchRecord,
    *,
    retrieval_score: float,
    rerank_score: float | None,
) -> RagContextChunk:
    return RagContextChunk(
        chunk_id=record.chunk_id,
        document_id=record.document_id,
        document_title=record.document_title,
        page_start=record.page_start,
        page_end=record.page_end,
        section_title=record.section_title,
        text=record.text,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
    )


class RagService:
    """Ground an LLM answer in semantically retrieved chunks."""

    def __init__(
        self,
        search: SemanticSearchService,
        llm: LLMClient,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._search = search
        self._llm = llm
        self._reranker = reranker

    def _ensure_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = reranker_from_settings()
        return self._reranker

    def select_context(
        self,
        query: str,
        retrieved: list[SemanticSearchRecord],
        *,
        top_k: int,
        use_reranker: bool = False,
    ) -> list[RagContextChunk]:
        """Narrow the top-20 retrieval list to the prompt window.

        Without reranking this is a prefix slice. When enabled, the same
        cut is a cross-encoder rerank then take top-N — not a second
        retrieval path.
        """

        if not retrieved or top_k <= 0:
            return []

        if not use_reranker:
            return [
                _to_context_chunk(
                    record,
                    retrieval_score=record.score,
                    rerank_score=None,
                )
                for record in retrieved[:top_k]
            ]

        ranked = self._ensure_reranker().rerank(
            query,
            retrieved,
            top_n=top_k,
        )
        selected: list[RagContextChunk] = []
        for item in ranked:
            record = item.chunk
            selected.append(
                _to_context_chunk(
                    record,
                    retrieval_score=item.retrieval_score,
                    rerank_score=item.rerank_score,
                )
            )
        return selected

    def generate(
        self,
        query: str,
        *,
        top_k: int = 5,
        use_reranker: bool = False,
    ) -> RagOutcome:
        started = perf_counter()
        retrieved = self._search.search(
            query,
            top_k=_RETRIEVAL_K,
        )
        # Existing top-20 -> top-N narrowing; rerank replaces the slice.
        context_records = self.select_context(
            query,
            retrieved,
            top_k=top_k,
            use_reranker=use_reranker,
        )

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
