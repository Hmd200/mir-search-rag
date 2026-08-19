"""Classical retrieval service joining index hits with stored metadata."""

from dataclasses import dataclass, replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.retrieval.reranker import CrossEncoderReranker, reranker_from_settings
from app.storage.keyword_index import (
    KeywordIndex,
    KeywordSearchHit,
    PrfExpansion,
)

_RERANK_RETRIEVE_K = 25


@dataclass(frozen=True, slots=True)
class KeywordSearchRecord:
    """A TF-IDF hit enriched with its source and citation fields."""

    chunk_id: str
    document_id: str
    document_title: str
    score: float
    text: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    matched_terms: tuple[str, ...]
    term_contributions: dict[str, float]
    retrieval_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class KeywordSearchServiceOutcome:
    """Hydrated keyword hits with optional PRF expansion metadata."""

    records: list[KeywordSearchRecord]
    expansion: PrfExpansion | None = None
    reranked: bool = False


class KeywordSearchService:
    """Run the custom TF-IDF engine and hydrate ranked chunk results."""

    def __init__(
        self,
        session: Session,
        keyword_index: KeywordIndex,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.session = session
        self.keyword_index = keyword_index
        self._reranker = reranker

    def _ensure_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = reranker_from_settings()
        return self._reranker

    def _apply_rerank(
        self,
        query: str,
        records: list[KeywordSearchRecord],
        *,
        top_n: int,
    ) -> list[KeywordSearchRecord]:
        ranked = self._ensure_reranker().rerank(
            query,
            records,
            top_n=top_n,
        )
        reranked_records: list[KeywordSearchRecord] = []
        for item in ranked:
            record = item.chunk
            reranked_records.append(
                replace(
                    record,
                    retrieval_score=item.retrieval_score,
                    rerank_score=item.rerank_score,
                )
            )
        return reranked_records

    def search(
        self,
        query: str,
        *,
        top_k: int,
        candidate_limit: int,
        use_prf: bool = False,
        feedback_docs: int = 5,
        max_expansion_terms: int = 10,
        expansion_terms: int | None = None,
        alpha: float = 1.0,
        beta: float = 0.75,
        use_reranker: bool = False,
    ) -> KeywordSearchServiceOutcome:
        retrieve_k = _RERANK_RETRIEVE_K if use_reranker else top_k
        if use_prf:
            prf = self.keyword_index.search_with_prf(
                query,
                top_k=retrieve_k,
                feedback_docs=feedback_docs,
                max_expansion_terms=(
                    expansion_terms
                    if expansion_terms is not None
                    else max_expansion_terms
                ),
                alpha=alpha,
                beta=beta,
                scoring_mode="tfidf",
            )
            records = self._hydrate(list(prf.hits))
            expansion = prf.expansion
        else:
            hits = self.keyword_index.search(
                query,
                top_k=retrieve_k,
            )
            records = self._hydrate(hits)
            expansion = None

        reranked = False
        if use_reranker and records:
            records = self._apply_rerank(
                query,
                records,
                top_n=top_k,
            )
            reranked = True

        return KeywordSearchServiceOutcome(
            records=records,
            expansion=expansion,
            reranked=reranked,
        )

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int,
        candidate_limit: int,
        k1: float,
        b: float,
        use_prf: bool = False,
        feedback_docs: int = 5,
        max_expansion_terms: int = 10,
        expansion_terms: int | None = None,
        alpha: float = 1.0,
        beta: float = 0.75,
        use_reranker: bool = False,
    ) -> KeywordSearchServiceOutcome:
        """Run BM25 and hydrate its ranked chunks with citation metadata."""

        retrieve_k = _RERANK_RETRIEVE_K if use_reranker else top_k
        if use_prf:
            prf = self.keyword_index.search_with_prf(
                query,
                top_k=retrieve_k,
                feedback_docs=feedback_docs,
                max_expansion_terms=(
                    expansion_terms
                    if expansion_terms is not None
                    else max_expansion_terms
                ),
                alpha=alpha,
                beta=beta,
                scoring_mode="bm25",
                k1=k1,
                b=b,
            )
            records = self._hydrate(list(prf.hits))
            expansion = prf.expansion
        else:
            hits = self.keyword_index.search_bm25(
                query,
                top_k=retrieve_k,
                k1=k1,
                b=b,
            )
            records = self._hydrate(hits)
            expansion = None

        reranked = False
        if use_reranker and records:
            records = self._apply_rerank(
                query,
                records,
                top_n=top_k,
            )
            reranked = True

        return KeywordSearchServiceOutcome(
            records=records,
            expansion=expansion,
            reranked=reranked,
        )

    def _hydrate(
        self,
        hits: list[KeywordSearchHit],
    ) -> list[KeywordSearchRecord]:
        if not hits:
            return []

        chunks = {
            chunk.id: chunk
            for chunk in self.session.scalars(
                select(Chunk).where(Chunk.id.in_([hit.chunk_id for hit in hits]))
            )
        }

        results: list[KeywordSearchRecord] = []
        for hit in hits:
            chunk = chunks.get(hit.chunk_id)
            if chunk is None:
                continue
            results.append(
                KeywordSearchRecord(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=chunk.document.title,
                    score=hit.score,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    matched_terms=hit.matched_terms,
                    term_contributions=hit.term_contributions,
                    retrieval_score=hit.score,
                    rerank_score=None,
                )
            )
        return results
