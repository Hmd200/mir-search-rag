"""Classical retrieval service joining index hits with stored metadata."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.storage.keyword_index import (
    KeywordIndex,
    KeywordSearchHit,
    PrfExpansion,
)


@dataclass(frozen=True, slots=True)
class KeywordSearchRecord:
    """A TF-IDF hit enriched with its source and citation fields."""

    chunk_id: str
    document_id: str
    document_title: str
    score: float
    text: str
    page_number: int | None
    section_title: str | None
    matched_terms: tuple[str, ...]
    term_contributions: dict[str, float]


@dataclass(frozen=True, slots=True)
class KeywordSearchServiceOutcome:
    """Hydrated keyword hits with optional PRF expansion metadata."""

    records: list[KeywordSearchRecord]
    expansion: PrfExpansion | None = None


class KeywordSearchService:
    """Run the custom TF-IDF engine and hydrate ranked chunk results."""

    def __init__(self, session: Session, keyword_index: KeywordIndex) -> None:
        self.session = session
        self.keyword_index = keyword_index

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
    ) -> KeywordSearchServiceOutcome:
        if use_prf:
            prf = self.keyword_index.search_with_prf(
                query,
                top_k=top_k,
                feedback_docs=feedback_docs,
                max_expansion_terms=(
                    expansion_terms
                    if expansion_terms is not None
                    else max_expansion_terms
                ),
                alpha=alpha,
                beta=beta,
                scoring_mode="tfidf",
                candidate_limit=max(top_k, candidate_limit),
            )
            return KeywordSearchServiceOutcome(
                records=self._hydrate(list(prf.hits)),
                expansion=prf.expansion,
            )

        hits = self.keyword_index.search(
            query,
            top_k=top_k,
            candidate_limit=max(top_k, candidate_limit),
        )
        return KeywordSearchServiceOutcome(
            records=self._hydrate(hits),
            expansion=None,
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
    ) -> KeywordSearchServiceOutcome:
        """Run BM25 and hydrate its ranked chunks with citation metadata."""

        if use_prf:
            prf = self.keyword_index.search_with_prf(
                query,
                top_k=top_k,
                feedback_docs=feedback_docs,
                max_expansion_terms=(
                    expansion_terms
                    if expansion_terms is not None
                    else max_expansion_terms
                ),
                alpha=alpha,
                beta=beta,
                scoring_mode="bm25",
                candidate_limit=max(top_k, candidate_limit),
                k1=k1,
                b=b,
            )
            return KeywordSearchServiceOutcome(
                records=self._hydrate(list(prf.hits)),
                expansion=prf.expansion,
            )

        hits = self.keyword_index.search_bm25(
            query,
            top_k=top_k,
            candidate_limit=max(top_k, candidate_limit),
            k1=k1,
            b=b,
        )
        return KeywordSearchServiceOutcome(
            records=self._hydrate(hits),
            expansion=None,
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
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    matched_terms=hit.matched_terms,
                    term_contributions=hit.term_contributions,
                )
            )
        return results
