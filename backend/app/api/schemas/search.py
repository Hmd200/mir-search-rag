"""Search API response schemas."""

from typing import Literal

from pydantic import BaseModel


class KeywordSearchResult(BaseModel):
    """One ranked TF-IDF chunk with citation and scoring evidence."""

    chunk_id: str
    document_id: str
    document_title: str
    score: float
    text: str
    page_number: int | None
    section_title: str | None
    matched_terms: tuple[str, ...]
    term_contributions: dict[str, float]


class KeywordSearchResponse(BaseModel):
    """Ranked response from the custom vector-space retrieval engine."""

    query: str
    mode: Literal["tfidf"] = "tfidf"
    result_count: int
    elapsed_ms: float
    results: list[KeywordSearchResult]


class BM25SearchResponse(BaseModel):
    """Ranked response from the custom Okapi BM25 retrieval engine."""

    query: str
    mode: Literal["bm25"] = "bm25"
    k1: float
    b: float
    result_count: int
    elapsed_ms: float
    results: list[KeywordSearchResult]


class KeywordIndexStatsResponse(BaseModel):
    """Summary statistics for the custom inverted index."""

    document_count: int
    chunk_count: int
    vocabulary_size: int
    posting_count: int
