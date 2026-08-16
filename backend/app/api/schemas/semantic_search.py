"""Semantic-search API response schemas."""

from typing import Literal

from pydantic import BaseModel


class SemanticSearchResult(BaseModel):
    """One vector-ranked chunk with source and citation metadata."""

    chunk_id: str
    document_id: str
    document_title: str
    score: float
    distance: float
    text: str
    page_number: int | None
    section_title: str | None


class SemanticSearchResponse(BaseModel):
    """Ranked response from the Chroma semantic-search engine."""

    query: str
    mode: Literal["semantic"] = "semantic"
    result_count: int
    elapsed_ms: float
    results: list[SemanticSearchResult]


class VectorStoreStatsResponse(BaseModel):
    """Summary statistics for the Chroma collection."""

    chunk_count: int
