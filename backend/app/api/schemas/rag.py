"""RAG API request and response schemas."""

from typing import Literal

from pydantic import BaseModel, Field

RetrievalSource = Literal["dense", "bm25"]


class RagRequest(BaseModel):
    """JSON body for grounded answer generation."""

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)
    use_reranker: bool = False
    use_query_rewrite: bool = False
    llm_provider: Literal["ollama", "gemini"] | None = None


class RagCitedChunk(BaseModel):
    """A retrieved chunk cited by the generated answer."""

    chunk_id: str
    document_id: str
    document_title: str
    page_start: int | None
    page_end: int | None
    text: str
    score: float
    retrieval_score: float
    rerank_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    retrieval_sources: tuple[RetrievalSource, ...] = ()
    prompt_index: int | None = None


AbstentionReason = Literal[
    "no_context",
    "low_relevance",
    "model_abstained",
    "citation_failure",
    "grounding_failure",
]


class RagResponse(BaseModel):
    """Grounded answer with citation diagnostics."""

    query: str
    answer: str
    cited_chunks: list[RagCitedChunk]
    context_chunks: list[RagCitedChunk]
    invalid_citations: list[str]
    abstained: bool
    elapsed_ms: float
    rewritten_query: str | None = None
    llm_provider: Literal["ollama", "gemini"]
    citation_enforced: bool = False
    abstention_reason: AbstentionReason | None = None
