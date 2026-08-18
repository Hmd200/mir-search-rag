"""RAG API request and response schemas."""

from pydantic import BaseModel, Field


class RagRequest(BaseModel):
    """JSON body for grounded answer generation."""

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=50)
    use_reranker: bool = False


class RagCitedChunk(BaseModel):
    """A retrieved chunk cited by the generated answer."""

    chunk_id: str
    document_id: str
    document_title: str
    page_number: int | None
    text: str
    score: float
    retrieval_score: float
    rerank_score: float | None = None


class RagResponse(BaseModel):
    """Grounded answer with citation diagnostics."""

    query: str
    answer: str
    cited_chunks: list[RagCitedChunk]
    invalid_citations: list[str]
    abstained: bool
    elapsed_ms: float
