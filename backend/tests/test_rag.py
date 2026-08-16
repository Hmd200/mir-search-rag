"""Tests for grounded RAG generation with a mocked LLM client."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.rag import get_rag_service, router
from app.retrieval.llm import LLMError
from app.services.rag import RagService
from app.services.semantic_search import SemanticSearchRecord


def _record(index: int) -> SemanticSearchRecord:
    return SemanticSearchRecord(
        chunk_id=f"chunk-{index}",
        document_id=f"document-{index}",
        document_title=f"Document {index}",
        score=1.0 - (index * 0.05),
        distance=0.1 * index,
        text=f"Chunk {index} discusses retrieval ranking.",
        page_number=index,
        section_title=None,
    )


class FakeSearch:
    def __init__(self, records: list[SemanticSearchRecord]) -> None:
        self._records = records

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SemanticSearchRecord]:
        del query
        return self._records[:top_k]


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        return self._answer


class BoomLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        raise LLMError("The language model is unreachable.")


def _service(answer: str, chunk_count: int = 5) -> RagService:
    records = [_record(index) for index in range(1, chunk_count + 1)]
    return RagService(FakeSearch(records), FakeLLM(answer))


def test_well_cited_answer_keeps_all_citations() -> None:
    service = _service(
        "Inverted indexes support lexical search [1] and embeddings capture meaning [2].",
        chunk_count=5,
    )

    outcome = service.generate("How does retrieval work?", top_k=5)

    assert "[1]" in outcome.answer
    assert "[2]" in outcome.answer
    assert outcome.invalid_citations == ()
    assert [chunk.chunk_id for chunk in outcome.cited_chunks] == [
        "chunk-1",
        "chunk-2",
    ]
    assert outcome.abstained is False


def test_out_of_range_citation_is_reported_invalid() -> None:
    service = _service(
        "This claim is unsupported [9] while this one is grounded [1].",
        chunk_count=5,
    )

    outcome = service.generate("What is ranking?", top_k=5)

    assert outcome.invalid_citations == ("[9]",)
    assert "[9]" not in outcome.answer
    assert "[1]" in outcome.answer
    assert [chunk.chunk_id for chunk in outcome.cited_chunks] == ["chunk-1"]


def test_insufficient_evidence_sets_abstained_flag() -> None:
    service = _service("INSUFFICIENT_EVIDENCE", chunk_count=5)

    outcome = service.generate("What is the capital of Mars?", top_k=5)

    assert outcome.abstained is True
    assert outcome.answer == "INSUFFICIENT_EVIDENCE"
    assert outcome.cited_chunks == ()
    assert outcome.invalid_citations == ()


def test_think_blocks_are_stripped_from_the_answer() -> None:
    service = _service(
        "<think>internal chain</think>BM25 saturates term frequency [1].",
        chunk_count=5,
    )

    outcome = service.generate("What does BM25 do?", top_k=5)

    assert "<think>" not in outcome.answer
    assert "internal chain" not in outcome.answer
    assert "BM25 saturates term frequency [1]." == outcome.answer
    assert outcome.invalid_citations == ()


def test_llm_error_surfaces_as_503() -> None:
    application = FastAPI()
    application.include_router(router, prefix="/api/v1")

    def override_rag_service() -> RagService:
        records = [_record(index) for index in range(1, 4)]
        return RagService(FakeSearch(records), BoomLLM())

    application.dependency_overrides[get_rag_service] = override_rag_service
    client = TestClient(application)

    response = client.post(
        "/api/v1/search/rag",
        json={"query": "What is BM25?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "The language model is unreachable."
