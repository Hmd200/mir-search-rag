"""Tests for cross-encoder reranking with a mocked CrossEncoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.retrieval.reranker import CrossEncoderReranker
from app.services.rag import RagService
from app.services.semantic_search import SemanticSearchRecord


@dataclass
class _Chunk:
    text: str
    score: float = 0.0
    chunk_id: str = ""


class _PhraseMatchCrossEncoder:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def predict(self, pairs: list[list[str]]) -> list[float]:
        scores: list[float] = []
        for query, text in pairs:
            scores.append(1.0 if query in text else 0.0)
        return scores


class _RecordingCrossEncoder:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.pair_counts: list[int] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.pair_counts.append(len(pairs))
        return [0.0] * len(pairs)


def _record(
    chunk_id: str,
    text: str,
    *,
    score: float,
) -> SemanticSearchRecord:
    return SemanticSearchRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        document_title=f"Title {chunk_id}",
        score=score,
        distance=1.0 - score,
        text=text,
        page_start=1,
        page_end=1,
        section_title=None,
    )


class _DummySearch:
    def search(self, query: str, *, top_k: int) -> list[SemanticSearchRecord]:
        del query, top_k
        return []


class _DummyLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return "INSUFFICIENT_EVIDENCE"


def test_exact_phrase_match_ranks_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        _PhraseMatchCrossEncoder,
    )
    reranker = CrossEncoderReranker(
        "mock-model",
        cache_dir=tmp_path,
        device="cpu",
    )
    query = "quantum superposition"
    mismatch = _Chunk(text="unrelated boolean algebra", score=0.99, chunk_id="b")
    exact = _Chunk(
        text="notes on quantum superposition in circuits",
        score=0.01,
        chunk_id="a",
    )

    ranked = reranker.rerank(query, [mismatch, exact], top_n=10)

    assert [item.chunk.chunk_id for item in ranked] == ["a", "b"]
    assert ranked[0].chunk is exact


def test_rerank_respects_top_n(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        _PhraseMatchCrossEncoder,
    )
    reranker = CrossEncoderReranker(
        "mock-model",
        cache_dir=tmp_path,
        device="cpu",
    )
    chunks = [
        _Chunk(text=f"chunk {index}", score=float(index), chunk_id=str(index))
        for index in range(5)
    ]

    ranked = reranker.rerank("chunk 4", chunks, top_n=2)

    assert len(ranked) == 2


def test_more_than_25_chunks_are_capped_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = _RecordingCrossEncoder()

    def _factory(*args: object, **kwargs: object) -> _RecordingCrossEncoder:
        del args, kwargs
        return recorder

    monkeypatch.setattr("app.retrieval.reranker.CrossEncoder", _factory)
    reranker = CrossEncoderReranker(
        "mock-model",
        cache_dir=tmp_path,
        device="cpu",
    )
    chunks = [
        _Chunk(text=f"text {index}", score=1.0, chunk_id=str(index))
        for index in range(30)
    ]

    ranked = reranker.rerank("query", chunks, top_n=10)

    assert recorder.pair_counts == [25]
    assert len(ranked) == 10


def test_load_failure_keeps_original_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("model missing")

    monkeypatch.setattr("app.retrieval.reranker.CrossEncoder", _boom)
    reranker = CrossEncoderReranker(
        "missing-model",
        cache_dir=tmp_path,
        device="cpu",
    )
    first = _Chunk(text="alpha", score=0.8, chunk_id="first")
    second = _Chunk(text="beta", score=0.2, chunk_id="second")

    ranked = reranker.rerank("query", [first, second], top_n=10)

    assert [item.chunk.chunk_id for item in ranked] == ["first", "second"]
    assert ranked[0].chunk is first


def test_rag_context_selection_exposes_both_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        _PhraseMatchCrossEncoder,
    )
    reranker = CrossEncoderReranker(
        "mock-model",
        cache_dir=tmp_path,
        device="cpu",
    )
    service = RagService(_DummySearch(), _DummyLLM(), reranker=reranker)
    query = "quantum superposition"
    retrieved = [
        _record("mismatch", "unrelated boolean algebra", score=0.91),
        _record(
            "exact",
            "lecture notes on quantum superposition",
            score=0.12,
        ),
    ]

    selected = service.select_context(
        query,
        retrieved,
        top_k=2,
        use_reranker=True,
    )

    assert selected[0].chunk_id == "exact"
    assert selected[0].retrieval_score == pytest.approx(0.12)
    assert selected[0].rerank_score is not None
    assert selected[1].retrieval_score == pytest.approx(0.91)
    assert selected[1].rerank_score is not None
