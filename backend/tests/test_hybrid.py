"""Hybrid RAG fusion, lexical gate, pinning, and provenance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_rag import FakeLLM, FakeSearch, _low_score_record

from app.retrieval import TextAnalyzer
from app.retrieval.hybrid import (
    fused_chunk_order,
    is_lexically_strong,
    lexical_coverages,
    pinning_relative_bm25,
    reciprocal_rank_fusion,
    retrieval_sources,
)
from app.retrieval.reranker import CrossEncoderReranker, RerankResult
from app.services.rag import RagContextChunk, RagService
from app.storage.keyword_index import KeywordIndex, KeywordSearchHit


def _context(
    chunk_id: str,
    *,
    text: str,
    dense_score: float | None,
    bm25_score: float | None,
    fusion_score: float | None,
    title: str = "Doc",
) -> RagContextChunk:
    sources = retrieval_sources(
        has_dense=dense_score is not None,
        has_bm25=bm25_score is not None,
    )
    ordering = fusion_score if fusion_score is not None else (dense_score or 0.0)
    return RagContextChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        document_title=title,
        page_start=1,
        page_end=1,
        section_title=None,
        text=text,
        retrieval_score=ordering,
        rerank_score=None,
        dense_score=dense_score,
        bm25_score=bm25_score,
        fusion_score=fusion_score,
        retrieval_sources=sources,
    )


@dataclass
class FakeKeywordIndex:
    hits: list[KeywordSearchHit]
    idf_value: float = 1.0
    queries: list[str] = field(default_factory=list)
    k1_values: list[float] = field(default_factory=list)
    b_values: list[float] = field(default_factory=list)

    def search_bm25(self, query: str, *, top_k: int, k1: float, b: float, **_: object):
        self.queries.append(query)
        self.k1_values.append(k1)
        self.b_values.append(b)
        return self.hits[:top_k]

    def bm25_idf(self, term: str) -> float:
        del term
        return self.idf_value


class ReverseReranker:
    def rerank(self, query: str, chunks: list[RagContextChunk], top_n: int = 10):
        del query
        ordered = list(reversed(list(chunks)))
        return [
            RerankResult(
                chunk=chunk,
                retrieval_score=chunk.score,
                rerank_score=float(index),
            )
            for index, chunk in enumerate(ordered[:top_n])
        ]


def test_rrf_missing_arm_is_absent_not_zero_filled() -> None:
    scores = reciprocal_rank_fusion(["a", "b"], ["b", "c"])

    assert scores["a"] == pytest.approx(1.0 / 61)
    assert scores["c"] == pytest.approx(1.0 / 62)
    assert scores["b"] == pytest.approx(1.0 / 62 + 1.0 / 61)
    only_dense = reciprocal_rank_fusion(["solo"], [])
    assert only_dense["solo"] == pytest.approx(1.0 / 61)
    only_bm25 = reciprocal_rank_fusion([], ["solo"])
    assert only_bm25["solo"] == pytest.approx(1.0 / 61)


def test_rrf_tie_breaks_by_chunk_id() -> None:
    scores = reciprocal_rank_fusion(["zeta"], ["alpha"])
    assert fused_chunk_order(scores) == ["alpha", "zeta"]


def test_lexical_gate_thresholds_and_empty_query() -> None:
    assert lexical_coverages(frozenset(), frozenset({"bonus"}), {}) is None
    passed = lexical_coverages(
        frozenset({"bonus", "point"}),
        frozenset({"bonus", "point", "award"}),
        {"bonus": 1.0, "point": 1.0},
    )
    assert passed == (1.0, 1.0)
    assert is_lexically_strong(
        bm25_score=3.9,
        coverage=passed[0],
        idf_coverage=passed[1],
        coverage_min=0.60,
        idf_coverage_min=0.40,
    )
    assert not is_lexically_strong(
        bm25_score=None,
        coverage=1.0,
        idf_coverage=1.0,
        coverage_min=0.60,
        idf_coverage_min=0.40,
    )


def test_oov_terms_stay_in_idf_denominator() -> None:
    query_terms = frozenset({"bonus", "point", "mars"})
    chunk_terms = frozenset({"bonus", "point"})
    idf = {"bonus": 1.0, "point": 1.0, "mars": 5.0}
    with_oov = lexical_coverages(query_terms, chunk_terms, idf)
    without_oov = lexical_coverages(
        frozenset({"bonus", "point"}),
        chunk_terms,
        {"bonus": 1.0, "point": 1.0},
    )
    assert with_oov == (2 / 3, 2 / 7)
    assert without_oov == (1.0, 1.0)
    assert with_oov[0] >= 0.60
    assert with_oov[1] < 0.40
    assert is_lexically_strong(
        bm25_score=6.0,
        coverage=without_oov[0],
        idf_coverage=without_oov[1],
        coverage_min=0.60,
        idf_coverage_min=0.40,
    )
    assert not is_lexically_strong(
        bm25_score=6.0,
        coverage=with_oov[0],
        idf_coverage=with_oov[1],
        coverage_min=0.60,
        idf_coverage_min=0.40,
    )


def test_zero_idf_denominator_rejects() -> None:
    terms = frozenset({"ubiquitous"})
    assert (
        lexical_coverages(terms, terms, {"ubiquitous": 0.0}) is None
    )


def test_pinning_relative_bm25_handles_empty_and_all_zero() -> None:
    assert pinning_relative_bm25(1.0, []) is None
    assert pinning_relative_bm25(1.0, [None, None]) is None
    assert pinning_relative_bm25(0.0, [0.0, 0.0]) is None
    assert pinning_relative_bm25(2.0, [4.0, 1.0]) == pytest.approx(0.5)


def test_keyword_index_idf_includes_oov_terms(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "hybrid-idf.json")
    index.upsert_document("spec", [("chunk", "bonus points awarded")])
    in_vocab = index.analyzer.analyze("bonus")[0]
    oov = "quokka"

    assert index.document_frequency(in_vocab) >= 1
    assert index.document_frequency(oov) == 0
    assert index.bm25_idf(oov) > index.bm25_idf(in_vocab)
    assert index.indexed_chunk_count() == 1


def test_pinning_restores_prefix_dropped_lexical_candidate() -> None:
    query = "bonus points awarded"
    wiki = _context(
        "wiki",
        text="Unrelated encyclopedia article about sports.",
        dense_score=0.21,
        bm25_score=None,
        fusion_score=0.02,
    )
    other = _context(
        "other",
        text="More unrelated sports commentary.",
        dense_score=0.18,
        bm25_score=None,
        fusion_score=0.016,
    )
    spec = _context(
        "spec",
        text="Bonus points awarded for visualization work.",
        dense_score=0.11,
        bm25_score=4.2,
        fusion_score=0.015,
        title="Project specification",
    )
    pool = [wiki, other, spec]
    service = RagService(
        FakeSearch([]),
        FakeLLM("unused"),
        keyword_index=FakeKeywordIndex([]),
        bm25_k1=1.5,
        bm25_b=0.75,
    )

    pinned = service._pin_lexical_candidate(query, pool, [wiki, other])

    assert [chunk.chunk_id for chunk in pinned] == ["wiki", "spec"]
    assert pinned[1].bm25_score == pytest.approx(4.2)


def test_pinning_restores_reranker_dropped_lexical_candidate() -> None:
    query = "bonus points awarded"
    spec = _context(
        "spec",
        text="Bonus points awarded for visualization work.",
        dense_score=0.11,
        bm25_score=5.0,
        fusion_score=0.03,
        title="Project specification",
    )
    wiki = _context(
        "wiki",
        text="Unrelated encyclopedia article about sports.",
        dense_score=0.21,
        bm25_score=None,
        fusion_score=0.02,
    )
    other = _context(
        "other",
        text="More unrelated sports commentary.",
        dense_score=0.18,
        bm25_score=None,
        fusion_score=0.016,
    )
    service = RagService(
        FakeSearch([]),
        FakeLLM("unused"),
        reranker=ReverseReranker(),
        keyword_index=FakeKeywordIndex([]),
        bm25_k1=1.5,
        bm25_b=0.75,
    )
    pool = [spec, wiki, other]
    selected = service.select_context(query, pool, top_k=2, use_reranker=True)
    assert [chunk.chunk_id for chunk in selected] == ["other", "wiki"]

    pinned = service._pin_lexical_candidate(query, pool, selected)

    assert "spec" in [chunk.chunk_id for chunk in pinned]
    assert pinned[-1].chunk_id == "spec" or pinned[0].chunk_id == "spec"


def _dense_fallback_chunk() -> RagContextChunk:
    return _context(
        "dense",
        text="Dense-only fallback chunk.",
        dense_score=0.44,
        bm25_score=None,
        fusion_score=None,
    )


def test_reranker_raises_on_none_score_but_accepts_hybrid_chunks(
    tmp_path: Path,
) -> None:
    reranker = CrossEncoderReranker(
        "missing-model",
        cache_dir=tmp_path,
        device="cpu",
    )
    none_chunk = SimpleNamespace(text="body", score=None, chunk_id="none")
    with pytest.raises(TypeError):
        reranker.rerank("query", [none_chunk], top_n=1)

    dense_only = _dense_fallback_chunk()
    hybrid_only = _context(
        "bm25-only",
        text="Bonus points awarded.",
        dense_score=None,
        bm25_score=3.9,
        fusion_score=1.0 / 61,
    )
    ranked_dense = reranker.rerank("query", [dense_only], top_n=1)
    ranked_hybrid = reranker.rerank("query", [hybrid_only], top_n=1)
    assert ranked_dense[0].chunk is dense_only
    assert ranked_hybrid[0].chunk is hybrid_only
    assert isinstance(hybrid_only.score, float)


def test_provenance_survives_retrieve_fuse_rerank() -> None:
    wiki = _low_score_record(1, 0.21, text="Unrelated wiki page.")
    spec = _low_score_record(
        2,
        0.11,
        title="Project specification",
        text="Bonus points awarded for visualization work.",
    )
    search = FakeSearch([wiki, spec], catalog=[wiki, spec])
    keyword_index = FakeKeywordIndex(
        [KeywordSearchHit(chunk_id="chunk-2", score=4.5, matched_terms=(), term_contributions={})]
    )
    service = RagService(
        search,
        FakeLLM("unused"),
        reranker=ReverseReranker(),
        keyword_index=keyword_index,
        bm25_k1=2.0,
        bm25_b=0.3,
    )
    fused = service._retrieve_candidates(
        original_query="bonus points awarded",
        dense_query="bonus points awarded",
    )
    spec_fused = next(chunk for chunk in fused if chunk.chunk_id == "chunk-2")
    assert spec_fused.bm25_score == pytest.approx(4.5)
    assert spec_fused.dense_score == pytest.approx(0.11)
    assert spec_fused.fusion_score is not None
    assert spec_fused.retrieval_sources == ("dense", "bm25")

    selected = service.select_context(
        "bonus points awarded",
        fused,
        top_k=2,
        use_reranker=True,
    )
    by_id = {chunk.chunk_id: chunk for chunk in selected}
    assert "chunk-2" in by_id
    restored = by_id["chunk-2"]
    assert restored.bm25_score == pytest.approx(4.5)
    assert restored.dense_score == pytest.approx(0.11)
    assert restored.fusion_score == spec_fused.fusion_score
    assert restored.retrieval_sources == ("dense", "bm25")
    assert restored.rerank_score is not None


def test_citation_one_maps_to_first_context_after_fusion_and_rerank() -> None:
    wiki = _low_score_record(1, 0.21, text="Unrelated wiki page.")
    spec = _low_score_record(
        2,
        0.11,
        title="Project specification",
        text="Bonus points awarded for visualization work.",
    )
    search = FakeSearch([wiki], catalog=[wiki, spec])
    keyword_index = FakeKeywordIndex(
        [KeywordSearchHit(chunk_id="chunk-2", score=4.5, matched_terms=(), term_contributions={})]
    )
    llm = FakeLLM(
        "",
        answers=[
            "Bonus points are awarded for visualization work [1].",
            "Bonus points are awarded for visualization work [1].",
        ],
    )
    service = RagService(
        search,
        llm,
        reranker=ReverseReranker(),
        keyword_index=keyword_index,
        bm25_k1=2.0,
        bm25_b=0.3,
    )

    outcome = service.generate(
        "bonus points awarded",
        top_k=2,
        use_reranker=True,
    )

    assert outcome.abstained is False
    first = outcome.context_chunks[0]
    cited = outcome.cited_chunks[0]
    assert cited.chunk_id == first.chunk_id
    assert cited.dense_score == first.dense_score
    assert cited.bm25_score == first.bm25_score
    assert cited.fusion_score == first.fusion_score
    assert cited.retrieval_sources == first.retrieval_sources
    assert "[1]" in outcome.answer


def test_rewritten_query_never_reaches_bm25_or_lexical_gate() -> None:
    wiki = _low_score_record(1, 0.12, text="Unrelated wiki page.")
    spec = _low_score_record(
        2,
        0.09,
        title="Project specification",
        text="Bonus points awarded for visualization work.",
    )
    search = FakeSearch([wiki], catalog=[wiki, spec])
    keyword_index = FakeKeywordIndex(
        [KeywordSearchHit(chunk_id="chunk-2", score=6.7, matched_terms=(), term_contributions={})]
    )
    llm = FakeLLM(
        "",
        answers=[
            "quantum superposition lecture notes",
            "Bonus points are awarded [1].",
            "Bonus points are awarded [1].",
        ],
    )
    service = RagService(
        search,
        llm,
        keyword_index=keyword_index,
        bm25_k1=2.0,
        bm25_b=0.3,
    )

    outcome = service.generate(
        "bonus points awarded",
        top_k=2,
        use_query_rewrite=True,
    )

    assert search.queries == ["quantum superposition lecture notes"]
    assert keyword_index.queries == ["bonus points awarded"]
    assert keyword_index.k1_values == [2.0]
    assert keyword_index.b_values == [0.3]
    assert outcome.abstained is False
    assert outcome.abstention_reason is None


def test_with_llm_keeps_hybrid_retriever() -> None:
    wiki = _low_score_record(1, 0.12, text="Unrelated wiki page.")
    spec = _low_score_record(
        2,
        0.09,
        title="Project specification",
        text="Bonus points awarded for visualization work.",
    )
    search = FakeSearch([wiki], catalog=[wiki, spec])
    keyword_index = FakeKeywordIndex(
        [KeywordSearchHit(chunk_id="chunk-2", score=6.7, matched_terms=(), term_contributions={})]
    )
    original = RagService(
        search,
        FakeLLM("unused"),
        keyword_index=keyword_index,
        bm25_k1=2.0,
        bm25_b=0.3,
    )
    switched = original.with_llm(
        FakeLLM(
            "",
            answers=[
                "Bonus points are awarded [1].",
                "Bonus points are awarded [1].",
            ],
        )
    )

    outcome = switched.generate("bonus points awarded", top_k=2)

    assert switched._keyword_index is keyword_index
    assert switched._bm25_k1 == 2.0
    assert switched._bm25_b == 0.3
    assert keyword_index.queries == ["bonus points awarded"]
    assert outcome.abstained is False
    assert outcome.abstention_reason is None
    assert any(chunk.chunk_id == "chunk-2" for chunk in outcome.context_chunks)


def test_hybrid_gate_admits_bm25_only_when_dense_is_weak() -> None:
    wiki = _low_score_record(1, 0.12, text="Unrelated wiki page.")
    spec = _low_score_record(
        2,
        0.09,
        title="Project specification",
        text="Bonus points awarded for visualization work.",
    )
    search = FakeSearch([wiki], catalog=[wiki, spec])
    keyword_index = FakeKeywordIndex(
        [KeywordSearchHit(chunk_id="chunk-2", score=6.7, matched_terms=(), term_contributions={})]
    )
    llm = FakeLLM(
        "",
        answers=[
            "Bonus points are awarded [1].",
            "Bonus points are awarded [1].",
        ],
    )
    service = RagService(
        search,
        llm,
        keyword_index=keyword_index,
        bm25_k1=1.5,
        bm25_b=0.75,
    )

    outcome = service.generate("bonus points awarded", top_k=2)

    assert outcome.abstained is False
    spec_hit = next(chunk for chunk in outcome.context_chunks if chunk.chunk_id == "chunk-2")
    assert spec_hit.dense_score is None
    assert spec_hit.bm25_score == pytest.approx(6.7)
    assert spec_hit.fusion_score is not None
    assert spec_hit.retrieval_sources == ("bm25",)


def test_dense_only_fallback_nulls_fusion_and_bm25() -> None:
    record = _low_score_record(1, 0.44, text="Dense-only fallback chunk.")
    service = RagService(FakeSearch([record]), FakeLLM("unused"))
    retrieved = service._retrieve_candidates(
        original_query="bonus points",
        dense_query="bonus points",
    )
    assert len(retrieved) == 1
    assert retrieved[0].dense_score == pytest.approx(0.44)
    assert retrieved[0].bm25_score is None
    assert retrieved[0].fusion_score is None
    assert retrieved[0].retrieval_sources == ("dense",)
    assert retrieved[0].retrieval_score == pytest.approx(0.44)


def test_real_index_oov_terms_keep_formula_from_passing(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "oov-gate.json")
    spec_text = "Bonus points are available for bonus opportunities."
    index.upsert_document("spec", [("chunk-2", spec_text)])
    for i in range(7):
        index.upsert_document(
            f"filler-{i}",
            [(f"filler-{i}", f"Unrelated sports encyclopedia article {i}.")],
        )
    service = RagService(
        FakeSearch([]),
        FakeLLM("unused"),
        keyword_index=index,
        bm25_k1=1.5,
        bm25_b=0.75,
    )
    chunk = _context(
        "chunk-2",
        text=spec_text,
        title="Project specification",
        dense_score=None,
        bm25_score=6.0,
        fusion_score=0.02,
    )
    positive = "How many extra points are available for bonus opportunities?"
    negative = (
        "How many extra points are available for bonus opportunities on Mars?"
    )
    assert service._is_formula_strong(positive, chunk)
    assert not service._is_formula_strong(negative, chunk)

    analyzer = TextAnalyzer(min_token_length=1)
    query_terms = frozenset(analyzer.analyze(negative))
    chunk_terms = frozenset(analyzer.analyze(spec_text))
    idf = {term: index.bm25_idf(term) for term in query_terms}
    full = lexical_coverages(query_terms, chunk_terms, idf)
    in_vocab = frozenset(
        term for term in query_terms if index.document_frequency(term) > 0
    )
    dropped = lexical_coverages(
        in_vocab,
        chunk_terms,
        {term: idf[term] for term in in_vocab},
    )
    assert full is not None and dropped is not None
    assert not is_lexically_strong(
        bm25_score=6.0,
        coverage=full[0],
        idf_coverage=full[1],
        coverage_min=0.60,
        idf_coverage_min=0.40,
    )
    assert is_lexically_strong(
        bm25_score=6.0,
        coverage=dropped[0],
        idf_coverage=dropped[1],
        coverage_min=0.60,
        idf_coverage_min=0.40,
    )


def test_hybrid_retrieve_against_real_bm25_index(tmp_path: Path) -> None:
    spec_text = "Bonus points awarded for visualization work."
    wiki = _low_score_record(1, 0.12, text="Unrelated wiki page.")
    spec = _low_score_record(
        2,
        0.09,
        title="Project specification",
        text=spec_text,
    )
    index = KeywordIndex(tmp_path / "hybrid-retrieve.json")
    index.upsert_document("document-1", [("chunk-1", wiki.text)])
    index.upsert_document("document-2", [("chunk-2", spec_text)])
    service = RagService(
        FakeSearch([wiki], catalog=[wiki, spec]),
        FakeLLM("unused"),
        keyword_index=index,
        bm25_k1=2.0,
        bm25_b=0.3,
    )
    fused = service._retrieve_candidates(
        original_query="bonus points awarded",
        dense_query="rewritten unrelated query",
    )
    spec_hit = next(chunk for chunk in fused if chunk.chunk_id == "chunk-2")
    wiki_hit = next(chunk for chunk in fused if chunk.chunk_id == "chunk-1")
    assert spec_hit.bm25_score is not None and spec_hit.bm25_score > 0
    assert spec_hit.dense_score is None
    assert spec_hit.fusion_score is not None
    assert spec_hit.retrieval_sources == ("bm25",)
    assert wiki_hit.dense_score == pytest.approx(0.12)
    assert wiki_hit.fusion_score is not None
