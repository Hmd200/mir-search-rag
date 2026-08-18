"""Tests for Rocchio pseudo-relevance feedback."""

from pathlib import Path

import pytest

from app.storage.keyword_index import KeywordIndex


def _mismatch_corpus(tmp_path: Path) -> KeywordIndex:
    index = KeywordIndex(tmp_path / "prf.json")
    index.upsert_document(
        "seed",
        [
            (
                "feedback",
                "neural neural neural ranking quokka quokka quokka",
            ),
            ("also-query", "neural ranking"),
        ],
    )
    index.upsert_document(
        "mismatch",
        [("mismatch", "quokka embeddings vector space")],
    )
    index.upsert_document(
        "noise",
        [("noise", "boolean transactions storage")],
    )
    return index


def test_prf_adds_terms_that_cooccur_in_feedback_chunks(
    tmp_path: Path,
) -> None:
    index = _mismatch_corpus(tmp_path)
    query = "neural"
    query_terms = set(index.analyzer.analyze(query))
    feedback_terms = set(
        index.analyzer.analyze("neural neural neural ranking quokka quokka quokka")
    )

    outcome = index.search_with_prf(
        query,
        top_k=10,
        feedback_docs=2,
        max_expansion_terms=10,
    )

    added = {item.term for item in outcome.expansion.added_terms}
    assert added
    assert added <= (feedback_terms - query_terms)
    assert "quokka" in added


def test_prf_surfaces_vocabulary_mismatch_document(
    tmp_path: Path,
) -> None:
    index = _mismatch_corpus(tmp_path)
    baseline = [hit.chunk_id for hit in index.search("neural", top_k=10)]
    outcome = index.search_with_prf(
        "neural",
        top_k=10,
        feedback_docs=2,
        max_expansion_terms=10,
    )
    expanded = [hit.chunk_id for hit in outcome.hits]

    assert "mismatch" not in baseline
    assert "mismatch" in expanded
    assert expanded != baseline


def test_expansion_metadata_reports_terms_and_feedback_chunks(
    tmp_path: Path,
) -> None:
    index = _mismatch_corpus(tmp_path)
    outcome = index.search_with_prf(
        "neural",
        top_k=10,
        feedback_docs=2,
        max_expansion_terms=10,
    )

    assert outcome.expansion.feedback_chunk_ids
    assert "feedback" in outcome.expansion.feedback_chunk_ids
    assert outcome.expansion.added_terms
    assert all(item.weight > 0 for item in outcome.expansion.added_terms)
    assert all(item.term for item in outcome.expansion.added_terms)


def test_beta_zero_matches_non_prf_search(tmp_path: Path) -> None:
    index = _mismatch_corpus(tmp_path)
    baseline = index.search("neural", top_k=10)
    outcome = index.search_with_prf(
        "neural",
        top_k=10,
        feedback_docs=3,
        max_expansion_terms=10,
        beta=0.0,
    )

    assert [hit.chunk_id for hit in outcome.hits] == [hit.chunk_id for hit in baseline]
    assert [hit.score for hit in outcome.hits] == [hit.score for hit in baseline]
    assert outcome.expansion.added_terms == ()


def test_prf_with_no_matches_returns_empty(tmp_path: Path) -> None:
    index = _mismatch_corpus(tmp_path)
    outcome = index.search_with_prf("xyzzyplugh", top_k=5)

    assert outcome.hits == ()
    assert outcome.expansion.added_terms == ()
    assert outcome.expansion.feedback_chunk_ids == ()


@pytest.mark.parametrize(
    "options",
    [
        {"alpha": -1.0},
        {"beta": -0.1},
        {"feedback_docs": 0},
    ],
)
def test_invalid_prf_parameters_raise(
    tmp_path: Path,
    options: dict[str, float | int],
) -> None:
    index = _mismatch_corpus(tmp_path)

    with pytest.raises(ValueError):
        index.search_with_prf("neural", top_k=5, **options)
