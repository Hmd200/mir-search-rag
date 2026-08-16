"""Tests for the persistent custom inverted index and TF-IDF search."""

from pathlib import Path

import pytest

from app.retrieval import TextAnalyzer
from app.storage.keyword_index import KeywordIndex


def test_analyzer_normalizes_stop_words_and_word_forms() -> None:
    analyzer = TextAnalyzer()

    terms = analyzer.analyze("The RETRIEVAL systems are retrieving documents.")

    assert "the" not in terms
    assert terms.count("retriev") == 2
    assert "system" in terms
    assert "document" in terms


def test_tfidf_cosine_ranks_the_most_relevant_chunk_first(
    tmp_path: Path,
) -> None:
    index = KeywordIndex(tmp_path / "keyword-index.json")
    index.upsert_document(
        "document-a",
        [
            ("chunk-a1", "vector retrieval retrieval cosine ranking"),
            ("chunk-a2", "database transactions and storage"),
        ],
    )
    index.upsert_document(
        "document-b",
        [("chunk-b1", "probabilistic BM25 ranking model")],
    )

    hits = index.search("cosine retrieval", top_k=3)

    assert hits[0].chunk_id == "chunk-a1"
    assert hits[0].score > 0
    assert "retriev" in hits[0].matched_terms
    assert set(hits[0].term_contributions) == {"cosin", "retriev"}
    assert index.stats().document_count == 2
    assert index.stats().chunk_count == 3


def test_index_persists_reloads_and_deletes_by_document(tmp_path: Path) -> None:
    index_path = tmp_path / "persistent.json"
    index = KeywordIndex(index_path)
    index.upsert_document(
        "document-a",
        [("chunk-a", "neural semantic retrieval")],
    )
    index.upsert_document(
        "document-b",
        [("chunk-b", "classical boolean retrieval")],
    )

    reloaded = KeywordIndex(index_path)
    assert reloaded.search("neural")[0].chunk_id == "chunk-a"

    assert reloaded.delete_document("document-a") is True
    assert reloaded.search("neural") == []
    assert reloaded.search("boolean")[0].chunk_id == "chunk-b"
    assert reloaded.stats().document_count == 1

    after_delete = KeywordIndex(index_path)
    assert after_delete.stats().chunk_count == 1
    assert after_delete.delete_document("missing") is False


def test_upsert_replaces_old_document_postings(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "upsert.json")
    index.upsert_document("document", [("old-chunk", "obsolete terminology")])
    index.upsert_document("document", [("new-chunk", "modern retrieval")])

    assert index.search("obsolete") == []
    assert index.search("modern")[0].chunk_id == "new-chunk"
    assert index.stats().chunk_count == 1


@pytest.mark.parametrize(
    ("top_k", "candidate_limit"),
    [(0, 10), (10, 0)],
)
def test_invalid_search_limits_are_rejected(
    tmp_path: Path,
    top_k: int,
    candidate_limit: int,
) -> None:
    index = KeywordIndex(tmp_path / "limits.json")
    index.upsert_document("document", [("chunk", "retrieval")])

    with pytest.raises(ValueError):
        index.search(
            "retrieval",
            top_k=top_k,
            candidate_limit=candidate_limit,
        )
