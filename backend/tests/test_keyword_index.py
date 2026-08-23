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


def test_invalid_search_limits_are_rejected(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "limits.json")
    index.upsert_document("document", [("chunk", "retrieval")])

    with pytest.raises(ValueError):
        index.search("retrieval", top_k=0)


def test_bm25_prefers_rare_terms_and_reports_contributions(
    tmp_path: Path,
) -> None:
    index = KeywordIndex(tmp_path / "bm25-rare.json")
    index.upsert_document(
        "document-a",
        [("chunk-common", "retrieval ranking")],
    )
    index.upsert_document(
        "document-b",
        [("chunk-rare", "retrieval ranking quokka")],
    )
    index.upsert_document(
        "document-c",
        [("chunk-other", "retrieval storage")],
    )

    hits = index.search_bm25("retrieval quokka", top_k=3)

    assert hits[0].chunk_id == "chunk-rare"
    assert hits[0].matched_terms == ("quokka", "retriev")
    assert hits[0].term_contributions["quokka"] > hits[0].term_contributions["retriev"]


def test_bm25_saturates_repeated_term_frequency(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "bm25-saturation.json")
    index.upsert_document(
        "document",
        [
            ("single", "signal"),
            ("repeated", "signal " * 10),
        ],
    )

    hits = index.search_bm25("signal", top_k=2, b=0.0)
    scores = {hit.chunk_id: hit.score for hit in hits}

    assert scores["repeated"] > scores["single"]
    assert scores["repeated"] < scores["single"] * 2.5


def test_bm25_normalizes_chunk_length(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "bm25-length.json")
    index.upsert_document(
        "document",
        [
            ("short", "signal"),
            (
                "long",
                (
                    "signal alpha beta gamma delta epsilon zeta eta theta iota "
                    "kappa lambda mu nu xi omicron pi rho sigma tau"
                ),
            ),
        ],
    )

    hits = index.search_bm25("signal", top_k=2, b=1.0)

    assert [hit.chunk_id for hit in hits] == ["short", "long"]


def test_bm25_top_k_order_is_deterministic(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "bm25-order.json")
    index.upsert_document(
        "document",
        [
            ("chunk-b", "equal signal"),
            ("chunk-a", "equal signal"),
        ],
    )

    hits = index.search_bm25("signal", top_k=1)

    assert [hit.chunk_id for hit in hits] == ["chunk-a"]


@pytest.mark.parametrize(
    ("k1", "b"),
    [(0.0, 0.75), (-1.0, 0.75), (1.5, -0.1), (1.5, 1.1)],
)
def test_invalid_bm25_tunables_are_rejected(
    tmp_path: Path,
    k1: float,
    b: float,
) -> None:
    index = KeywordIndex(tmp_path / "bm25-tunables.json")
    index.upsert_document("document", [("chunk", "signal")])

    with pytest.raises(ValueError):
        index.search_bm25(
            "signal",
            k1=k1,
            b=b,
        )


def test_bm25_idf_is_defined_for_oov_terms(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "bm25-idf-oov.json")
    index.upsert_document("document", [("chunk", "signal retrieval")])

    assert index.document_frequency("signal") == 1
    assert index.document_frequency("quokka") == 0
    assert index.bm25_idf("quokka") > 0.0
    assert index.indexed_chunk_count() == 1
