"""Tests for champion-list inexact top-K retrieval."""

from pathlib import Path

from app.storage.keyword_index import KeywordIndex


def _graded_signal_index(
    tmp_path: Path,
    *,
    champion_size: int,
) -> KeywordIndex:
    index = KeywordIndex(
        tmp_path / "champions.json",
        champion_size=champion_size,
    )
    chunks = [
        (
            f"chunk-{i:02d}",
            "alpha beta gamma " + " ".join(["signal"] * i),
        )
        for i in range(1, 21)
    ]
    index.upsert_document("corpus", chunks)
    return index


def test_champion_search_visits_fewer_postings(
    tmp_path: Path,
) -> None:
    index = _graded_signal_index(tmp_path, champion_size=8)

    index.search("signal", top_k=3, use_champions=True)
    champion_visited = index.postings_visited

    index.search("signal", top_k=3, use_champions=False)
    exact_visited = index.postings_visited

    assert champion_visited < exact_visited
    assert champion_visited > 0
    assert exact_visited > 0


def test_champion_top_three_matches_exact_on_twenty_chunks(
    tmp_path: Path,
) -> None:
    index = _graded_signal_index(tmp_path, champion_size=8)

    champion_hits = index.search(
        "signal",
        top_k=3,
        use_champions=True,
    )
    exact_hits = index.search(
        "signal",
        top_k=3,
        use_champions=False,
    )

    assert [hit.chunk_id for hit in champion_hits] == [
        hit.chunk_id for hit in exact_hits
    ]
    assert [hit.score for hit in champion_hits] == [
        hit.score for hit in exact_hits
    ]
    assert len(champion_hits) == 3


def test_fallback_matches_exact_search_for_a_rare_term(
    tmp_path: Path,
) -> None:
    index = KeywordIndex(
        tmp_path / "rare.json",
        champion_size=2,
    )
    index.upsert_document(
        "common",
        [
            (f"common-{i}", "signal ranking")
            for i in range(10)
        ],
    )
    index.upsert_document(
        "rare",
        [
            ("rare-a", "quokka appears once"),
            ("rare-b", "quokka appears twice quokka"),
            ("rare-c", "quokka appears thrice quokka quokka"),
        ],
    )

    champion_hits = index.search(
        "quokka",
        top_k=10,
        use_champions=True,
    )
    exact_hits = index.search(
        "quokka",
        top_k=10,
        use_champions=False,
    )

    assert champion_hits
    assert [hit.chunk_id for hit in champion_hits] == [
        hit.chunk_id for hit in exact_hits
    ]
    assert [hit.score for hit in champion_hits] == [
        hit.score for hit in exact_hits
    ]


def test_deleting_a_document_removes_chunks_from_champion_lists(
    tmp_path: Path,
) -> None:
    index = KeywordIndex(
        tmp_path / "delete-champions.json",
        champion_size=10,
    )
    index.upsert_document(
        "keep",
        [("keep-chunk", "signal ranking retrieval")],
    )
    index.upsert_document(
        "drop",
        [("drop-chunk", "signal ranking retrieval")],
    )

    champion_chunk_ids = {
        chunk_id
        for chunk_ids in index._champion_lists.values()
        for chunk_id in chunk_ids
    }
    assert "drop-chunk" in champion_chunk_ids
    assert "keep-chunk" in champion_chunk_ids

    index.delete_document("drop")

    remaining_chunk_ids = {
        chunk_id
        for chunk_ids in index._champion_lists.values()
        for chunk_id in chunk_ids
    }
    assert "drop-chunk" not in remaining_chunk_ids
    assert "keep-chunk" in remaining_chunk_ids
