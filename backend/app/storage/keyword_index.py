"""Persistent custom inverted index with TF-IDF and BM25 retrieval."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Literal

from app.retrieval import TextAnalyzer

_INDEX_VERSION = 1
RetrievalMode = Literal["champion", "exact"]
ScoringMode = Literal["tfidf", "bm25"]


class KeywordIndexError(RuntimeError):
    """Raised when the custom keyword index cannot be read or persisted."""


@dataclass(frozen=True, slots=True)
class KeywordSearchHit:
    """A scored chunk returned by a keyword retrieval strategy."""

    chunk_id: str
    score: float
    matched_terms: tuple[str, ...]
    term_contributions: dict[str, float]


@dataclass(frozen=True, slots=True)
class KeywordIndexStats:
    """Collection statistics used by administration and visualization views."""

    document_count: int
    chunk_count: int
    vocabulary_size: int
    posting_count: int


@dataclass(frozen=True, slots=True)
class KeywordSearchDiagnostics:
    """Measurements describing the retrieval work performed for one query."""

    retrieval_mode: RetrievalMode
    champion_size: int | None
    total_postings_available: int
    postings_visited: int
    candidate_count: int
    reduction_percentage: float
    fallback_occurred: bool
    champion_rebuilt: bool
    exact_top_k_overlap: float | None
    search_latency_ms: float


@dataclass(frozen=True, slots=True)
class KeywordSearchOutcome:
    """Ranked hits accompanied by retrieval diagnostics."""

    hits: tuple[KeywordSearchHit, ...]
    diagnostics: KeywordSearchDiagnostics


@dataclass(frozen=True, slots=True)
class _CandidateSelection:
    chunk_ids: frozenset[str]
    total_postings: int
    postings_visited: int
    fallback_occurred: bool
    champion_rebuilt: bool


class KeywordIndex:
    """Thread-safe positional inverted index persisted as atomic JSON."""

    def __init__(
        self,
        path: str | Path,
        *,
        analyzer: TextAnalyzer | None = None,
        champion_size: int = 50,
    ) -> None:
        if champion_size < 1:
            raise ValueError(
                "champion_size must be greater than zero."
            )

        self.path = Path(path)
        self.analyzer = analyzer or TextAnalyzer()
        self._champion_size = champion_size
        self._lock = RLock()
        self._postings: dict[str, dict[str, list[int]]] = {}
        self._chunks: dict[str, dict[str, Any]] = {}
        self._documents: dict[str, list[str]] = {}
        self._document_norms: dict[str, float] = {}
        self._champion_lists: dict[str, list[str]] = {}
        self._postings_visited = 0

        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") != _INDEX_VERSION:
                raise KeywordIndexError("Unsupported keyword-index version.")

            postings = payload.get("postings", {})
            chunks = payload.get("chunks", {})
            documents = payload.get("documents", {})

            if not all(
                isinstance(value, dict)
                for value in (postings, chunks, documents)
            ):
                raise KeywordIndexError("Invalid keyword-index structure.")

            self._postings = postings
            self._chunks = chunks
            self._documents = documents
            self._recompute_derived_statistics()
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise KeywordIndexError(
                f"Could not load keyword index: {self.path}"
            ) from error

    def _persist_state(
        self,
        postings: dict[str, dict[str, list[int]]],
        chunks: dict[str, dict[str, Any]],
        documents: dict[str, list[str]],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        payload = {
            "version": _INDEX_VERSION,
            "postings": postings,
            "chunks": chunks,
            "documents": documents,
        }

        try:
            with temporary_path.open("w", encoding="utf-8") as output:
                json.dump(
                    payload,
                    output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                output.flush()
            temporary_path.replace(self.path)
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            raise KeywordIndexError(
                f"Could not persist keyword index: {self.path}"
            ) from error

    @staticmethod
    def _remove_document_from_state(
        document_id: str,
        postings: dict[str, dict[str, list[int]]],
        chunks: dict[str, dict[str, Any]],
        documents: dict[str, list[str]],
    ) -> bool:
        chunk_ids = documents.pop(document_id, None)
        if chunk_ids is None:
            return False

        for chunk_id in chunk_ids:
            chunk_record = chunks.pop(chunk_id, None)
            if chunk_record is None:
                continue

            term_frequencies = chunk_record.get("term_frequencies", {})
            for term in term_frequencies:
                term_postings = postings.get(term)
                if term_postings is None:
                    continue

                term_postings.pop(chunk_id, None)
                if not term_postings:
                    postings.pop(term, None)

        return True

    def upsert_document(
        self,
        document_id: str,
        chunks: Iterable[tuple[str, str]],
    ) -> None:
        """Atomically replace all indexed chunks belonging to one document."""

        chunk_values = list(chunks)
        if not chunk_values:
            raise KeywordIndexError("Cannot index a document without chunks.")

        chunk_ids = [chunk_id for chunk_id, _ in chunk_values]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise KeywordIndexError(
                "Chunk IDs must be unique within a document."
            )

        with self._lock:
            postings = deepcopy(self._postings)
            chunk_records = deepcopy(self._chunks)
            documents = deepcopy(self._documents)

            self._remove_document_from_state(
                document_id,
                postings,
                chunk_records,
                documents,
            )

            for chunk_id, text in chunk_values:
                analyzed = self.analyzer.analyze_with_positions(text)
                positions_by_term: dict[str, list[int]] = defaultdict(list)

                for token in analyzed:
                    positions_by_term[token.term].append(token.position)

                term_frequencies = {
                    term: len(positions)
                    for term, positions in positions_by_term.items()
                }
                chunk_records[chunk_id] = {
                    "document_id": document_id,
                    "term_frequencies": term_frequencies,
                    "length": len(analyzed),
                }

                for term, positions in positions_by_term.items():
                    postings.setdefault(term, {})[chunk_id] = positions

            documents[document_id] = chunk_ids
            self._persist_state(postings, chunk_records, documents)

            self._postings = postings
            self._chunks = chunk_records
            self._documents = documents
            self._recompute_derived_statistics()

    def delete_document(self, document_id: str) -> bool:
        """Atomically remove every posting belonging to a document."""

        with self._lock:
            if document_id not in self._documents:
                return False

            postings = deepcopy(self._postings)
            chunks = deepcopy(self._chunks)
            documents = deepcopy(self._documents)

            removed = self._remove_document_from_state(
                document_id,
                postings,
                chunks,
                documents,
            )
            self._persist_state(postings, chunks, documents)

            self._postings = postings
            self._chunks = chunks
            self._documents = documents
            self._recompute_derived_statistics()

            return removed

    @staticmethod
    def _idf(document_frequency: int, chunk_count: int) -> float:
        return (
            math.log(
                (chunk_count + 1) / (document_frequency + 1)
            )
            + 1.0
        )

    @staticmethod
    def _bm25_idf(
        document_frequency: int,
        chunk_count: int,
    ) -> float:
        """Return Robertson-Sparck Jones IDF with a positive lower bound."""

        return math.log(
            1.0
            + (
                chunk_count
                - document_frequency
                + 0.5
            )
            / (document_frequency + 0.5)
        )

    def _recompute_document_norms(self) -> None:
        chunk_count = len(self._chunks)
        norms: dict[str, float] = {}

        for chunk_id, chunk_record in self._chunks.items():
            squared_sum = 0.0
            term_frequencies = chunk_record.get(
                "term_frequencies",
                {},
            )

            for term, raw_frequency in term_frequencies.items():
                document_frequency = len(
                    self._postings.get(term, {})
                )
                idf = self._idf(
                    document_frequency,
                    chunk_count,
                )
                weight = (
                    1.0 + math.log(raw_frequency)
                ) * idf
                squared_sum += weight * weight

            norms[chunk_id] = math.sqrt(squared_sum)

        self._document_norms = norms

    def _recompute_derived_statistics(self) -> None:
        self._recompute_document_norms()
        champion_lists: dict[str, list[str]] = {}

        for term, postings in self._postings.items():
            ranked = sorted(
                postings,
                key=lambda chunk_id: (
                    -len(postings[chunk_id]),
                    chunk_id,
                ),
            )
            champion_lists[term] = ranked[: self._champion_size]

        self._champion_lists = champion_lists

    def _average_chunk_length(self) -> float:
        if not self._chunks:
            return 0.0

        return sum(
            max(
                0,
                int(record.get("length", 0)),
            )
            for record in self._chunks.values()
        ) / len(self._chunks)

    @staticmethod
    def _validate_search_options(
        *,
        top_k: int,
        retrieval_mode: str,
    ) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )
        if retrieval_mode not in {"champion", "exact"}:
            raise ValueError(
                "retrieval_mode must be 'champion' or 'exact'."
            )

    def _champion_strength(
        self,
        term: str,
        chunk_id: str,
        scoring: ScoringMode,
        *,
        k1: float,
        b: float,
        average_length: float,
    ) -> float:
        """Return the term-specific strength of one posting."""

        positions = self._postings[term][chunk_id]
        term_frequency = len(positions)
        chunk_count = len(self._chunks)
        document_frequency = len(self._postings[term])

        if scoring == "tfidf":
            document_norm = self._document_norms.get(
                chunk_id,
                0.0,
            )
            if document_norm <= 0:
                return 0.0

            idf = self._idf(
                document_frequency,
                chunk_count,
            )
            document_weight = (
                1.0 + math.log(term_frequency)
            ) * idf
            return document_weight / document_norm

        if average_length <= 0:
            return 0.0

        document_length = max(
            0,
            int(
                self._chunks[chunk_id].get(
                    "length",
                    0,
                )
            ),
        )
        length_factor = k1 * (
            1.0
            - b
            + b
            * document_length
            / average_length
        )
        idf = self._bm25_idf(
            document_frequency,
            chunk_count,
        )
        return (
            idf
            * term_frequency
            * (k1 + 1.0)
            / (term_frequency + length_factor)
        )

    def _get_champion_lists(
        self,
        scoring: ScoringMode,
        *,
        k1: float,
        b: float,
    ) -> tuple[dict[str, tuple[str, ...]], bool]:
        """Return TF-ranked champion lists built from constructor size."""

        if not self._champion_lists and self._postings:
            self._recompute_derived_statistics()
            rebuilt = True
        else:
            rebuilt = False

        return (
            {
                term: tuple(chunk_ids)
                for term, chunk_ids in self._champion_lists.items()
            },
            rebuilt,
        )

    def _select_candidates(
        self,
        query_terms: Iterable[str],
        *,
        scoring: ScoringMode,
        retrieval_mode: RetrievalMode,
        top_k: int,
        fallback: bool,
        k1: float,
        b: float,
    ) -> _CandidateSelection:
        """Select documents before full TF-IDF or BM25 scoring."""

        matched_terms = tuple(
            term
            for term in query_terms
            if term in self._postings
        )
        total_postings = sum(
            len(self._postings[term])
            for term in matched_terms
        )

        if not matched_terms:
            return _CandidateSelection(
                chunk_ids=frozenset(),
                total_postings=0,
                postings_visited=0,
                fallback_occurred=False,
                champion_rebuilt=False,
            )

        if retrieval_mode == "exact":
            candidate_ids = {
                chunk_id
                for term in matched_terms
                for chunk_id in self._postings[term]
            }
            return _CandidateSelection(
                chunk_ids=frozenset(candidate_ids),
                total_postings=total_postings,
                postings_visited=total_postings,
                fallback_occurred=False,
                champion_rebuilt=False,
            )

        champion_lists, rebuilt = self._get_champion_lists(
            scoring,
            k1=k1,
            b=b,
        )

        candidate_ids: set[str] = set()
        postings_visited = 0

        for term in matched_terms:
            champions = champion_lists.get(term, ())
            postings_visited += len(champions)
            candidate_ids.update(champions)

        fallback_occurred = (
            fallback
            and len(candidate_ids) < top_k
        )

        if fallback_occurred:
            candidate_ids = set()
            postings_visited = 0
            for term in matched_terms:
                postings = self._postings[term]
                postings_visited += len(postings)
                candidate_ids.update(postings)

        return _CandidateSelection(
            chunk_ids=frozenset(candidate_ids),
            total_postings=total_postings,
            postings_visited=min(
                postings_visited,
                total_postings,
            ),
            fallback_occurred=fallback_occurred,
            champion_rebuilt=rebuilt,
        )

    def _score_tfidf_candidates(
        self,
        query_frequencies: Counter[str],
        candidate_ids: Iterable[str],
        top_k: int,
    ) -> list[KeywordSearchHit]:
        """Calculate full cosine scores only for selected candidates."""

        chunk_count = len(self._chunks)
        query_weights: dict[str, float] = {}

        for term, query_frequency in query_frequencies.items():
            postings = self._postings.get(term)
            if not postings:
                continue

            idf = self._idf(
                len(postings),
                chunk_count,
            )
            query_weights[term] = (
                1.0 + math.log(query_frequency)
            ) * idf

        if not query_weights:
            return []

        query_norm = math.sqrt(
            sum(
                weight * weight
                for weight in query_weights.values()
            )
        )
        hits: list[KeywordSearchHit] = []

        for chunk_id in candidate_ids:
            denominator = (
                query_norm
                * self._document_norms.get(
                    chunk_id,
                    0.0,
                )
            )
            if denominator <= 0:
                continue

            contributions: dict[str, float] = {}

            for term, query_weight in query_weights.items():
                positions = self._postings[term].get(
                    chunk_id
                )
                if not positions:
                    continue

                idf = self._idf(
                    len(self._postings[term]),
                    chunk_count,
                )
                document_weight = (
                    1.0 + math.log(len(positions))
                ) * idf
                contributions[term] = (
                    query_weight
                    * document_weight
                    / denominator
                )

            if contributions:
                hits.append(
                    KeywordSearchHit(
                        chunk_id=chunk_id,
                        score=sum(
                            contributions.values()
                        ),
                        matched_terms=tuple(
                            sorted(contributions)
                        ),
                        term_contributions=contributions,
                    )
                )

        hits.sort(
            key=lambda hit: (
                -hit.score,
                hit.chunk_id,
            )
        )
        return hits[:top_k]

    def _score_bm25_candidates(
        self,
        query_frequencies: Counter[str],
        candidate_ids: Iterable[str],
        top_k: int,
        *,
        k1: float,
        b: float,
    ) -> list[KeywordSearchHit]:
        """Calculate full BM25 scores only for selected candidates."""

        chunk_count = len(self._chunks)
        average_length = self._average_chunk_length()

        if chunk_count == 0 or average_length <= 0:
            return []

        hits: list[KeywordSearchHit] = []

        for chunk_id in candidate_ids:
            document_length = max(
                0,
                int(
                    self._chunks[chunk_id].get(
                        "length",
                        0,
                    )
                ),
            )
            contributions: dict[str, float] = {}

            for term, query_frequency in query_frequencies.items():
                postings = self._postings.get(term)
                if not postings:
                    continue

                positions = postings.get(chunk_id)
                if not positions:
                    continue

                term_frequency = len(positions)
                length_factor = k1 * (
                    1.0
                    - b
                    + b
                    * document_length
                    / average_length
                )
                idf = self._bm25_idf(
                    len(postings),
                    chunk_count,
                )
                contributions[term] = (
                    query_frequency
                    * idf
                    * term_frequency
                    * (k1 + 1.0)
                    / (
                        term_frequency
                        + length_factor
                    )
                )

            if contributions:
                hits.append(
                    KeywordSearchHit(
                        chunk_id=chunk_id,
                        score=sum(
                            contributions.values()
                        ),
                        matched_terms=tuple(
                            sorted(contributions)
                        ),
                        term_contributions=contributions,
                    )
                )

        hits.sort(
            key=lambda hit: (
                -hit.score,
                hit.chunk_id,
            )
        )
        return hits[:top_k]

    @staticmethod
    def _top_k_overlap(
        approximate_hits: Iterable[KeywordSearchHit],
        exact_hits: Iterable[KeywordSearchHit],
        top_k: int,
    ) -> float:
        approximate_ids = {
            hit.chunk_id
            for hit in approximate_hits
        }
        exact_ids = {
            hit.chunk_id
            for hit in exact_hits
        }
        denominator = min(top_k, len(exact_ids))

        if denominator == 0:
            return 1.0

        return len(
            approximate_ids & exact_ids
        ) / denominator

    @staticmethod
    def _build_diagnostics(
        *,
        retrieval_mode: RetrievalMode,
        champion_size: int,
        selection: _CandidateSelection,
        overlap: float | None,
        latency_ms: float,
    ) -> KeywordSearchDiagnostics:
        if selection.total_postings:
            reduction_percentage = (
                100.0
                * (
                    selection.total_postings
                    - selection.postings_visited
                )
                / selection.total_postings
            )
        else:
            reduction_percentage = 0.0

        return KeywordSearchDiagnostics(
            retrieval_mode=retrieval_mode,
            champion_size=(
                champion_size
                if retrieval_mode == "champion"
                else None
            ),
            total_postings_available=(
                selection.total_postings
            ),
            postings_visited=(
                selection.postings_visited
            ),
            candidate_count=len(
                selection.chunk_ids
            ),
            reduction_percentage=(
                reduction_percentage
            ),
            fallback_occurred=(
                selection.fallback_occurred
            ),
            champion_rebuilt=(
                selection.champion_rebuilt
            ),
            exact_top_k_overlap=overlap,
            search_latency_ms=latency_ms,
        )

    def search_detailed(
        self,
        query: str,
        *,
        top_k: int = 10,
        retrieval_mode: RetrievalMode = "champion",
        champion_size: int | None = None,
        fallback: bool = True,
        compare_exact: bool = False,
        use_champions: bool = True,
    ) -> KeywordSearchOutcome:
        """Run TF-IDF retrieval and return measurable diagnostics."""

        if not use_champions:
            retrieval_mode = "exact"
        if champion_size is not None and champion_size < 1:
            raise ValueError(
                "champion_size must be greater than zero."
            )

        self._validate_search_options(
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        query_frequencies = Counter(
            self.analyzer.analyze(query)
        )
        started = perf_counter()

        with self._lock:
            selection = self._select_candidates(
                query_frequencies,
                scoring="tfidf",
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                fallback=fallback,
                k1=1.5,
                b=0.75,
            )
            self._postings_visited = selection.postings_visited
            hits = self._score_tfidf_candidates(
                query_frequencies,
                selection.chunk_ids,
                top_k,
            )
            latency_ms = (
                perf_counter() - started
            ) * 1000

            overlap: float | None = None
            if compare_exact:
                if retrieval_mode == "exact":
                    overlap = 1.0
                else:
                    exact_selection = self._select_candidates(
                        query_frequencies,
                        scoring="tfidf",
                        retrieval_mode="exact",
                        top_k=top_k,
                        fallback=False,
                        k1=1.5,
                        b=0.75,
                    )
                    exact_hits = self._score_tfidf_candidates(
                        query_frequencies,
                        exact_selection.chunk_ids,
                        top_k,
                    )
                    overlap = self._top_k_overlap(
                        hits,
                        exact_hits,
                        top_k,
                    )

        diagnostics = self._build_diagnostics(
            retrieval_mode=retrieval_mode,
            champion_size=self._champion_size,
            selection=selection,
            overlap=overlap,
            latency_ms=latency_ms,
        )
        return KeywordSearchOutcome(
            hits=tuple(hits),
            diagnostics=diagnostics,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        use_champions: bool = True,
        champion_size: int | None = None,
        retrieval_mode: RetrievalMode = "champion",
        fallback: bool = True,
        candidate_limit: int | None = None,
    ) -> list[KeywordSearchHit]:
        """Return TF-IDF hits using champion or exact retrieval.

        candidate_limit remains temporarily supported as a legacy alias
        for champion_size validation. Retrieval always uses the
        constructor champion_size.
        """

        if candidate_limit is not None and candidate_limit < 1:
            raise ValueError(
                "champion_size must be greater than zero."
            )

        outcome = self.search_detailed(
            query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            champion_size=champion_size,
            fallback=fallback,
            use_champions=use_champions,
        )
        return list(outcome.hits)

    def search_bm25_detailed(
        self,
        query: str,
        *,
        top_k: int = 10,
        retrieval_mode: RetrievalMode = "champion",
        champion_size: int | None = None,
        fallback: bool = True,
        compare_exact: bool = False,
        use_champions: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> KeywordSearchOutcome:
        """Run BM25 retrieval and return measurable diagnostics."""

        if not use_champions:
            retrieval_mode = "exact"
        if champion_size is not None and champion_size < 1:
            raise ValueError(
                "champion_size must be greater than zero."
            )

        self._validate_search_options(
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        if k1 <= 0:
            raise ValueError(
                "k1 must be greater than zero."
            )
        if not 0.0 <= b <= 1.0:
            raise ValueError(
                "b must be between zero and one."
            )

        query_frequencies = Counter(
            self.analyzer.analyze(query)
        )
        started = perf_counter()

        with self._lock:
            selection = self._select_candidates(
                query_frequencies,
                scoring="bm25",
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                fallback=fallback,
                k1=k1,
                b=b,
            )
            self._postings_visited = selection.postings_visited
            hits = self._score_bm25_candidates(
                query_frequencies,
                selection.chunk_ids,
                top_k,
                k1=k1,
                b=b,
            )
            latency_ms = (
                perf_counter() - started
            ) * 1000

            overlap: float | None = None
            if compare_exact:
                if retrieval_mode == "exact":
                    overlap = 1.0
                else:
                    exact_selection = self._select_candidates(
                        query_frequencies,
                        scoring="bm25",
                        retrieval_mode="exact",
                        top_k=top_k,
                        fallback=False,
                        k1=k1,
                        b=b,
                    )
                    exact_hits = self._score_bm25_candidates(
                        query_frequencies,
                        exact_selection.chunk_ids,
                        top_k,
                        k1=k1,
                        b=b,
                    )
                    overlap = self._top_k_overlap(
                        hits,
                        exact_hits,
                        top_k,
                    )

        diagnostics = self._build_diagnostics(
            retrieval_mode=retrieval_mode,
            champion_size=self._champion_size,
            selection=selection,
            overlap=overlap,
            latency_ms=latency_ms,
        )
        return KeywordSearchOutcome(
            hits=tuple(hits),
            diagnostics=diagnostics,
        )

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 10,
        use_champions: bool = True,
        champion_size: int | None = None,
        retrieval_mode: RetrievalMode = "champion",
        fallback: bool = True,
        candidate_limit: int | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[KeywordSearchHit]:
        """Return BM25 hits using champion or exact retrieval.

        candidate_limit remains temporarily supported as a legacy alias
        for champion_size validation. Retrieval always uses the
        constructor champion_size.
        """

        if candidate_limit is not None and candidate_limit < 1:
            raise ValueError(
                "champion_size must be greater than zero."
            )

        outcome = self.search_bm25_detailed(
            query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            champion_size=champion_size,
            fallback=fallback,
            use_champions=use_champions,
            k1=k1,
            b=b,
        )
        return list(outcome.hits)

    @property
    def postings_visited(self) -> int:
        """Return how many postings the most recent search examined."""

        return self._postings_visited

    def stats(self) -> KeywordIndexStats:
        """Return collection-level inverted-index statistics."""

        with self._lock:
            return KeywordIndexStats(
                document_count=len(self._documents),
                chunk_count=len(self._chunks),
                vocabulary_size=len(self._postings),
                posting_count=sum(
                    len(term_postings)
                    for term_postings
                    in self._postings.values()
                ),
            )


@lru_cache(maxsize=32)
def open_keyword_index(path: str) -> KeywordIndex:
    """Reuse one thread-safe index instance per normalized storage path."""

    return KeywordIndex(Path(path))