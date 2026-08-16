"""Persistent custom inverted index and TF-IDF cosine retrieval."""

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
from typing import Any

from app.retrieval import TextAnalyzer

_INDEX_VERSION = 1


class KeywordIndexError(RuntimeError):
    """Raised when the custom keyword index cannot be read or persisted."""


@dataclass(frozen=True, slots=True)
class KeywordSearchHit:
    """A scored chunk returned by TF-IDF retrieval."""

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


class KeywordIndex:
    """Thread-safe positional inverted index persisted as atomic JSON."""

    def __init__(
        self,
        path: str | Path,
        *,
        analyzer: TextAnalyzer | None = None,
    ) -> None:
        self.path = Path(path)
        self.analyzer = analyzer or TextAnalyzer()
        self._lock = RLock()
        self._postings: dict[str, dict[str, list[int]]] = {}
        self._chunks: dict[str, dict[str, Any]] = {}
        self._documents: dict[str, list[str]] = {}
        self._document_norms: dict[str, float] = {}
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
                isinstance(value, dict) for value in (postings, chunks, documents)
            ):
                raise KeywordIndexError("Invalid keyword-index structure.")
            self._postings = postings
            self._chunks = chunks
            self._documents = documents
            self._recompute_document_norms()
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
            raise KeywordIndexError("Chunk IDs must be unique within a document.")

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
            self._recompute_document_norms()

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
            self._recompute_document_norms()
            return removed

    @staticmethod
    def _idf(document_frequency: int, chunk_count: int) -> float:
        return math.log((chunk_count + 1) / (document_frequency + 1)) + 1.0

    def _recompute_document_norms(self) -> None:
        chunk_count = len(self._chunks)
        norms: dict[str, float] = {}
        for chunk_id, chunk_record in self._chunks.items():
            squared_sum = 0.0
            term_frequencies = chunk_record.get("term_frequencies", {})
            for term, raw_frequency in term_frequencies.items():
                document_frequency = len(self._postings.get(term, {}))
                idf = self._idf(document_frequency, chunk_count)
                weight = (1.0 + math.log(raw_frequency)) * idf
                squared_sum += weight * weight
            norms[chunk_id] = math.sqrt(squared_sum)
        self._document_norms = norms

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_limit: int = 200,
    ) -> list[KeywordSearchHit]:
        """Return approximate candidates reranked by exact TF-IDF cosine score."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than zero.")

        query_frequencies = Counter(self.analyzer.analyze(query))
        if not query_frequencies:
            return []

        with self._lock:
            chunk_count = len(self._chunks)
            query_weights: dict[str, float] = {}
            raw_contributions: dict[str, dict[str, float]] = defaultdict(dict)

            for term, query_frequency in query_frequencies.items():
                term_postings = self._postings.get(term)
                if not term_postings:
                    continue
                idf = self._idf(len(term_postings), chunk_count)
                query_weight = (1.0 + math.log(query_frequency)) * idf
                query_weights[term] = query_weight
                for chunk_id, positions in term_postings.items():
                    document_weight = (1.0 + math.log(len(positions))) * idf
                    raw_contributions[chunk_id][term] = query_weight * document_weight

            if not query_weights:
                return []

            query_norm = math.sqrt(
                sum(weight * weight for weight in query_weights.values())
            )
            candidate_count = max(top_k, candidate_limit)
            candidate_ids = sorted(
                raw_contributions,
                key=lambda chunk_id: (
                    -sum(raw_contributions[chunk_id].values()),
                    chunk_id,
                ),
            )[:candidate_count]

            hits: list[KeywordSearchHit] = []
            for chunk_id in candidate_ids:
                denominator = query_norm * self._document_norms.get(chunk_id, 0.0)
                if denominator <= 0:
                    continue
                contributions = {
                    term: value / denominator
                    for term, value in raw_contributions[chunk_id].items()
                }
                score = sum(contributions.values())
                hits.append(
                    KeywordSearchHit(
                        chunk_id=chunk_id,
                        score=score,
                        matched_terms=tuple(sorted(contributions)),
                        term_contributions=contributions,
                    )
                )

            hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
            return hits[:top_k]

    def stats(self) -> KeywordIndexStats:
        """Return collection-level inverted-index statistics."""

        with self._lock:
            return KeywordIndexStats(
                document_count=len(self._documents),
                chunk_count=len(self._chunks),
                vocabulary_size=len(self._postings),
                posting_count=sum(
                    len(term_postings) for term_postings in self._postings.values()
                ),
            )


@lru_cache(maxsize=32)
def open_keyword_index(path: str) -> KeywordIndex:
    """Reuse one thread-safe index instance per normalized storage path."""

    return KeywordIndex(Path(path))
