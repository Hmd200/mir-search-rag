"""Lazy cross-encoder reranking for retrieved search chunks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from sentence_transformers import CrossEncoder

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_MAX_RERANK_CANDIDATES = 25
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True, slots=True)
class RerankResult:
    """One chunk after cross-encoder scoring."""

    chunk: Any
    retrieval_score: float
    rerank_score: float | None


class CrossEncoderReranker:
    """Score query-document pairs jointly with a cross-encoder.

    The bi-encoder used for retrieval embeds the query and each chunk
    independently, then compares vectors (fast, approximate). A
    cross-encoder reads the query and chunk together and produces one
    relevance score per pair, which is slower but typically more
    accurate — so it is applied only to a short retrieved candidate list.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        cache_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._model: CrossEncoder | None = None
        self._load_failed = False
        self._lock = RLock()

    def _get_model(self) -> CrossEncoder | None:
        """Load the CrossEncoder on first use; never at import time."""

        with self._lock:
            if self._load_failed:
                return None
            if self._model is None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self._model = CrossEncoder(
                        self.model_name,
                        device=self.device,
                        cache_folder=str(self.cache_dir),
                    )
                except TypeError:
                    # Older sentence-transformers builds omit cache_folder.
                    try:
                        self._model = CrossEncoder(
                            self.model_name,
                            device=self.device,
                        )
                    except Exception as error:
                        logger.warning(
                            "Could not load cross-encoder '%s'; skipping rerank.",
                            self.model_name,
                        )
                        logger.debug("Cross-encoder load failed.", exc_info=error)
                        self._load_failed = True
                        return None
                except Exception as error:
                    logger.warning(
                        "Could not load cross-encoder '%s'; skipping rerank.",
                        self.model_name,
                    )
                    logger.debug("Cross-encoder load failed.", exc_info=error)
                    self._load_failed = True
                    return None
            return self._model

    def rerank(
        self,
        query: str,
        chunks: Sequence[Any],
        top_n: int = 10,
    ) -> list[RerankResult]:
        """Return chunks sorted by cross-encoder score, truncated to top_n.

        Each chunk is paired with the query as [query, chunk.text] and
        scored in one forward pass. At most 25 chunks are scored, to keep
        latency bounded even if the caller passes a longer list.
        """

        originals = list(chunks)
        if not originals or top_n <= 0:
            return []

        def _passthrough() -> list[RerankResult]:
            # Keep the retrieval order; do not invent a cross-encoder score.
            return [
                RerankResult(
                    chunk=chunk,
                    retrieval_score=float(
                        getattr(chunk, "score", 0.0)
                    ),
                    rerank_score=None,
                )
                for chunk in originals[:top_n]
            ]

        model = self._get_model()
        if model is None:
            return _passthrough()

        # Cap before predict() so a large candidate list cannot explode cost.
        candidates = originals[:_MAX_RERANK_CANDIDATES]
        pairs = [
            [query, getattr(chunk, "text", "")]
            for chunk in candidates
        ]
        try:
            raw_scores = model.predict(pairs)
        except Exception as error:
            logger.warning(
                "Cross-encoder scoring failed; returning retrieval order."
            )
            logger.debug("Cross-encoder predict failed.", exc_info=error)
            return _passthrough()

        scored: list[RerankResult] = []
        score_values = [float(value) for value in raw_scores]
        for chunk, raw_score in zip(candidates, score_values):
            scored.append(
                RerankResult(
                    chunk=chunk,
                    retrieval_score=float(
                        getattr(chunk, "score", 0.0)
                    ),
                    rerank_score=float(raw_score),
                )
            )
        scored.sort(
            key=lambda item: (
                -(item.rerank_score or 0.0),
                getattr(item.chunk, "chunk_id", ""),
            )
        )
        return scored[:top_n]


@lru_cache(maxsize=8)
def open_cross_encoder_reranker(
    model_name: str,
    cache_dir: str,
    device: str,
) -> CrossEncoderReranker:
    """Reuse one lazy reranker instance per complete configuration."""

    return CrossEncoderReranker(
        model_name,
        cache_dir=cache_dir,
        device=device,
    )


def reranker_from_settings(
    settings: Settings | None = None,
) -> CrossEncoderReranker:
    """Build the process-wide reranker from application settings."""

    config = settings or get_settings()
    return open_cross_encoder_reranker(
        config.rerank_model_name,
        str(config.model_dir),
        config.embedding_device,
    )
