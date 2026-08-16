"""Lazy sentence-transformer embeddings with an injectable test interface."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Protocol, runtime_checkable

from sentence_transformers import SentenceTransformer


class EmbeddingError(RuntimeError):
    """Raised when the local embedding model cannot load or encode text."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal embedding interface used by Chroma and search services."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query."""


class SentenceTransformerEmbeddingProvider:
    """Load a sentence-transformer once and return normalized CPU vectors."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | Path,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._lock = RLock()

    def _get_model(self) -> SentenceTransformer:
        with self._lock:
            if self._model is None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        cache_folder=str(self.cache_dir),
                        device=self.device,
                    )
                except Exception as error:
                    raise EmbeddingError(
                        f"Could not load embedding model '{self.model_name}'."
                    ) from error
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._get_model().encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise EmbeddingError("Could not embed document chunks.") from error
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self.embed_documents([query])[0]


@lru_cache(maxsize=8)
def open_embedding_provider(
    model_name: str,
    cache_dir: str,
    device: str,
    batch_size: int,
) -> SentenceTransformerEmbeddingProvider:
    """Reuse one lazy model instance for each complete configuration."""

    return SentenceTransformerEmbeddingProvider(
        model_name,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
    )
