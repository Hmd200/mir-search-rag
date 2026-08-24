"""Embedding providers with an injectable test interface."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Protocol, runtime_checkable

import httpx
from sentence_transformers import SentenceTransformer

from app.core.config import Settings, resolve_embedding_provider

DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"


class EmbeddingError(RuntimeError):
    """Raised when an embedding model cannot load or encode text."""


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


class GeminiEmbeddingProvider:
    """AvalAI native Gemini embedContent client (HTTP only, no extra SDK)."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
        document_task_type: str = DOCUMENT_TASK_TYPE,
        query_task_type: str = QUERY_TASK_TYPE,
    ) -> None:
        key = api_key.strip()
        if not key:
            raise EmbeddingError(
                "Gemini embeddings are not configured. Set MIR_GEMINI_API_KEY "
                "in the repository .env file."
            )
        if dimensions <= 0:
            raise EmbeddingError("Gemini embedding dimensions must be positive.")
        model_name = model.strip().removeprefix("models/")
        if not model_name:
            raise EmbeddingError("Gemini embedding model is not configured.")
        self._api_key = key
        self._base_url = api_base.rstrip("/")
        self._model = model_name
        self._dimensions = dimensions
        self._timeout = timeout_seconds
        self._document_task_type = document_task_type
        self._query_task_type = query_task_type

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed empty document text.")
        return [
            self._embed_text(text, self._document_task_type) for text in texts
        ]

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self._embed_text(query, self._query_task_type)

    def _endpoint(self) -> str:
        return f"{self._base_url}/models/{self._model}:embedContent"

    def _payload(self, text: str, task_type: str) -> dict[str, object]:
        return {
            "contents": [{"parts": [{"text": text}]}],
            "embedding_config": {
                "task_type": task_type,
                "output_dimensionality": self._dimensions,
            },
        }

    def _embed_text(self, text: str, task_type: str) -> list[float]:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    self._endpoint(),
                    headers=headers,
                    json=self._payload(text, task_type),
                )
        except httpx.TimeoutException as error:
            raise EmbeddingError("The embedding provider timed out.") from error
        except httpx.RequestError as error:
            raise EmbeddingError(
                "The embedding provider is unreachable."
            ) from error

        if response.status_code in {401, 403}:
            raise EmbeddingError("Gemini rejected the embedding API key.")
        if response.status_code == 429:
            raise EmbeddingError("The embedding provider rate limit was exceeded.")
        if response.status_code != 200:
            raise EmbeddingError("The embedding provider returned an error.")

        try:
            body = response.json()
        except ValueError as error:
            raise EmbeddingError(
                "The embedding provider returned an invalid response."
            ) from error

        vectors = _vectors_from_response(body, expected_count=1)
        return _normalized_vector(vectors[0], self._dimensions)


def _vectors_from_response(body: object, *, expected_count: int) -> list[object]:
    if not isinstance(body, dict):
        raise EmbeddingError("The embedding provider returned an invalid response.")

    raw_list = body.get("embeddings")
    if raw_list is None and isinstance(body.get("embedding"), dict):
        raw_list = [body["embedding"]]
    if not isinstance(raw_list, list):
        raise EmbeddingError("The embedding provider returned an invalid response.")
    if len(raw_list) != expected_count:
        raise EmbeddingError(
            "The embedding provider returned the wrong vector count."
        )
    return raw_list


def _normalized_vector(item: object, dimensions: int) -> list[float]:
    if not isinstance(item, dict):
        raise EmbeddingError("The embedding provider returned an invalid response.")
    values = item.get("values")
    if not isinstance(values, list) or not values:
        raise EmbeddingError("The embedding provider returned an invalid response.")
    if len(values) != dimensions:
        raise EmbeddingError(
            "The embedding provider returned the wrong dimensions."
        )

    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingError(
                "The embedding provider returned an invalid response."
            )
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingError(
                "The embedding provider returned a non-finite vector."
            )
        vector.append(number)

    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise EmbeddingError("The embedding provider returned a zero vector.")
    return [component / norm for component in vector]


@lru_cache(maxsize=8)
def open_embedding_provider(
    provider: str,
    model_name: str,
    cache_dir: str,
    device: str,
    batch_size: int,
    gemini_api_key: str,
    gemini_api_base: str,
    gemini_embedding_model: str,
    gemini_embedding_dimensions: int,
    gemini_embedding_timeout_seconds: float,
) -> EmbeddingProvider:
    """Reuse one provider instance for each complete configuration."""

    try:
        resolved = resolve_embedding_provider(provider)
    except ValueError as error:
        raise EmbeddingError(str(error)) from error

    if resolved == "local":
        return SentenceTransformerEmbeddingProvider(
            model_name,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
        )

    return GeminiEmbeddingProvider(
        api_key=gemini_api_key,
        api_base=gemini_api_base,
        model=gemini_embedding_model,
        dimensions=gemini_embedding_dimensions,
        timeout_seconds=gemini_embedding_timeout_seconds,
    )


def embedding_provider_from_settings(settings: Settings) -> EmbeddingProvider:
    """Construct the configured embedder, failing closed for Gemini."""

    return open_embedding_provider(
        settings.embedding_provider,
        settings.embedding_model,
        str(settings.model_dir.resolve()),
        settings.embedding_device,
        settings.embedding_batch_size,
        settings.gemini_api_key,
        settings.gemini_api_base,
        settings.gemini_embedding_model,
        settings.gemini_embedding_dimensions,
        settings.gemini_embedding_timeout_seconds,
    )
