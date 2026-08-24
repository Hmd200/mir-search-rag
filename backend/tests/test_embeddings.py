"""Mocked tests for local vs Gemini embedding provider selection."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.core.config import Settings, gemini_vector_collection_name
from app.retrieval.embeddings import (
    EmbeddingError,
    GeminiEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    embedding_provider_from_settings,
    open_embedding_provider,
)

FAKE_KEY = "TEST_GEMINI_EMBED_KEY_DO_NOT_LEAK"
AVALAI_BASE = "https://api.avalai.ir/v1beta"
EMBED_URL = f"{AVALAI_BASE}/models/gemini-embedding-001:embedContent"


@pytest.fixture(autouse=True)
def _clear_provider_cache() -> Iterator[None]:
    open_embedding_provider.cache_clear()
    yield
    open_embedding_provider.cache_clear()


def _provider(**overrides: object) -> GeminiEmbeddingProvider:
    values: dict[str, object] = {
        "api_key": FAKE_KEY,
        "api_base": AVALAI_BASE,
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "timeout_seconds": 30.0,
    }
    values.update(overrides)
    return GeminiEmbeddingProvider(**values)  # type: ignore[arg-type]


def _values(dimensions: int = 768, scale: float = 1.0) -> list[float]:
    vector = [0.0] * dimensions
    vector[0] = 3.0 * scale
    vector[1] = 4.0 * scale
    return vector


def _body(values: list[float] | list[list[float]]) -> dict[str, object]:
    if values and isinstance(values[0], list):
        return {"embeddings": [{"values": item} for item in values]}  # type: ignore[misc]
    return {"embeddings": [{"values": values}]}


def _response(payload: object, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = f"raw-provider-body {FAKE_KEY}"
    if isinstance(payload, Exception):
        response.json.side_effect = payload
    else:
        response.json.return_value = payload
    return response


def _patch_post(response: MagicMock | None = None, side_effect: object = None):
    client_cls = patch("app.retrieval.embeddings.httpx.Client")
    mocked = client_cls.start()
    post = mocked.return_value.__enter__.return_value.post
    if side_effect is not None:
        post.side_effect = side_effect
    else:
        post.return_value = response
    return client_cls, post


def _assert_no_secret(exc: BaseException) -> None:
    message = str(exc)
    assert FAKE_KEY not in message
    cause = exc.__cause__
    if cause is not None:
        assert FAKE_KEY not in str(cause)


def test_local_empty_inputs_do_not_load_model(tmp_path: Path) -> None:
    provider = SentenceTransformerEmbeddingProvider("unused-model", cache_dir=tmp_path)
    assert provider.embed_documents([]) == []
    with pytest.raises(EmbeddingError, match="empty query"):
        provider.embed_query("  ")


def test_provider_selection_local(tmp_path: Path) -> None:
    provider = open_embedding_provider(
        "local",
        "sentence-transformers/all-MiniLM-L6-v2",
        str(tmp_path),
        "cpu",
        32,
        FAKE_KEY,
        AVALAI_BASE,
        "gemini-embedding-001",
        768,
        30.0,
    )
    assert isinstance(provider, SentenceTransformerEmbeddingProvider)


def test_provider_selection_gemini(tmp_path: Path) -> None:
    provider = open_embedding_provider(
        "gemini",
        "sentence-transformers/all-MiniLM-L6-v2",
        str(tmp_path),
        "cpu",
        32,
        FAKE_KEY,
        AVALAI_BASE,
        "gemini-embedding-001",
        768,
        30.0,
    )
    assert isinstance(provider, GeminiEmbeddingProvider)


def test_unknown_provider_rejected(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="Unsupported embedding provider") as exc:
        open_embedding_provider(
            "openai",
            "sentence-transformers/all-MiniLM-L6-v2",
            str(tmp_path),
            "cpu",
            32,
            FAKE_KEY,
            AVALAI_BASE,
            "gemini-embedding-001",
            768,
            30.0,
        )
    _assert_no_secret(exc.value)


def test_gemini_selected_with_missing_key(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="MIR_GEMINI_API_KEY") as exc:
        embedding_provider_from_settings(
            Settings(
                embedding_provider="gemini",
                gemini_api_key="",
                model_dir=tmp_path,
            )
        )
    _assert_no_secret(exc.value)


def test_endpoint_and_api_key_header() -> None:
    provider = _provider()
    patcher, post = _patch_post(_response(_body(_values())))
    try:
        vector = provider.embed_query("What is BM25?")
    finally:
        patcher.stop()

    assert len(vector) == 768
    post.assert_called_once()
    url = post.call_args.args[0]
    headers = post.call_args.kwargs["headers"]
    payload = post.call_args.kwargs["json"]
    assert url == EMBED_URL
    assert "/v1/chat/completions" not in url
    assert "/v1/embeddings" not in url
    assert "generativelanguage.googleapis.com" not in url
    assert headers["x-goog-api-key"] == FAKE_KEY
    assert "Authorization" not in headers
    assert payload["embedding_config"]["output_dimensionality"] == 768
    assert payload["embedding_config"]["task_type"] == "RETRIEVAL_QUERY"


def test_document_and_query_task_types_and_normalization() -> None:
    provider = _provider()
    document_response = _response(_body(_values(scale=1.0)))
    query_response = _response(_body(_values(scale=2.0)))
    patcher, post = _patch_post(side_effect=[document_response, query_response])
    try:
        documents = provider.embed_documents(["chunk text"])
        query = provider.embed_query("query text")
    finally:
        patcher.stop()

    assert post.call_count == 2
    document_payload = post.call_args_list[0].kwargs["json"]
    query_payload = post.call_args_list[1].kwargs["json"]
    assert document_payload["embedding_config"]["task_type"] == "RETRIEVAL_DOCUMENT"
    assert query_payload["embedding_config"]["task_type"] == "RETRIEVAL_QUERY"
    assert document_payload["contents"][0]["parts"][0]["text"] == "chunk text"
    assert query_payload["contents"][0]["parts"][0]["text"] == "query text"
    assert documents[0][0] == pytest.approx(0.6)
    assert documents[0][1] == pytest.approx(0.8)
    assert query[0] == pytest.approx(0.6)
    assert query[1] == pytest.approx(0.8)
    assert sum(value * value for value in documents[0]) == pytest.approx(1.0)
    assert sum(value * value for value in query) == pytest.approx(1.0)


def test_empty_document_list_does_not_call_http() -> None:
    provider = _provider()
    patcher, post = _patch_post(_response(_body(_values())))
    try:
        assert provider.embed_documents([]) == []
    finally:
        patcher.stop()
    post.assert_not_called()


def test_empty_query_raises_without_http() -> None:
    provider = _provider()
    patcher, post = _patch_post(_response(_body(_values())))
    try:
        with pytest.raises(EmbeddingError, match="empty query"):
            provider.embed_query("   ")
    finally:
        patcher.stop()
    post.assert_not_called()


def test_malformed_json_response() -> None:
    provider = _provider()
    patcher, _post = _patch_post(_response(ValueError("not json")))
    try:
        with pytest.raises(EmbeddingError, match="invalid response") as exc:
            provider.embed_query("query")
    finally:
        patcher.stop()
    _assert_no_secret(exc.value)


def test_malformed_body_shape() -> None:
    provider = _provider()
    patcher, _post = _patch_post(_response({"data": [{"embedding": [1.0]}]}))
    try:
        with pytest.raises(EmbeddingError, match="invalid response") as exc:
            provider.embed_query("query")
    finally:
        patcher.stop()
    _assert_no_secret(exc.value)
    assert "raw-provider-body" not in str(exc.value)


def test_wrong_vector_count() -> None:
    provider = _provider()
    payload = {"embeddings": [{"values": _values()}, {"values": _values()}]}
    patcher, _post = _patch_post(_response(payload))
    try:
        with pytest.raises(EmbeddingError, match="wrong vector count") as exc:
            provider.embed_query("query")
    finally:
        patcher.stop()
    _assert_no_secret(exc.value)


def test_wrong_dimensions() -> None:
    provider = _provider()
    patcher, _post = _patch_post(_response(_body([0.1, 0.2, 0.3])))
    try:
        with pytest.raises(EmbeddingError, match="wrong dimensions") as exc:
            provider.embed_query("query")
    finally:
        patcher.stop()
    _assert_no_secret(exc.value)


def test_non_finite_values() -> None:
    nan_vector = _values()
    nan_vector[0] = float("nan")
    inf_vector = _values()
    inf_vector[1] = float("inf")
    provider = _provider()
    patcher, _post = _patch_post(
        side_effect=[_response(_body(nan_vector)), _response(_body(inf_vector))]
    )
    try:
        with pytest.raises(EmbeddingError, match="non-finite") as nan_exc:
            provider.embed_query("query")
        with pytest.raises(EmbeddingError, match="non-finite") as inf_exc:
            provider.embed_documents(["chunk"])
    finally:
        patcher.stop()
    _assert_no_secret(nan_exc.value)
    _assert_no_secret(inf_exc.value)


def test_zero_vector() -> None:
    provider = _provider()
    patcher, _post = _patch_post(_response(_body([0.0] * 768)))
    try:
        with pytest.raises(EmbeddingError, match="zero vector") as exc:
            provider.embed_query("query")
    finally:
        patcher.stop()
    _assert_no_secret(exc.value)


@pytest.mark.parametrize("status_code", [401, 403])
def test_http_auth_failures_hide_secret(status_code: int) -> None:
    provider = _provider()
    patcher, _post = _patch_post(_response({"error": FAKE_KEY}, status_code=status_code))
    try:
        with pytest.raises(EmbeddingError, match="rejected the embedding API key") as exc:
            provider.embed_query("query")
    finally:
        patcher.stop()
    _assert_no_secret(exc.value)
    assert "raw-provider-body" not in str(exc.value)


def test_http_rate_limit() -> None:
    provider = _provider()
    patcher, _post = _patch_post(_response({"error": "slow down"}, status_code=429))
    try:
        with pytest.raises(EmbeddingError, match="rate limit") as exc:
            provider.embed_documents(["chunk"])
    finally:
        patcher.stop()
    _assert_no_secret(exc.value)


def test_timeout_and_network_errors() -> None:
    provider = _provider()
    patcher, _post = _patch_post(side_effect=httpx.TimeoutException("timed out"))
    try:
        with pytest.raises(EmbeddingError, match="timed out") as timeout_exc:
            provider.embed_query("query")
    finally:
        patcher.stop()
    _assert_no_secret(timeout_exc.value)

    patcher, _post = _patch_post(side_effect=httpx.ConnectError("refused"))
    try:
        with pytest.raises(EmbeddingError, match="unreachable") as network_exc:
            provider.embed_documents(["chunk"])
    finally:
        patcher.stop()
    _assert_no_secret(network_exc.value)


def test_sequential_document_requests_stay_on_native_endpoint() -> None:
    provider = _provider()
    patcher, post = _patch_post(
        side_effect=[
            _response(_body(_values())),
            _response(_body(_values())),
        ]
    )
    try:
        vectors = provider.embed_documents(["first chunk", "second chunk"])
    finally:
        patcher.stop()

    assert len(vectors) == 2
    assert post.call_count == 2
    assert all(call.args[0] == EMBED_URL for call in post.call_args_list)
    assert post.call_args_list[0].kwargs["json"]["contents"][0]["parts"][0]["text"] == (
        "first chunk"
    )
    assert post.call_args_list[1].kwargs["json"]["contents"][0]["parts"][0]["text"] == (
        "second chunk"
    )


def test_cache_key_includes_every_relevant_configuration(tmp_path: Path) -> None:
    local_args = (
        "local",
        "sentence-transformers/all-MiniLM-L6-v2",
        str(tmp_path / "a"),
        "cpu",
        32,
        "",
        AVALAI_BASE,
        "gemini-embedding-001",
        768,
        30.0,
    )
    first = open_embedding_provider(*local_args)
    second = open_embedding_provider(*local_args)
    assert first is second

    changed = [
        ("local", "other-model", str(tmp_path / "a"), "cpu", 32, "", AVALAI_BASE, "gemini-embedding-001", 768, 30.0),
        ("local", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "b"), "cpu", 32, "", AVALAI_BASE, "gemini-embedding-001", 768, 30.0),
        ("local", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "a"), "cuda", 32, "", AVALAI_BASE, "gemini-embedding-001", 768, 30.0),
        ("local", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "a"), "cpu", 16, "", AVALAI_BASE, "gemini-embedding-001", 768, 30.0),
        ("local", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "a"), "cpu", 32, "other-key", AVALAI_BASE, "gemini-embedding-001", 768, 30.0),
        ("local", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "a"), "cpu", 32, "", "https://example.invalid/v1beta", "gemini-embedding-001", 768, 30.0),
        ("local", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "a"), "cpu", 32, "", AVALAI_BASE, "gemini-embedding-2", 768, 30.0),
        ("local", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "a"), "cpu", 32, "", AVALAI_BASE, "gemini-embedding-001", 1536, 30.0),
        ("local", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "a"), "cpu", 32, "", AVALAI_BASE, "gemini-embedding-001", 768, 15.0),
        ("gemini", "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "a"), "cpu", 32, FAKE_KEY, AVALAI_BASE, "gemini-embedding-001", 768, 30.0),
    ]
    for args in changed:
        other = open_embedding_provider(*args)
        assert other is not first


def test_gemini_collection_is_distinct_from_local() -> None:
    local = Settings(embedding_provider="local", vector_collection_name="mir_chunks")
    gemini = Settings(
        embedding_provider="gemini",
        gemini_api_key=FAKE_KEY,
        gemini_embedding_model="gemini-embedding-001",
        gemini_embedding_dimensions=768,
    )
    assert local.active_vector_collection_name() == "mir_chunks"
    assert gemini.active_vector_collection_name() == "mir_chunks_gemini_001_768"
    assert gemini.active_vector_collection_name() != local.active_vector_collection_name()
    assert gemini_vector_collection_name("gemini-embedding-001", 768) == (
        "mir_chunks_gemini_001_768"
    )


def test_key_absent_from_caplog(caplog: pytest.LogCaptureFixture) -> None:
    provider = _provider()
    patcher, _post = _patch_post(_response({"error": FAKE_KEY}, status_code=401))
    try:
        with caplog.at_level("DEBUG"), pytest.raises(EmbeddingError):
            provider.embed_query("query")
    finally:
        patcher.stop()
    assert FAKE_KEY not in caplog.text
