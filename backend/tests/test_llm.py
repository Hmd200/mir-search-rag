"""Tests for Ollama/Gemini client selection."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.rag import get_rag_service, router
from app.core.config import Settings, get_settings
from app.retrieval.llm import (
    GeminiClient,
    LLMError,
    OllamaClient,
    create_llm_client,
    resolve_llm_provider,
)
from app.services.rag import RagService
from tests.test_rag import BoomLLM, FakeSearch, _record


def test_resolve_provider_defaults_to_settings() -> None:
    settings = Settings(llm_provider="ollama")
    assert resolve_llm_provider(settings, None) == "ollama"
    assert resolve_llm_provider(settings, "gemini") == "gemini"


def test_create_gemini_client_without_key_raises() -> None:
    settings = Settings(llm_provider="gemini", gemini_api_key="")
    with pytest.raises(LLMError, match="MIR_GEMINI_API_KEY"):
        create_llm_client(settings, provider="gemini")


def test_gemini_client_reads_candidate_text() -> None:
    settings = Settings(gemini_api_key="test-key")
    client = GeminiClient(settings)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Grounded [1]."}]}}],
    }

    with patch("app.retrieval.llm.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = response
        text = client.generate("system", "user")

    assert text == "Grounded [1]."


def test_ollama_payload_uses_temperature_zero() -> None:
    settings = Settings(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="qwen3:8b",
    )
    client = OllamaClient(settings)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "message": {"content": "Grounded answer [1]."},
    }

    with patch("app.retrieval.llm.httpx.Client") as client_cls:
        post = client_cls.return_value.__enter__.return_value.post
        post.return_value = response
        text = client.generate("system prompt text", "user prompt text")

    assert text == "Grounded answer [1]."
    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == "http://127.0.0.1:11434/api/chat"
    payload = kwargs["json"]
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["messages"] == [
        {"role": "system", "content": "system prompt text"},
        {"role": "user", "content": "user prompt text"},
    ]


def test_gemini_without_key_returns_400() -> None:
    application = FastAPI()
    application.include_router(router, prefix="/api/v1")

    def override_rag_service() -> RagService:
        records = [_record(index) for index in range(1, 3)]
        return RagService(FakeSearch(records), BoomLLM())

    application.dependency_overrides[get_rag_service] = override_rag_service
    application.dependency_overrides[get_settings] = lambda: Settings(
        gemini_api_key="",
    )
    client = TestClient(application)

    response = client.post(
        "/api/v1/search/rag",
        json={"query": "What is BM25?", "llm_provider": "gemini"},
    )

    assert response.status_code == 400
    assert "MIR_GEMINI_API_KEY" in response.json()["detail"]
