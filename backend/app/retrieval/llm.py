"""Generative LLM clients used by the RAG pipeline."""

from __future__ import annotations

import re
from typing import Literal, Protocol

import httpx

from app.core.config import Settings

LlmProvider = Literal["ollama", "gemini"]

_THINK_BLOCK = re.compile(
    r"<think>.*?</think>",
    flags=re.IGNORECASE | re.DOTALL,
)


class LLMError(RuntimeError):
    """Raised when a generative language model cannot complete a request."""


class LLMClient(Protocol):
    """Minimal generation interface shared by provider implementations."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return a completion for the given system and user prompts."""


def strip_think_blocks(text: str) -> str:
    """Remove model-internal reasoning tags from a completion."""

    return _THINK_BLOCK.sub("", text).strip()


class OllamaClient:
    """Chat-completions client for a local Ollama server."""

    def __init__(
        self,
        settings: Settings,
        *,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload)
        except httpx.TimeoutException as error:
            raise LLMError("The language model timed out.") from error
        except httpx.RequestError as error:
            raise LLMError("The language model is unreachable.") from error

        if response.status_code != 200:
            raise LLMError("The language model is unreachable.")

        try:
            body = response.json()
        except ValueError as error:
            raise LLMError(
                "The language model returned an invalid response."
            ) from error

        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError("The language model returned an empty response.")

        return strip_think_blocks(content)


class GeminiClient:
    """Google Gemini generateContent client (API key, no extra SDK)."""

    def __init__(
        self,
        settings: Settings,
        *,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = settings.gemini_api_key.strip()
        self._model = settings.gemini_model.strip() or "gemini-2.0-flash"
        self._base_url = settings.gemini_api_base.rstrip("/")
        self._timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self._base_url}/models/{self._model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as error:
            raise LLMError("The language model timed out.") from error
        except httpx.RequestError as error:
            raise LLMError("The language model is unreachable.") from error

        if response.status_code in {401, 403}:
            raise LLMError("Gemini rejected the API key.")
        if response.status_code != 200:
            raise LLMError("The language model is unreachable.")

        try:
            body = response.json()
        except ValueError as error:
            raise LLMError(
                "The language model returned an invalid response."
            ) from error

        if not isinstance(body, dict):
            raise LLMError("The language model returned an invalid response.")

        text = _gemini_text(body)
        if not text.strip():
            raise LLMError("The language model returned an empty response.")
        return strip_think_blocks(text)


def _gemini_text(body: dict[str, object]) -> str:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMError("The language model returned an empty response.")
    first = candidates[0]
    if not isinstance(first, dict):
        raise LLMError("The language model returned an invalid response.")
    content = first.get("content")
    if not isinstance(content, dict):
        raise LLMError("The language model returned an empty response.")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise LLMError("The language model returned an empty response.")
    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts)


def resolve_llm_provider(
    settings: Settings,
    requested: str | None = None,
) -> LlmProvider:
    """Return ollama or gemini from the request, else from settings."""

    raw = (requested or settings.llm_provider or "ollama").strip().lower()
    if raw == "ollama":
        return "ollama"
    if raw == "gemini":
        return "gemini"
    raise LLMError(f"Unsupported LLM provider: {raw}.")


def create_llm_client(
    settings: Settings,
    *,
    provider: str | None = None,
    timeout: float = 120.0,
) -> LLMClient:
    """Return Ollama or Gemini from the request or MIR_LLM_PROVIDER."""

    resolved = resolve_llm_provider(settings, provider)
    if resolved == "ollama":
        return OllamaClient(settings, timeout=timeout)
    if not settings.gemini_api_key.strip():
        raise LLMError(
            "Gemini is not configured. Set MIR_GEMINI_API_KEY in the "
            "repository .env file."
        )
    return GeminiClient(settings, timeout=timeout)
