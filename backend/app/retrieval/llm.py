"""Generative LLM clients used by the RAG pipeline."""

from __future__ import annotations

import re
from typing import Protocol

import httpx

from app.core.config import Settings

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


def create_llm_client(
    settings: Settings,
    *,
    timeout: float = 120.0,
) -> LLMClient:
    """Return the configured LLM implementation."""

    provider = settings.llm_provider.strip().lower()
    if provider == "ollama":
        return OllamaClient(settings, timeout=timeout)

    raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}.")
