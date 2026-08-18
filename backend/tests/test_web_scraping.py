"""Tests for URL ingest with mocked network I/O."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator

import httpx
import pytest

from tests.test_documents_api import DocumentApiContext

pytest_plugins = ["tests.test_documents_api"]

_PAGE_URL = "https://example.com/neural-ranking"
_PAGE_HTML = """
<html>
  <head><title>Neural Ranking Primer</title></head>
  <body>
    <article>
      <h1>Neural Ranking Primer</h1>
      <p>
        Cosine similarity ranks vector retrieval results in modern
        information retrieval systems that combine lexical and dense search.
      </p>
      <p>
        Lexical search uses inverted indexes while dense retrievers embed
        queries and documents into a shared vector space for ranking.
      </p>
      <p>
        Pseudo relevance feedback expands a query from the top retrieved
        documents so vocabulary mismatch is less likely to hide relevant text.
      </p>
    </article>
  </body>
</html>
"""


class _FakeStreamResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        url: str = _PAGE_URL,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._content = content
        self.status_code = status_code
        self.url = url
        self.headers = headers or {}

    @property
    def is_redirect(self) -> bool:
        return 300 <= self.status_code < 400

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        del chunk_size
        yield self._content

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeClient:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs: object) -> _FakeStreamResponse:
        del method, url, kwargs
        return self._response


class _TimeoutClient:
    def __enter__(self) -> _TimeoutClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs: object) -> _FakeStreamResponse:
        del method, url, kwargs
        raise httpx.TimeoutException("timed out")


class _OversizedStreamResponse(_FakeStreamResponse):
    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        del chunk_size
        yield b"x" * (5 * 1024 * 1024 + 1)


def _public_addrinfo(
    host: str,
    port: int | None,
    *args: object,
    **kwargs: object,
) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    del args, kwargs
    resolved_port = port or 80
    try:
        ip = str(ipaddress.ip_address(host))
    except ValueError:
        if host == "localhost":
            ip = "127.0.0.1"
        else:
            ip = "93.184.216.34"
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, resolved_port)),
    ]


def _patch_public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.processing.extractors.socket.getaddrinfo",
        _public_addrinfo,
    )


def _patch_html_fetch(monkeypatch: pytest.MonkeyPatch, html: str = _PAGE_HTML) -> None:
    response = _FakeStreamResponse(html.encode("utf-8"))
    monkeypatch.setattr(
        "app.processing.extractors.httpx.Client",
        lambda **kwargs: _FakeClient(response),
    )


def test_successful_scrape_indexes_like_an_upload(
    document_api: DocumentApiContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_public_dns(monkeypatch)
    _patch_html_fetch(monkeypatch)

    response = document_api.client.post(
        "/api/v1/documents/from-url",
        json={"url": _PAGE_URL},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "indexed"
    assert payload["keyword_indexed"] is True
    assert payload["vector_indexed"] is True
    assert payload["chunk_count"] >= 1
    assert payload["source_type"] == "web"
    assert payload["source_url"] == _PAGE_URL

    search = document_api.client.get(
        "/api/v1/search/keyword",
        params={"q": "cosine retrieval", "top_k": 5},
    )
    assert search.status_code == 200
    assert search.json()["results"][0]["document_id"] == payload["id"]


def test_source_type_and_url_are_stored(
    document_api: DocumentApiContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_public_dns(monkeypatch)
    _patch_html_fetch(monkeypatch)

    created = document_api.client.post(
        "/api/v1/documents/from-url",
        json={"url": _PAGE_URL},
    ).json()
    listed = document_api.client.get("/api/v1/documents")
    fetched = document_api.client.get(f"/api/v1/documents/{created['id']}")

    assert listed.status_code == 200
    assert listed.json()["items"][0]["source_type"] == "web"
    assert listed.json()["items"][0]["source_url"] == _PAGE_URL
    assert fetched.json()["source_type"] == "web"
    assert fetched.json()["source_url"] == _PAGE_URL


def test_deleting_url_document_removes_both_indexes(
    document_api: DocumentApiContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_public_dns(monkeypatch)
    _patch_html_fetch(monkeypatch)

    created = document_api.client.post(
        "/api/v1/documents/from-url",
        json={"url": _PAGE_URL},
    )
    document_id = created.json()["id"]

    assert (
        document_api.client.delete(f"/api/v1/documents/{document_id}").status_code
        == 204
    )

    keyword = document_api.client.get(
        "/api/v1/search/keyword",
        params={"q": "cosine retrieval"},
    )
    semantic = document_api.client.get(
        "/api/v1/search/semantic",
        params={"q": "cosine retrieval"},
    )
    stats = document_api.client.get("/api/v1/search/keyword/stats")
    vector_stats = document_api.client.get("/api/v1/search/semantic/stats")

    assert keyword.json()["results"] == []
    assert semantic.json()["results"] == []
    assert stats.json()["document_count"] == 0
    assert vector_stats.json()["chunk_count"] == 0


def test_non_http_url_is_rejected(
    document_api: DocumentApiContext,
) -> None:
    response = document_api.client.post(
        "/api/v1/documents/from-url",
        json={"url": "file:///etc/passwd"},
    )

    assert response.status_code == 400
    assert "http" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/secret",
        "http://localhost/secret",
        "http://10.0.0.8/internal",
        "http://192.168.1.10/router",
        "http://172.16.0.4/service",
        "http://169.254.12.1/link-local",
    ],
)
def test_private_network_url_is_rejected(
    document_api: DocumentApiContext,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    _patch_public_dns(monkeypatch)
    response = document_api.client.post(
        "/api/v1/documents/from-url",
        json={"url": url},
    )

    assert response.status_code == 400
    assert "internal" in response.json()["detail"].lower()


def test_unreachable_url_times_out_with_400(
    document_api: DocumentApiContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_public_dns(monkeypatch)
    monkeypatch.setattr(
        "app.processing.extractors.httpx.Client",
        lambda **kwargs: _TimeoutClient(),
    )

    response = document_api.client.post(
        "/api/v1/documents/from-url",
        json={"url": "https://example.com/slow"},
    )

    assert response.status_code == 400
    assert "timed out" in response.json()["detail"].lower()


def test_oversized_content_is_rejected(
    document_api: DocumentApiContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_public_dns(monkeypatch)
    monkeypatch.setattr(
        "app.processing.extractors.httpx.Client",
        lambda **kwargs: _FakeClient(_OversizedStreamResponse(b"")),
    )

    response = document_api.client.post(
        "/api/v1/documents/from-url",
        json={"url": _PAGE_URL},
    )

    assert response.status_code == 400
    assert "5 mb" in response.json()["detail"].lower()
