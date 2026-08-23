"""BM25 default / tunable / finetuned mode contract."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker
from test_documents_api import (
    DeterministicTestEmbedder,
    DocumentApiContext,
    make_pdf_bytes,
)

from app.api.dependencies import get_embedding_provider
from app.api.routes.search import _resolve_bm25_parameters
from app.core.config import Settings, get_settings
from app.main import create_app
from app.models import Base
from app.storage.database import create_database_engine, get_database_session


@pytest.fixture
def document_api(tmp_path: Path) -> Iterator[DocumentApiContext]:
    database_path = tmp_path / "api.db"
    database_engine = create_database_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(database_engine)
    session_factory = sessionmaker(
        bind=database_engine,
        autoflush=False,
        expire_on_commit=False,
    )
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        index_dir=tmp_path / "indexes",
        chroma_dir=tmp_path / "chroma",
        database_dir=tmp_path / "database",
        database_url=f"sqlite:///{database_path.as_posix()}",
        chunk_size=20,
        chunk_overlap=5,
        bm25_finetuned_k1=2.0,
        bm25_finetuned_b=0.3,
    )
    settings.ensure_data_directories()

    def override_database_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    application = create_app()
    application.dependency_overrides[get_database_session] = override_database_session
    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[get_embedding_provider] = lambda: (
        DeterministicTestEmbedder()
    )
    client = TestClient(application)

    yield DocumentApiContext(client, session_factory, settings)

    client.close()
    application.dependency_overrides.clear()
    database_engine.dispose()


def _index_signal_docs(client: TestClient) -> None:
    short = client.post(
        "/api/v1/documents",
        files={
            "file": ("short.pdf", make_pdf_bytes("signal"), "application/pdf"),
        },
    )
    long_text = (
        "signal alpha beta gamma delta epsilon zeta eta theta iota "
        "kappa lambda mu nu xi omicron pi rho sigma tau"
    )
    long = client.post(
        "/api/v1/documents",
        files={
            "file": ("long.pdf", make_pdf_bytes(long_text), "application/pdf"),
        },
    )
    assert short.status_code == 201
    assert long.status_code == 201


def _search_bm25(client: TestClient, **params: object):
    return client.get(
        "/api/v1/search/bm25",
        params={"q": "signal", "top_k": 5, **params},
    )


def _ranking(payload: dict) -> list[tuple[str, float]]:
    return [(item["chunk_id"], item["score"]) for item in payload["results"]]


def _scores(payload: dict) -> list[float]:
    return [item["score"] for item in payload["results"]]


def test_resolve_default_ignores_request_k1_b() -> None:
    settings = Settings(bm25_finetuned_k1=2.0, bm25_finetuned_b=0.3)
    mode, k1, b = _resolve_bm25_parameters("default", 0.9, 1.0, settings)
    assert (mode, k1, b) == ("default", 1.5, 0.75)


def test_resolve_tunable_uses_request_k1_b() -> None:
    settings = Settings(bm25_finetuned_k1=2.0, bm25_finetuned_b=0.3)
    mode, k1, b = _resolve_bm25_parameters("tunable", 0.9, 1.0, settings)
    assert (mode, k1, b) == ("tunable", 0.9, 1.0)


def test_resolve_finetuned_uses_settings_and_ignores_request() -> None:
    settings = Settings(bm25_finetuned_k1=2.0, bm25_finetuned_b=0.3)
    mode, k1, b = _resolve_bm25_parameters("finetuned", 0.9, 1.0, settings)
    assert (mode, k1, b) == ("finetuned", 2.0, 0.3)


def test_resolve_omitted_mode_preserves_request_params() -> None:
    settings = Settings()
    mode, k1, b = _resolve_bm25_parameters(None, 1.2, 0.6, settings)
    assert (mode, k1, b) == (None, 1.2, 0.6)


def test_default_mode_forces_standard_params_and_ignores_request(
    document_api: DocumentApiContext,
) -> None:
    _index_signal_docs(document_api.client)
    response = _search_bm25(
        document_api.client,
        bm25_mode="default",
        k1=0.9,
        b=1.0,
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["mode"] == "bm25"
    assert payload["bm25_mode"] == "default"
    assert payload["k1"] == 1.5
    assert payload["b"] == 0.75
    assert payload["results"]


def test_tunable_mode_uses_validated_request_k1_b(
    document_api: DocumentApiContext,
) -> None:
    _index_signal_docs(document_api.client)
    response = _search_bm25(
        document_api.client,
        bm25_mode="tunable",
        k1=0.9,
        b=1.0,
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["bm25_mode"] == "tunable"
    assert payload["k1"] == 0.9
    assert payload["b"] == 1.0


def test_finetuned_mode_uses_settings_and_ignores_request_k1_b(
    document_api: DocumentApiContext,
) -> None:
    _index_signal_docs(document_api.client)
    response = _search_bm25(
        document_api.client,
        bm25_mode="finetuned",
        k1=0.9,
        b=1.0,
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["bm25_mode"] == "finetuned"
    assert payload["k1"] == document_api.settings.bm25_finetuned_k1
    assert payload["b"] == document_api.settings.bm25_finetuned_b
    assert payload["k1"] == 2.0
    assert payload["b"] == 0.3


def test_omitted_mode_preserves_legacy_request_k1_b(
    document_api: DocumentApiContext,
) -> None:
    _index_signal_docs(document_api.client)
    with_params = _search_bm25(document_api.client, k1=1.2, b=0.6)
    without_params = _search_bm25(document_api.client)
    with_payload = with_params.json()
    without_payload = without_params.json()

    assert with_params.status_code == 200
    assert with_payload["bm25_mode"] is None
    assert with_payload["k1"] == 1.2
    assert with_payload["b"] == 0.6
    assert without_payload["bm25_mode"] is None
    assert without_payload["k1"] == 1.5
    assert without_payload["b"] == 0.75


def test_invalid_bm25_mode_is_rejected(document_api: DocumentApiContext) -> None:
    _index_signal_docs(document_api.client)
    response = _search_bm25(document_api.client, bm25_mode="banana")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("bm25_mode" in item.get("loc", []) for item in detail)


def test_irrelevant_params_do_not_affect_default_or_finetuned_ranking(
    document_api: DocumentApiContext,
) -> None:
    _index_signal_docs(document_api.client)
    default_clean = _search_bm25(document_api.client, bm25_mode="default")
    default_noisy = _search_bm25(
        document_api.client,
        bm25_mode="default",
        k1=0.9,
        b=1.0,
    )
    finetuned_clean = _search_bm25(document_api.client, bm25_mode="finetuned")
    finetuned_noisy = _search_bm25(
        document_api.client,
        bm25_mode="finetuned",
        k1=0.9,
        b=1.0,
    )
    tunable_as_finetuned = _search_bm25(
        document_api.client,
        bm25_mode="tunable",
        k1=2.0,
        b=0.3,
    )
    tunable_as_default = _search_bm25(
        document_api.client,
        bm25_mode="tunable",
        k1=1.5,
        b=0.75,
    )
    tunable_other = _search_bm25(
        document_api.client,
        bm25_mode="tunable",
        k1=0.9,
        b=1.0,
    )

    assert _ranking(default_clean.json()) == _ranking(default_noisy.json())
    assert _ranking(default_clean.json()) == _ranking(tunable_as_default.json())
    assert _ranking(finetuned_clean.json()) == _ranking(finetuned_noisy.json())
    assert _ranking(finetuned_clean.json()) == _ranking(tunable_as_finetuned.json())
    assert _scores(default_clean.json()) != _scores(tunable_other.json())
    assert _scores(finetuned_clean.json()) != _scores(tunable_other.json())
    assert _scores(default_noisy.json()) != _scores(tunable_other.json())
    assert _scores(finetuned_noisy.json()) != _scores(tunable_other.json())
    assert default_clean.json()["k1"] == 1.5
    assert default_noisy.json()["k1"] == 1.5
    assert finetuned_clean.json()["k1"] == 2.0
    assert finetuned_noisy.json()["k1"] == 2.0


def test_response_reports_selected_mode_and_effective_k1_b(
    document_api: DocumentApiContext,
) -> None:
    _index_signal_docs(document_api.client)
    payload = _search_bm25(
        document_api.client,
        bm25_mode="finetuned",
        k1=0.9,
        b=1.0,
    ).json()

    assert payload["mode"] == "bm25"
    assert payload["bm25_mode"] == "finetuned"
    assert payload["k1"] == 2.0
    assert payload["b"] == 0.3
