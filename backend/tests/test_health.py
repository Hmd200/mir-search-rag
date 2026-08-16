"""Tests for the initial FastAPI foundation."""

from fastapi.testclient import TestClient

from app.main import app


def test_service_info() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["health"] == "/api/v1/health"
    assert response.json()["docs"] == "/docs"


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "MIR Search & RAG API",
        "version": "0.1.0",
        "environment": "development",
    }


def test_frontend_origin_is_allowed() -> None:
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
