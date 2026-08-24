"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.retrieval.embeddings import embedding_provider_from_settings
from app.storage.database import close_database, init_database


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare local storage before accepting requests."""

    app.state.settings.ensure_data_directories()
    embedding_provider_from_settings(app.state.settings)
    init_database()
    try:
        yield
    finally:
        close_database()


def create_app() -> FastAPI:
    """Build and configure the backend application."""

    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Dual-engine document retrieval and grounded RAG API.",
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix=settings.api_prefix)

    @application.get("/", tags=["System"])
    async def service_info() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "health": f"{settings.api_prefix}/health",
            "docs": "/docs",
        }

    return application


app = create_app()
