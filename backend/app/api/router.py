"""Top-level API router."""

from fastapi import APIRouter

from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router
from app.api.routes.semantic_search import router as semantic_search_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["System"])
api_router.include_router(documents_router, tags=["Documents"])
api_router.include_router(search_router, tags=["Search"])
api_router.include_router(semantic_search_router, tags=["Search"])
