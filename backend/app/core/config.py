"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Central configuration for the search and RAG backend."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="MIR_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "MIR Search & RAG API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"

    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    )

    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    index_dir: Path = PROJECT_ROOT / "data" / "indexes"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    database_dir: Path = PROJECT_ROOT / "data" / "database"
    model_dir: Path = PROJECT_ROOT / "data" / "models"
    database_url: str = (
        f"sqlite:///{(PROJECT_ROOT / 'data' / 'database' / 'mir.db').as_posix()}"
    )
    database_echo: bool = False

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    vector_collection_name: str = "mir_chunks"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    llm_provider: str = "ollama"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    prf_feedback_docs: int = 5
    prf_max_expansion_terms: int = 10
    prf_alpha: float = 1.0
    prf_beta: float = 0.75
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_enabled_default: bool = False
    rag_min_retrieval_score: float = Field(default=0.30, ge=0.0, le=1.0)
    rag_max_sentences_per_citation_group: int = Field(default=2, ge=1, le=8)
    rag_lexical_coverage_min: float = Field(default=0.60, ge=0.0, le=1.0)
    rag_lexical_idf_coverage_min: float = Field(default=0.40, ge=0.0, le=1.0)
    bm25_finetuned_k1: float = Field(default=1.5, gt=0.0, le=10.0)
    bm25_finetuned_b: float = Field(default=0.75, ge=0.0, le=1.0)

    max_upload_size_mb: int = 25
    chunk_size: int = 500
    chunk_overlap: int = 75

    def ensure_data_directories(self) -> None:
        """Create local runtime directories when the API starts."""

        for directory in (
            self.data_dir,
            self.upload_dir,
            self.index_dir,
            self.chroma_dir,
            self.database_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings object per process."""

    return Settings()
