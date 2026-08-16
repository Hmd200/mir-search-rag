"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

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
