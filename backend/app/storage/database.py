"""SQLAlchemy engine, sessions, and database initialization."""

import sqlite3
from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.models import Base


def _enable_sqlite_integrity(dbapi_connection: Any, _: Any) -> None:
    """Enable foreign-key cascades for each SQLite connection."""

    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_database_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create a configured SQLAlchemy engine for production or tests."""

    connect_args = (
        {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    )
    database_engine = create_engine(
        database_url,
        echo=echo,
        connect_args=connect_args,
    )
    if database_url.startswith("sqlite"):
        event.listen(database_engine, "connect", _enable_sqlite_integrity)
    return database_engine


settings = get_settings()
engine = create_database_engine(settings.database_url, echo=settings.database_echo)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_database() -> None:
    """Create missing application tables without deleting existing data."""

    Base.metadata.create_all(bind=engine)


def close_database() -> None:
    """Release pooled database connections during application shutdown."""

    engine.dispose()


def get_database_session() -> Generator[Session, None, None]:
    """Provide one database session to a request and always close it."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
