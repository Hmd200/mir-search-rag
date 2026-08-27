"""CLI: re-embed SQLite chunks into the active Chroma collection."""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.retrieval.embeddings import (
    EmbeddingError,
    embedding_provider_from_settings,
)
from app.services.vector_reindex import VectorReindexError, reindex_vectors
from app.storage.database import SessionLocal, init_database
from app.storage.vector_store import VectorStoreError, open_vector_store


def main(argv: list[str] | None = None) -> int:
    """Re-embed stored chunks into the active collection; return an exit code."""

    parser = argparse.ArgumentParser(
        description=(
            "Embed stored SQLite chunk text with the active provider and write "
            "vectors into the active Chroma collection. Does not reparse files "
            "or modify SQLite/BM25. Only ever touches the active collection; "
            "with --overwrite that collection is dropped and rebuilt."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Drop and rebuild the active collection. Other collections stay. "
            "Under MIR_EMBEDDING_PROVIDER=local this rebuilds mir_chunks."
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_data_directories()
    init_database()
    embeddings = embedding_provider_from_settings(settings)
    vector_store = open_vector_store(
        str(settings.chroma_dir.resolve()),
        settings.active_vector_collection_name(),
    )
    session = SessionLocal()
    try:
        report = reindex_vectors(
            session,
            vector_store,
            embeddings,
            provider=settings.resolved_embedding_provider(),
            overwrite=args.overwrite,
        )
    except (VectorReindexError, EmbeddingError, VectorStoreError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        session.close()

    print(f"provider={report.provider}")
    print(f"collection={report.collection_name}")
    print(f"documents={report.document_count}")
    print(f"sqlite_chunks={report.sqlite_chunk_count}")
    print(f"chroma_before={report.chroma_chunk_count_before}")
    print(f"chroma_after={report.chroma_chunk_count_after}")
    print("verified=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
