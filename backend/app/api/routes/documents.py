"""Document administration endpoints."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_embedding_provider,
    get_keyword_index,
    get_vector_store,
)
from app.api.schemas.documents import (
    DocumentFromUrlRequest,
    DocumentListResponse,
    DocumentResponse,
)
from app.core.config import Settings, get_settings
from app.models import DocumentStatus, SourceType
from app.processing import DocumentProcessingError
from app.processing.extractors import ExtractionError
from app.retrieval.embeddings import EmbeddingProvider
from app.services.documents import (
    DocumentNotFoundError,
    DocumentService,
    DocumentServiceError,
    DuplicateDocumentError,
    EmptyUploadError,
    UnsupportedUploadError,
    UploadTooLargeError,
)
from app.storage.database import get_database_session
from app.storage.keyword_index import KeywordIndex
from app.storage.vector_store import ChromaVectorStore

router = APIRouter(prefix="/documents")


def _service(
    session: Annotated[Session, Depends(get_database_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
    vector_store: Annotated[ChromaVectorStore, Depends(get_vector_store)],
    embeddings: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> DocumentService:
    """Build a DocumentService from the request-scoped dependencies."""

    return DocumentService(
        session,
        settings,
        keyword_index,
        vector_store,
        embeddings,
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: Annotated[
        UploadFile,
        File(description="A PDF or DOCX document to add to the collection."),
    ],
    service: Annotated[DocumentService, Depends(_service)],
) -> DocumentResponse:
    """Store, extract, chunk, and register a document for indexing."""

    try:
        document = service.ingest_upload(file)
    except UnsupportedUploadError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error)
        ) from error
    except UploadTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(error)
        ) from error
    except EmptyUploadError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except DuplicateDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(error),
                "existing_document_id": error.existing_document_id,
            },
        ) from error
    except DocumentProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except DocumentServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be stored.",
        ) from error

    return DocumentResponse.model_validate(document)


@router.post(
    "/from-url",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_document_from_url(
    payload: DocumentFromUrlRequest,
    service: Annotated[DocumentService, Depends(_service)],
) -> DocumentResponse:
    """Scrape a public web page and index it like an uploaded file."""

    try:
        document = service.ingest_from_url(payload.url)
    except ExtractionError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except DuplicateDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(error),
                "existing_document_id": error.existing_document_id,
            },
        ) from error
    except DocumentProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except DocumentServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be stored.",
        ) from error

    return DocumentResponse.model_validate(document)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    service: Annotated[DocumentService, Depends(_service)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    source_type: SourceType | None = None,
    document_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
) -> DocumentListResponse:
    """List documents for the administration dashboard."""

    documents, total = service.list_documents(
        offset=offset,
        limit=limit,
        source_type=source_type,
        status=document_status,
    )
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(document) for document in documents],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    service: Annotated[DocumentService, Depends(_service)],
) -> DocumentResponse:
    """Return metadata for one document."""

    try:
        document = service.get_document(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    service: Annotated[DocumentService, Depends(_service)],
) -> Response:
    """Delete a document, all of its chunks, and its stored source file."""

    try:
        service.delete_document(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    except DocumentServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be deleted.",
        ) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)
