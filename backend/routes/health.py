from fastapi import APIRouter
import sqlite3

from backend.dependencies import get_chroma_collection
from backend.services.doc_registry import list_documents
from backend.models.responses import HealthResponse
from backend.core.config import settings

health_router = APIRouter()


@health_router.get("/", response_model=HealthResponse)
def health() -> HealthResponse:

    try:
        get_chroma_collection()
        vector_store_connected = True
    except RuntimeError:
        vector_store_connected = False

    try:
        list_documents(db_path=settings.registry_db_path)
        registry_connected = True
    except sqlite3.Error:
        registry_connected = False

    status = "healthy" if vector_store_connected and registry_connected else "unhealthy"

    return HealthResponse(
        status=status,
        vector_store_connected=vector_store_connected,
        registry_connected=registry_connected,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    )
