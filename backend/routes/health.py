from fastapi import APIRouter

from backend.dependencies import get_chroma_collection
from backend.services.doc_registry import list_documents
from backend.models.responses import HealthResponse
from backend.core.config import settings


health_router = APIRouter()


@health_router.get("/", response_model=HealthResponse)
def health() -> HealthResponse:


    vector_store_connected: bool = get_chroma_collection() is not None
    registry_connected: bool = list_documents() is not None
    if vector_store_connected or registry_connected:
        status = "ok"
    else:
        status = "error"   


    return HealthResponse(
        status=status,
        vector_store_connected=vector_store_connected,
        registry_connected=registry_connected,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    ) 
