from fastapi import APIRouter, Request

from backend.dependencies import get_chroma_collection
from backend.services.doc_registry import list_documents
from backend.models.responses import HealthResponse
from backend.core.config import settings
from backend.limiter import limiter

health_router = APIRouter()


@health_router.get("/", response_model=HealthResponse)
@limiter.limit("10/minute")
async def health(request: Request) -> HealthResponse:

    try:
        collection = get_chroma_collection()
        collection.count()
        vector_store_connected = True
    except Exception:
        vector_store_connected = False

    try:
        await list_documents(db_path=settings.registry_db_path)
        registry_connected = True
    except Exception:
        registry_connected = False

    status = "healthy" if vector_store_connected and registry_connected else "unhealthy"

    return HealthResponse(
        status=status,
        vector_store_connected=vector_store_connected,
        registry_connected=registry_connected,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    )
