from fastapi import APIRouter, status, Request, HTTPException

from backend.services.doc_registry import (
    list_documents,
    get_document_by_id,
    delete_document,
)
from backend.services.vector_store_client import delete_by_source_async
from backend.dependencies import get_chroma_collection
from backend.core.config import settings
from backend.models.responses import DocumentListResponse, DocumentRecord
from backend.limiter import limiter

documents_router = APIRouter()


@documents_router.get("/", response_model=DocumentListResponse)
@limiter.limit("10/minute")
async def get_documents(request: Request) -> DocumentListResponse:
    documents = await list_documents(db_path=settings.registry_db_path)
    return DocumentListResponse(documents=[DocumentRecord(**doc) for doc in documents])


@documents_router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def delete_document_by_id(request: Request, doc_id: int):
    document = await get_document_by_id(settings.registry_db_path, doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    collection = get_chroma_collection()
    try:
        await delete_by_source_async(collection, document["sanitized_filename"])
        await delete_document(settings.registry_db_path, doc_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )

    return {"message": "Document deleted successfully."}
