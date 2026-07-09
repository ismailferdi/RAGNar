from fastapi import APIRouter, status, Response, HTTPException

from backend.services.doc_registry import (
    list_documents,
    get_document_by_id,
    delete_document,
)
from backend.services.vector_store_client import delete_by_source
from backend.dependencies import get_chroma_collection
from backend.core.errors import DocumentNotFoundError
from backend.core.config import settings
from backend.models.responses import DocumentListResponse, DocumentRecord

documents_router = APIRouter()


@documents_router.get("/", response_model=DocumentListResponse)
def get_documents() -> DocumentListResponse:

    documents = list_documents(db_path=settings.registry_db_path)
    return DocumentListResponse(documents=[DocumentRecord(**doc) for doc in documents])


@documents_router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document_by_id(doc_id: int):

    document = get_document_by_id(settings.registry_db_path, doc_id)

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    collection = get_chroma_collection()
    delete_by_source(collection, document["sanitized_filename"])
    delete_document(settings.registry_db_path, doc_id)

    return {"message": "Document deleted successfully."}
