import openai
import chromadb
from chromadb.api import ClientAPI
from backend.core.config import settings
from backend.services.vector_store_client import (
    get_or_create_collection,
    validate_collection_config,
)
from backend.services.doc_registry import initialize_registry

_openai_client: openai.AsyncOpenAI | None = None
_chroma_client: ClientAPI | None = None
_collection: chromadb.Collection | None = None
_registry_db_path: str = settings.registry_db_path


async def initialize_clients() -> None:
    global _openai_client, _chroma_client, _collection

    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.base_url,
        )

    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.vector_store_path)

    if _collection is None:
        _collection = get_or_create_collection(
            client=_chroma_client,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    validate_collection_config(
        collection=_collection,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    await initialize_registry(db_path=_registry_db_path)


def get_openai_client() -> openai.AsyncOpenAI:
    if _openai_client is None:
        raise RuntimeError(
            "OpenAI client has not been initialized. Call initialize_clients() first."
        )
    return _openai_client


def get_chroma_collection() -> chromadb.Collection:
    if _collection is None:
        raise RuntimeError(
            "Chroma collection has not been initialized. Call initialize_clients() first."
        )
    return _collection
