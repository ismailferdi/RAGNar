import chromadb
import asyncio
from chromadb.api import ClientAPI
from backend.core.errors import EmbeddingModelMismatchError, ChunkConfigMismatchError
from backend.services.chunker import TextChunk

COLLECTION_NAME = "documents"


def get_or_create_collection(
    client: ClientAPI, embedding_model: str, chunk_size: int, chunk_overlap: int
) -> chromadb.Collection:

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )

    return collection


def validate_collection_config(
    collection: chromadb.Collection,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):

    metadata = collection.metadata

    if metadata.get("embedding_model") != embedding_model:
        raise EmbeddingModelMismatchError(
            f"Embedding model mismatch: collection uses '{metadata.get('embedding_model')}', "
            f"but configured model is '{embedding_model}'"
        )
    if (
        metadata.get("chunk_size") != chunk_size
        or metadata.get("chunk_overlap") != chunk_overlap
    ):
        raise ChunkConfigMismatchError(
            f"Chunk configuration mismatch: collection expects chunk_size={metadata.get('chunk_size')}, chunk_overlap={metadata.get('chunk_overlap')}, "
            f"but got chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )


def add_chunks(
    collection: chromadb.Collection,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
):

    collection.add(
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        ids=[f"{c.source_file}__{c.chunk_index}" for c in chunks],
        metadatas=[
            {
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
            }
            for c in chunks
        ],
    )


async def source_exists_async(collection, source_file: str) -> bool:
    return await asyncio.to_thread(
        lambda: len(collection.get(where={"source_file": source_file}, limit=1)["ids"])
        > 0
    )


async def delete_by_source_async(collection, source_file: str) -> None:
    await asyncio.to_thread(collection.delete, where={"source_file": source_file})
    collection.delete(where={"source_file": source_file})


def search(
    collection: chromadb.Collection, query_embedding: list[float], top_k: int
) -> list[dict]:

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    result_dics = []

    for doc, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        result_dics.append(
            {
                "text": doc,
                "source_file": metadata["source_file"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return result_dics


def list_sources(collection: chromadb.Collection) -> list[str]:

    results = collection.get(include=["metadatas"])

    sources = set()

    for metadata in results["metadatas"]:
        sources.add(metadata["source_file"])

    return list(sources)
