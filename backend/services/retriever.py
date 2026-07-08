from dataclasses import dataclass
import openai
from backend.services.embedder import embed_query
from backend.services.vector_store_client import search, validate_collection_config


@dataclass
class SourceChunk:
    text: str
    source_file: str
    chunk_index: int
    similarity_score: float


def retrieve(
    query: str,
    collection,
    client: openai.OpenAI,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    min_similarity: float,
) -> list[SourceChunk]:

    validate_collection_config(
        collection=collection,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embedding = embed_query(query, client, embedding_model)

    results = search(collection, embedding, top_k)

    results = [
        {
            "text": result["text"],
            "source_file": result["source_file"],
            "chunk_index": result["chunk_index"],
            "similarity_score": 1 / (1 + result["distance"]),
        }
        for result in results
        if 1 / (1 + result["distance"]) >= min_similarity
    ]

    if results:
        results = [
            SourceChunk(
                text=result["text"],
                source_file=result["source_file"],
                chunk_index=result["chunk_index"],
                similarity_score=result["similarity_score"],
            )
            for result in results
        ]

        return sorted(results, key=lambda x: x.similarity_score, reverse=True)

    else:
        return []
