import uuid
from unittest.mock import MagicMock, AsyncMock
import pytest
from backend.services.retriever import retrieve, SourceChunk
from backend.services.vector_store_client import add_chunks, get_or_create_collection
from backend.services.chunker import TextChunk
from backend.core.errors import EmbeddingModelMismatchError, ChunkConfigMismatchError
from tests.conftest import EMBEDDING_DIM

EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 375


def _unique_collection(chroma_client, metadata: dict | None = None):
    name = f"retrieval_test_{uuid.uuid4().hex}"
    return chroma_client.create_collection(name, metadata=metadata)


async def test_retrieve_empty_list_when_no_matching_chunks(
    chroma_client, mock_openai_client
):
    collection = get_or_create_collection(
        chroma_client, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
    )
    add_chunks(
        collection,
        [
            TextChunk(
                text="Python is a programming language.",
                source_file="doc.pdf",
                chunk_index=0,
                char_start=0,
                char_end=35,
                total_chunks=1,
            )
        ],
        [[0.01] * EMBEDDING_DIM],
    )

    result = await retrieve(
        query="something completely unrelated",
        collection=collection,
        client=mock_openai_client,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=5,
        min_similarity=0.99,
    )

    assert result == []


async def test_retrieve_filters_by_min_similarity(
    chroma_client, mock_openai_client
):
    collection = _unique_collection(chroma_client, metadata={
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    })
    close_embedding = [0.9] * EMBEDDING_DIM
    far_embedding = [0.1] * EMBEDDING_DIM

    chunks = [
        TextChunk(
            text="Close chunk about RAG.",
            source_file="doc.pdf",
            chunk_index=0,
            char_start=0,
            char_end=20,
            total_chunks=2,
        ),
        TextChunk(
            text="Far chunk about something else.",
            source_file="doc.pdf",
            chunk_index=1,
            char_start=21,
            char_end=50,
            total_chunks=2,
        ),
    ]
    add_chunks(collection, chunks, [close_embedding, far_embedding])

    mock_openai_client.embeddings.create = AsyncMock(return_value=MagicMock(
        data=[MagicMock(embedding=close_embedding)]
    ))

    result = await retrieve(
        query="RAG",
        collection=collection,
        client=mock_openai_client,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=5,
        min_similarity=0.5,
    )

    assert len(result) == 1
    assert result[0].text == "Close chunk about RAG."


async def test_retrieve_returns_sorted_by_similarity_descending(
    chroma_client, mock_openai_client
):
    collection = _unique_collection(chroma_client, metadata={
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    })
    emb1 = [0.2] * EMBEDDING_DIM
    emb2 = [0.5] * EMBEDDING_DIM
    emb3 = [0.9] * EMBEDDING_DIM

    chunks = [
        TextChunk(text="Chunk A", source_file="doc.pdf", chunk_index=0, char_start=0, char_end=7, total_chunks=3),
        TextChunk(text="Chunk B", source_file="doc.pdf", chunk_index=1, char_start=8, char_end=15, total_chunks=3),
        TextChunk(text="Chunk C", source_file="doc.pdf", chunk_index=2, char_start=16, char_end=23, total_chunks=3),
    ]
    add_chunks(collection, chunks, [emb1, emb2, emb3])

    mock_openai_client.embeddings.create = AsyncMock(return_value=MagicMock(
        data=[MagicMock(embedding=[0.9] * EMBEDDING_DIM)]
    ))

    result = await retrieve(
        query="test",
        collection=collection,
        client=mock_openai_client,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=5,
        min_similarity=0.0,
    )

    assert len(result) >= 1
    scores = [chunk.similarity_score for chunk in result]
    assert scores == sorted(scores, reverse=True)


async def test_retrieve_raises_on_embedding_model_mismatch(chroma_client, mock_openai_client):
    collection = chroma_client.create_collection(
        f"mismatch_{uuid.uuid4().hex}",
        metadata={"embedding_model": "different-model", "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
    )

    with pytest.raises(EmbeddingModelMismatchError):
        await retrieve(
            query="test",
            collection=collection,
            client=mock_openai_client,
            embedding_model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            top_k=5,
            min_similarity=0.3,
        )
