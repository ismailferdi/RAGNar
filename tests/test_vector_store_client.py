import uuid
import pytest
from chromadb.api import ClientAPI
from backend.services.vector_store_client import (
    get_or_create_collection,
    validate_collection_config,
    add_chunks,
    source_exists_async,
    delete_by_source_async,
)
from backend.core.errors import EmbeddingModelMismatchError
from backend.services.chunker import TextChunk
from tests.conftest import EMBEDDING_DIM

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 375


def _unique_name(prefix: str = "test") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _make_chunks(source_file: str, count: int):
    return [
        TextChunk(
            text=f"Chunk {i} from {source_file}",
            source_file=source_file,
            chunk_index=i,
            char_start=0,
            char_end=10,
            total_chunks=count,
        )
        for i in range(count)
    ]


def test_get_or_create_collection_creates_with_correct_metadata(chroma_client: ClientAPI):
    collection = get_or_create_collection(
        chroma_client, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
    )
    assert collection.name == COLLECTION_NAME
    assert collection.metadata["embedding_model"] == EMBEDDING_MODEL
    assert collection.metadata["chunk_size"] == CHUNK_SIZE
    assert collection.metadata["chunk_overlap"] == CHUNK_OVERLAP


def test_get_or_create_collection_is_idempotent(chroma_client: ClientAPI):
    c1 = get_or_create_collection(
        chroma_client, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
    )
    c2 = get_or_create_collection(
        chroma_client, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
    )
    assert c1.name == c2.name
    assert c1.metadata["embedding_model"] == EMBEDDING_MODEL


def test_validate_collection_config_raises_on_model_mismatch(chroma_client: ClientAPI):
    collection = chroma_client.create_collection(
        _unique_name("mismatch"),
        metadata={"embedding_model": "some-other-model", "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
    )
    with pytest.raises(EmbeddingModelMismatchError):
        validate_collection_config(collection, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)


def test_validate_collection_config_passes_on_match(chroma_client: ClientAPI):
    collection = chroma_client.create_collection(
        _unique_name("match"),
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )
    validate_collection_config(collection, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)


async def test_source_exists_false_for_empty_collection(in_memory_chroma):
    exists = await source_exists_async(in_memory_chroma, "test_doc.pdf")
    assert exists is False


async def test_source_exists_true_after_add_chunks(in_memory_chroma, sample_chunks):
    embeddings = [[0.1] * EMBEDDING_DIM for _ in sample_chunks]
    add_chunks(in_memory_chroma, sample_chunks, embeddings)

    exists = await source_exists_async(in_memory_chroma, "test_doc.pdf")
    assert exists is True


async def test_delete_by_source_removes_only_target_source(in_memory_chroma):
    chunks_1 = _make_chunks("doc_a.pdf", 3)
    chunks_2 = _make_chunks("doc_b.pdf", 2)
    all_chunks = chunks_1 + chunks_2
    embeddings = [[0.1] * EMBEDDING_DIM for _ in all_chunks]
    add_chunks(in_memory_chroma, all_chunks, embeddings)

    await delete_by_source_async(in_memory_chroma, "doc_a.pdf")

    assert await source_exists_async(in_memory_chroma, "doc_a.pdf") is False
    assert await source_exists_async(in_memory_chroma, "doc_b.pdf") is True


async def test_add_chunks_duplicate_id_does_not_raise(in_memory_chroma, sample_chunks):
    embeddings = [[0.1] * EMBEDDING_DIM for _ in sample_chunks]
    add_chunks(in_memory_chroma, sample_chunks, embeddings)

    add_chunks(in_memory_chroma, sample_chunks, embeddings)

    results = in_memory_chroma.get(
        ids=[f"{c.source_file}__{c.chunk_index}" for c in sample_chunks]
    )
    assert len(results["ids"]) == 3
