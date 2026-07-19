from unittest.mock import MagicMock, patch, AsyncMock
from openai import RateLimitError, APIConnectionError, APITimeoutError
from backend.core.errors import EmbeddingCallError
from backend.services.embedder import embed_texts, embed_query
from backend.core.config import settings as global_settings


def _make_rate_limit_error():
    response = MagicMock()
    response.headers = {"x-request-id": "test-id"}
    response.status_code = 429
    response.request = MagicMock()
    return RateLimitError("Rate limited", response=response, body=None)


async def test_embed_texts_empty_list_returns_empty_list(mock_openai_client):
    embeddings = await embed_texts(
        texts=[],
        client=mock_openai_client,
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    assert embeddings == []


async def test_embed_texts_single_input_returns_1536_vector(mock_openai_client):
    embeddings = await embed_texts(
        texts=["hello"],
        client=mock_openai_client,
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1536


async def test_embed_query_returns_single_float_list(mock_openai_client):
    embeddings = await embed_query(
        query="hello",
        client=mock_openai_client,
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1536
    assert not any(isinstance(x, list) for x in embeddings)


async def test_embed_texts_batches_150_texts_into_two_calls(mock_openai_client):
    texts = [f"text_{i}" for i in range(150)]

    embed_response_1 = MagicMock()
    embed_response_1.data = [
        MagicMock(embedding=[0.1 + i * 0.001] * 1536) for i in range(100)
    ]

    embed_response_2 = MagicMock()
    embed_response_2.data = [
        MagicMock(embedding=[0.2 + i * 0.001] * 1536) for i in range(50)
    ]

    mock_openai_client.embeddings.create = AsyncMock(
        side_effect=[embed_response_1, embed_response_2]
    )

    with patch.object(global_settings, "embed_batch_size", 100):
        result = await embed_texts(
            texts, mock_openai_client, "text-embedding-3-small"
        )

    assert len(result) == 150
    assert mock_openai_client.embeddings.create.call_count == 2


async def test_embed_texts_retries_on_rate_limit(mock_openai_client):
    embed_response = MagicMock()
    embed_response.data = [MagicMock(embedding=[0.1] * 1536)]

    mock_openai_client.embeddings.create = AsyncMock(
        side_effect=[_make_rate_limit_error(), embed_response]
    )

    result = await embed_texts(
        ["hello"], mock_openai_client, "text-embedding-3-small"
    )

    assert len(result) == 1
    assert len(result[0]) == 1536
    assert mock_openai_client.embeddings.create.call_count == 2


async def test_embed_texts_raises_after_exhausting_retries(mock_openai_client):
    mock_openai_client.embeddings.create = AsyncMock(
        side_effect=_make_rate_limit_error()
    )

    try:
        await embed_texts(
            ["hello"], mock_openai_client, "text-embedding-3-small"
        )
    except EmbeddingCallError:
        pass
    else:
        raise AssertionError("Expected EmbeddingCallError")


async def test_embed_texts_retries_on_connection_error(mock_openai_client):
    embed_response = MagicMock()
    embed_response.data = [MagicMock(embedding=[0.1] * 1536)]

    mock_openai_client.embeddings.create = AsyncMock(
        side_effect=[APIConnectionError(request=MagicMock()), embed_response]
    )

    result = await embed_texts(
        ["hello"], mock_openai_client, "text-embedding-3-small"
    )

    assert len(result) == 1
    assert len(result[0]) == 1536
    assert mock_openai_client.embeddings.create.call_count == 2


async def test_embed_texts_retries_on_timeout(mock_openai_client):
    embed_response = MagicMock()
    embed_response.data = [MagicMock(embedding=[0.1] * 1536)]

    mock_openai_client.embeddings.create = AsyncMock(
        side_effect=[APITimeoutError(MagicMock()), embed_response]
    )

    result = await embed_texts(
        ["hello"], mock_openai_client, "text-embedding-3-small"
    )

    assert len(result) == 1
    assert len(result[0]) == 1536
    assert mock_openai_client.embeddings.create.call_count == 2
