import openai
import asyncio
from backend.core.config import settings
from backend.core.errors import EmbeddingCallError

MAX_RETRIES = 3


async def _call_embedding_api(
    batch: list[str],
    client: openai.AsyncOpenAI,
    model: str,
    input_type: str = "passage",
) -> list[list[float]]:

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float",
                extra_body={"input_type": input_type},
            )
            return [item.embedding for item in response.data]
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
            else:
                raise EmbeddingCallError(
                    "Embedding service temporarily unavailable. Please try again later."
                ) from e


async def embed_texts(
    texts: list[str],
    client: openai.AsyncOpenAI,
    model: str,
    input_type: str = "passage",
) -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    for i in range(0, len(texts), settings.embed_batch_size):
        batch = texts[i : i + settings.embed_batch_size]
        batch_embeddings = await _call_embedding_api(batch, client, model, input_type)
        embeddings.extend(batch_embeddings)

    return embeddings


async def embed_query(
    query: str, client: openai.AsyncOpenAI, model: str
) -> list[float]:
    result = await embed_texts([query], client, model, input_type="query")
    return result[0] if result else []
