import openai
import time
from backend.core.config import settings
from backend.core.errors import EmbeddingCallError

MAX_RETRIES = 3


def _call_embedding_api(
    batch: list[str], client: openai.OpenAI, model: str
) -> list[list[float]]:
    # NVIDIA model requires multimodal content format even for text
    formatted_input = [{"content": [{"type": "text", "text": text}]} for text in batch]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=model,
                input=formatted_input,
                encoding_format="float",
            )
            return [item.embedding for item in response.data]
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
            else:
                raise EmbeddingCallError(
                    "Embedding service temporarily unavailable. Please try again later."
                ) from e


def embed_texts(
    texts: list[str], client: openai.OpenAI, model: str
) -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    for i in range(0, len(texts), settings.embed_batch_size):
        batch = texts[i : i + settings.embed_batch_size]
        batch_embeddings = _call_embedding_api(batch, client, model)
        embeddings.extend(batch_embeddings)

    return embeddings


def embed_query(query: str, client: openai.OpenAI, model: str) -> list[float]:
    result = embed_texts([query], client, model)
    return result[0] if result else []
