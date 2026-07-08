import openai
from backend.core.config import settings
import time
from backend.core.errors import EmbeddingCallError

# Note:
# `nvidia/llama-nemotron-embed-vl-1b-v2:free` produces 2048-dimensional vectors;
# changing to a different model
# produces vectors in a different space —
# existing ChromaDB entries become incompatible and the collection must be wiped and re-ingested


MAX_RETRIES = 3


def _call_embedding_api(
    batch: list[str], client: openai.OpenAI, model: str
) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=model, input=batch)
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

    else:
        if len(texts) > settings.embed_batch_size:
            embeddings = []
            for i in range(0, len(texts), settings.embed_batch_size):
                batch = texts[i : i + settings.embed_batch_size]
                batch_embeddings = _call_embedding_api(batch, client, model)
                embeddings.extend(batch_embeddings)
            return embeddings
        else:
            return _call_embedding_api(texts, client, model)


def embed_query(query: str, client: openai.OpenAI, model: str) -> list[float]:

    embedding = embed_texts([query], client, model)
    return embedding[0] if embedding else []
