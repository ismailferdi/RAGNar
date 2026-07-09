from fastapi import APIRouter, status, HTTPException
from dataclasses import asdict

from backend.models.requests import AskRequest
from backend.models.responses import AskResponse, SourceChunkResponse
from backend.core.config import settings
from backend.core.errors import (
    EmbeddingModelMismatchError,
    ChunkConfigMismatchError,
    LLMCallError,
)
from backend.services.retriever import retrieve
from backend.services.prompt_builder import build_prompt
from backend.services.llm_client import generate_answer
from backend.dependencies import get_chroma_collection, get_openai_client

ask_router = APIRouter()


@ask_router.post("/", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:

    try:

        effective_top_k = request.top_k or settings.top_k

        collection = get_chroma_collection()
        client = get_openai_client()

        source_chunks = retrieve(
            query=request.question,
            collection=collection,
            client=client,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=effective_top_k,
            min_similarity=settings.min_similarity,
        )

        messages = build_prompt(
            question=request.question,
            chunks=source_chunks,
            max_context_tokens=settings.max_context_tokens,
        )

        llm_response = generate_answer(
            messages=messages, client=client, model=settings.chat_model
        )

        grounded = len(source_chunks) > 0

        return AskResponse(
            answer=llm_response.answer,
            sources=[SourceChunkResponse(**asdict(chunk)) for chunk in source_chunks],
            grounded=grounded,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )
    except (EmbeddingModelMismatchError, ChunkConfigMismatchError) as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The vector store must be wiped and re-ingested due to a mismatch in embedding model or chunk configuration. Please re-ingest your documents.",
        ) from e
    except LLMCallError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="An error occurred while calling the LLM. Please try again later.",
        ) from e
