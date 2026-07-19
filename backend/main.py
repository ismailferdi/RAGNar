from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.routes.documents import documents_router
from backend.routes.ask import ask_router
from backend.routes.health import health_router
from backend.routes.ingest import ingest_router
from backend.dependencies import initialize_clients
from backend.core.config import settings
from backend.core.errors import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    EmbeddingCallError,
    EmbeddingModelMismatchError,
    ChunkConfigMismatchError,
    LLMCallError,
    DocumentAlreadyIngestedError,
    DocumentNotFoundError,
    FileTooLargeError,
)
from backend import dependencies
from backend.limiter import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    await initialize_clients()
    yield
    print("Shutting down...")
    client = getattr(dependencies, "_openai_client", None)
    if client is not None:
        if hasattr(client, "_client") and hasattr(client._client, "aclose"):
            await client._client.aclose()
        if hasattr(client, "close"):
            await client.close()


app = FastAPI(
    title="Ragnar",
    description="RAG-powered document Q&A API",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.core_allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=["*"],
)

app.state.limiter = limiter


app.include_router(documents_router, prefix="/documents", tags=["Documents"])
app.include_router(ask_router, prefix="/ask", tags=["Ask"])
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])


@app.get("/")
async def root():
    return {"message": "Welcome to the RAG-powered document Q&A API!"}


app.add_exception_handler(429, _rate_limit_exceeded_handler)


@app.exception_handler(EmptyDocumentError)
async def empty_document_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": "The document is empty."},
    )


@app.exception_handler(UnsupportedFileTypeError)
async def unsupported_file_type_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": "The file type is not supported."},
    )


@app.exception_handler(EmbeddingCallError)
async def embedding_call_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "An error occurred while calling the embedding service."},
    )


@app.exception_handler(EmbeddingModelMismatchError)
async def embedding_model_mismatch_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": "The embedding model does not match the expected model."},
    )


@app.exception_handler(ChunkConfigMismatchError)
async def chunk_config_mismatch_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "message": "The chunk configuration does not match the expected configuration."
        },
    )


@app.exception_handler(LLMCallError)
async def llm_call_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "An error occurred while calling the LLM service."},
    )


@app.exception_handler(DocumentAlreadyIngestedError)
async def document_already_ingested_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": "The document has already been ingested."},
    )


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.exception_handler(FileTooLargeError)
async def file_too_large_exception_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"message": "The uploaded file is too large."},
    )
