from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import aiofiles

from backend.core.config import settings, MAX_UPLOAD_SIZE_MB
from backend.core.errors import (
    FileTooLargeError,
    DocumentAlreadyIngestedError,
    UnsupportedFileTypeError,
    EmptyDocumentError,
    EmbeddingCallError,
)
from backend.models.responses import IngestResponse
from backend.services.document_loader import sanitize_filename, load_document
from backend.services.vector_store_client import (
    source_exists,
    delete_by_source,
    add_chunks,
)
from backend.services.doc_registry import (
    delete_document,
    get_document_id_by_sanitized_filename,
    register_document,
)
from backend.services.chunker import chunk_document
from backend.services.embedder import embed_texts
from backend.dependencies import get_chroma_collection, get_openai_client

ingest_router = APIRouter()


@ingest_router.post("/", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), force: bool = Query(default=False)):

    try:

        filename = sanitize_filename(file.filename)
        upload_dir = Path(settings.document_store_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        collection = get_chroma_collection()
        client = get_openai_client()

        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        total_size = 0

        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await file.read(1024 * 1024):
                total_size += len(content)
                if total_size > max_bytes:
                    raise FileTooLargeError(
                        f"File size exceeds the maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB."
                    )
                await out_file.write(content)

        doc = await run_in_threadpool(load_document, str(file_path), file.filename)
        already_exists = await run_in_threadpool(
            source_exists, collection, doc.source_file
        )

        if already_exists and not force:
            raise DocumentAlreadyIngestedError(
                f"The document '{doc.source_file}' has already been ingested. Use the 'force' parameter to overwrite."
            )

        if force and already_exists:
            await run_in_threadpool(delete_by_source, collection, doc.source_file)
            doc_id = await run_in_threadpool(
                get_document_id_by_sanitized_filename,
                settings.registry_db_path,
                filename,
            )
            if doc_id is not None:
                await run_in_threadpool(
                    delete_document, settings.registry_db_path, doc_id
                )

        chunks = await run_in_threadpool(
            chunk_document, doc, settings.chunk_size, settings.chunk_overlap
        )
        texts = [chunk.text for chunk in chunks]
        embeddings = await run_in_threadpool(
            embed_texts, texts, client, settings.embedding_model
        )
        await run_in_threadpool(add_chunks, collection, chunks, embeddings)
        await run_in_threadpool(
            register_document,
            settings.registry_db_path,
            file.filename,
            filename,
            len(chunks),
        )

        return IngestResponse(
            message=f"File '{file.filename}' ingested successfully.",
            sanitized_filename=filename,
            chunk_count=len(chunks),
        )
    except OSError as e:
        raise HTTPException(
            status_code=500, detail="Failed to save uploaded file."
        ) from e
    except UnsupportedFileTypeError as e:
        raise HTTPException(
            status_code=422, detail="Only PDF and TXT files are supported."
        ) from e
    except EmptyDocumentError as e:
        raise HTTPException(
            status_code=422, detail="Document contains no extractable text."
        ) from e
    except DocumentAlreadyIngestedError as e:
        raise HTTPException(
            status_code=409,
            detail="Document already ingested. Use force=true to overwrite.",
        ) from e
    except FileTooLargeError as e:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {MAX_UPLOAD_SIZE_MB} MB limit."
        ) from e
    except EmbeddingCallError as e:
        raise HTTPException(
            status_code=502, detail="Embedding service temporarily unavailable."
        ) from e
