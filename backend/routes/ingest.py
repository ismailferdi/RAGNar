from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from pathlib import Path
import aiofiles
import asyncio

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
    source_exists_async,
    delete_by_source_async,
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
from backend.limiter import limiter

ingest_router = APIRouter()

_ingest_locks: dict[str, asyncio.Lock] = {}
_ingest_lock_users: dict[str, int] = {}
_ingest_locks_guard = asyncio.Lock()


@asynccontextmanager
async def _acquire_ingest_lock(filename: str):
    async with _ingest_locks_guard:
        lock = _ingest_locks.get(filename)
        if lock is None:
            lock = asyncio.Lock()
            _ingest_locks[filename] = lock
            _ingest_lock_users[filename] = 0
        _ingest_lock_users[filename] += 1

    try:
        async with lock:
            yield
    finally:
        async with _ingest_locks_guard:
            _ingest_lock_users[filename] -= 1
            if _ingest_lock_users[filename] == 0:
                _ingest_lock_users.pop(filename, None)
                _ingest_locks.pop(filename, None)


@ingest_router.post("/", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_file(
    request: Request, file: UploadFile = File(...), force: bool = Query(default=False)
):

    file_path = None
    try:
        if file.filename is None:
            raise HTTPException(
                status_code=400, detail="Upload must include a filename."
            )
        filename = sanitize_filename(file.filename)

        async with _acquire_ingest_lock(filename):
            upload_dir = Path(settings.document_store_path)
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / filename

            collection = get_chroma_collection()
            client = get_openai_client()

            max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
            total_bytes = 0

            async with aiofiles.open(file_path, "wb") as out_file:
                while content := await file.read(1024 * 1024):
                    total_bytes += len(content)
                    if total_bytes > max_bytes:
                        await out_file.close()
                        file_path.unlink(missing_ok=True)
                        raise FileTooLargeError(
                            f"File exceeds {MAX_UPLOAD_SIZE_MB} MB limit."
                        )
                    await out_file.write(content)

            doc = await run_in_threadpool(load_document, str(file_path), file.filename)
            already_exists = await source_exists_async(collection, doc.source_file)

            if already_exists and not force:
                raise DocumentAlreadyIngestedError(
                    f"The document '{doc.source_file}' has already been ingested. Use the 'force' parameter to overwrite."
                )

            chunks = await run_in_threadpool(
                chunk_document, doc, settings.chunk_size, settings.chunk_overlap
            )
            texts = [chunk.text for chunk in chunks]
            embeddings = await embed_texts(texts, client, settings.embedding_model)

            if force and already_exists:
                await delete_by_source_async(collection, doc.source_file)
                doc_id = await get_document_id_by_sanitized_filename(
                    settings.registry_db_path, filename
                )

                if doc_id is not None:
                    await delete_document(settings.registry_db_path, doc_id)

            await run_in_threadpool(add_chunks, collection, chunks, embeddings)
            await register_document(
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
    except EmbeddingCallError as e:
        raise HTTPException(
            status_code=502, detail="Embedding service temporarily unavailable."
        ) from e
    except OSError as e:
        raise HTTPException(
            status_code=500, detail="Failed to save uploaded file."
        ) from e

    finally:
        if file_path is not None and file_path.exists():
            await run_in_threadpool(file_path.unlink)
