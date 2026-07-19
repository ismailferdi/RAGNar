# AGENTS.md

## Architecture

- **Backend** (`backend/`): FastAPI, single package. Entrypoint: `backend.main:app`.
- **Frontend** (`frontend/`): Streamlit, separate process, HTTP to backend at `RAGNAR_API_URL` (default `http://localhost:8000`).
- **Root `requirements.txt`** covers everything. No dev-dependency split.
- Routes mounted **without** `/api/v1` prefix: `/documents`, `/ask`, `/health`, `/ingest`.
- Frontend API calls use trailing slashes: `/ask/`, `/ingest/`, `/documents/`.

## Commands

| Action | Command |
|---|---|
| Install | `pip install -r requirements.txt` (or `make install`) |
| Run API | `uvicorn backend.main:app --reload` (or `make run-api`) |
| Run UI | `streamlit run frontend/app.py` (or `make run-ui`) |
| Run tests | `pytest tests/ -v` (or `make test`) |
| Run eval | `python eval/run_eval.py` (or `make eval`) |
| Lint | `ruff check .` (no config — default rules) |
| Wipe vector store | `make wipe-vector-store` |
| Ingest eval docs | `make ingest-eval-docs` (requires running API) |

## Config & Setup

- pydantic-settings loads from **root `.env`** at `backend/core/config.py:5`.
- `.env` **must exist** (no defaults for `openrouter_api_key`). `.env.example` is stale — missing `TIKTOKEN_MODEL`, `TIKTOKEN_ENCODER`, `GROUNDING_THRESHOLD`, `RAGNAR_API_URL`.
- **NVIDIA NIM, not OpenAI**: `OPENROUTER_API_KEY` (misnamed — value is an NVIDIA key) + `BASE_URL=https://integrate.api.nvidia.com/v1`. `.env.example` shows OpenRouter defaults.
- `vector_store_path` and `document_store_path` are relative to project root. `registry_db_path` defaults to `registry.db` (run dir). These dirs are gitignored, created at runtime.
- `MAX_UPLOAD_SIZE_MB = 25` (hardcoded in `backend/core/config.py:6`).
- `pytest.ini` sets `asyncio_mode = auto` — all tests are async by default.

## Key Quirks

- **Fully async stack**: `openai.AsyncOpenAI`, `aiosqlite`, `aiofiles`. CPU-bound work (load, chunk, sync ChromaDB) runs via `run_in_threadpool`.
- **LLM call** (`llm_client.py:18`): sleeps **0.2s** before every call; sends `extra_body={"reasoning": {"enabled": True}}` (NVIDIA NIM supports this).
- **Embedding**: NVIDIA Nemotron is asymmetric — `extra_body={"input_type": "passage"}` for indexing, `"query"` for search. Batching via `EMBED_BATCH_SIZE` (100), retries 3× with exponential backoff.
- **Rate limiting**: All routes have `@limiter.limit("10/minute")` from `slowapi`. Every decorated function **must** have `request: Request` as first param.
- **FileTooLargeError** extends `HTTPException` directly — FastAPI intercepts it before the route's `except` block (dead code in `ingest.py`).
- **Exception handlers** are in `main.py` (global). Route-level `except` blocks re-raise as `HTTPException`, which bypasses them.
- **CORS**: explicit origin list from `core_allowed_origins` config (no wildcard).
- **Ingest**: per-filename `asyncio.Lock` for concurrency safety; always deletes uploaded file in `finally` block.

## Data Flow

```
ingest: load_document (threadpool) → chunk_document (threadpool) → embed_texts (async, batched 100) → add_chunks (threadpool) → register_document (async aiosqlite)
ask:   embed_query (async) → search (sync chromadb, run_in_threadpool) → build_prompt (sync) → generate_answer (async)
```

ChromaDB distance is L2; converted to similarity as `1 / (1 + distance)`.

## Known Bugs

- `vector_store_client.py:80-81` — `delete_by_source_async` calls `collection.delete` twice (once via `to_thread`, once directly). Second call is unintended.
- `chunker.py:31` — `char_start` uses `str.find()` which matches first occurrence, giving wrong offsets when text repeats.
- Health route bug is **fixed** — `health.py` is now `async def` with proper `await`.

## Tests

- Tests use `chromadb.EphemeralClient`, mocked `AsyncOpenAI` client, and `tmp_path` for registry DB.
- `conftest.py` provides `test_client` (FastAPI `TestClient` with patched deps), `sample_chunks`, `tmp_pdf`.
- Eval (`eval/run_eval.py`) ingests docs via HTTP (`POST /ingest/`) then calls service-layer functions directly. Requires API server running for ingestion.

## Per-Service

- **Chunker**: `chunk_size`/`chunk_overlap` passed as args — independently testable.
- **Retriever**: calls `validate_collection_config` inline. Returns empty list (never raises) when all chunks below `min_similarity`.
- **Prompt builder**: drops lowest-similarity chunk (from **end** of list — assumes descending sort) if token budget exceeded.
- **Doc registry**: `aiosqlite` (async), not stdlib sqlite3.
- **Frontend**: `source_viewer.py` renders expandable source panels; `uploader.py` handles force-re-ingest flow.
