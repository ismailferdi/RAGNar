### RAG-Powered Document Q&A System

**Detailed Explanation:**
This project builds a retrieval-augmented generation (RAG) system that lets users upload any document — a PDF research paper, a government report, a company policy, a book chapter — and ask questions about it in plain English. Rather than feeding the entire document into an LLM (expensive, slow, context-limited), the system splits each document into overlapping text chunks, embeds them into a vector space using OpenAI's `text-embedding-3-small` model, stores them in a persistent ChromaDB vector database, and at query time retrieves only the most relevant chunks before passing them to `gpt-4o-mini` to generate a grounded answer with source citations. The core engineering challenge — and what separates this from a tutorial clone — is enforcing strict grounding: the system answers only from retrieved context, never from the LLM's general knowledge, and explicitly tells the user when no relevant content was found rather than hallucinating a confident-sounding response.

Several silent failure modes must be designed against explicitly. The embedding model name and chunking parameters are stored in ChromaDB collection metadata at creation time; before every search, the configured embedding model is validated against the stored name — a mismatch raises `EmbeddingModelMismatchError` immediately rather than returning semantically incoherent results. Re-uploading a document with the same filename is blocked by default, requiring an explicit `force=True` parameter to delete existing chunks and re-ingest. A similarity threshold check ensures that if all retrieved chunks fall below a minimum cosine similarity, the system returns a grounded "I could not find the answer in the uploaded documents" response rather than passing low-quality context to the LLM. A token budget check using `tiktoken` ensures the assembled prompt never silently truncates mid-chunk by the model; if the combined token count of system prompt, context, and question exceeds `MAX_CONTEXT_TOKENS`, the lowest-similarity chunks are dropped until it fits.

The final output is a FastAPI backend with four endpoints (`POST /ingest`, `POST /ask`, `GET /documents`, `DELETE /documents/{doc_id}`) backed by ChromaDB for vector storage and a SQLite registry for document metadata, paired with a Streamlit frontend that provides a file upload sidebar, a chat interface, and an expandable source viewer that shows the exact retrieved chunks behind each answer. The project also includes an evaluation harness that scores the system on a hand-written question set across two metrics: retrieval recall@5 (was the relevant chunk in the top-5 results?) and LLM-as-judge answer quality (1–5 score comparing the generated answer to the expected answer).

**Necessary Skills:**
*   **RAG Architecture:** Document chunking strategy (chunk size, overlap, splitter choice), embedding pipeline design, vector similarity search, context injection into prompts, grounding enforcement — understanding the full data flow from raw document to grounded answer.
*   **Vector Databases:** ChromaDB persistent client, collection creation with metadata, adding and querying embeddings with metadata filters, deleting documents by source metadata, understanding distance metrics (L2 vs. cosine) and how they map to similarity scores.
*   **Embedding Models & APIs:** OpenAI `text-embedding-3-small` via the `openai` Python client, batching embed calls to stay within API rate limits, enforcing model consistency between ingestion and query time.
*   **LLM Prompt Engineering:** Grounded-only system prompts, numbered citation format (`[1]`, `[2]`), few-shot examples of grounded vs. ungrounded answers, token budget management with `tiktoken`, LLM-as-judge scoring for eval.
*   **REST API Development:** FastAPI multipart file upload endpoints, async file I/O, Pydantic v2 request and response models, dependency injection with `Depends`, custom exception handlers, CORS configuration.
*   **Document Parsing:** PDF text extraction with `pypdf`, plain-text loading, filename sanitization, handling of scanned/image PDFs that yield empty text (detect and reject gracefully).
*   **Evaluation Harness Design:** Retrieval recall@k as a metric (distinct from answer quality), LLM-as-judge as a scalable quality measure, separating retrieval evaluation from generation evaluation so you can diagnose failures at the right layer.
*   **Testing:** pytest, mocking the OpenAI client to avoid real API calls in tests, building an in-memory ChromaDB instance for vector store tests, FastAPI TestClient for endpoint tests.

**Project Structure:**
```
ragnar/
├── data/
│   └── documents/                       # Uploaded files saved here at ingest time (gitignored)
│       └── .gitkeep
│
├── vector_store/                         # Persisted ChromaDB index (gitignored)
│   └── .gitkeep
│
├── backend/
│   ├── main.py                           # FastAPI app instance, startup handler, middleware, router mounting
│   ├── dependencies.py                   # Singleton providers: OpenAI client, ChromaDB client, document registry DB path
│   ├── core/
│   │   ├── config.py                     # pydantic-settings: API key, model names, chunk params, top-k, thresholds, CORS
│   │   └── errors.py                     # Custom exception classes
│   ├── models/
│   │   ├── requests.py                   # AskRequest, DeleteRequest
│   │   └── responses.py                  # AskResponse (answer + sources), IngestResponse, DocumentListResponse, HealthResponse
│   ├── routes/
│   │   ├── ask.py                        # POST /ask
│   │   ├── ingest.py                     # POST /ingest (multipart file upload)
│   │   ├── documents.py                  # GET /documents, DELETE /documents/{doc_id}
│   │   └── health.py                     # GET /health
│   └── services/
│       ├── document_loader.py            # Load PDF (pypdf) and TXT files; return raw text + source metadata
│       ├── chunker.py                    # RecursiveCharacterTextSplitter; produce overlapping chunk list with metadata
│       ├── embedder.py                   # Embed text lists via OpenAI; validate model consistency before every search
│       ├── vector_store_client.py        # ChromaDB wrapper: init collection with metadata, add, search, delete, list
│       ├── retriever.py                  # Embed query, search, apply similarity threshold, return SourceChunk list
│       ├── prompt_builder.py             # Assemble grounded RAG prompt with numbered context + token budget enforcement
│       ├── llm_client.py                 # OpenAI chat completion with grounded-only system prompt; return answer + usage
│       └── doc_registry.py              # SQLite document registry: insert, list, delete, lookup by id or sanitized filename
│
├── frontend/
│   ├── app.py                            # Streamlit entry point: page config, session state, layout
│   └── components/
│       ├── uploader.py                   # Sidebar file upload widget, ingest progress, document list with delete buttons
│       ├── chat.py                       # Chat history display, question input, answer rendering with inline citations
│       └── source_viewer.py             # Expandable "Sources" panel showing retrieved chunk text + filename + chunk index
│
├── eval/
│   ├── documents/                        # Small static document set used exclusively for eval (not gitignored — part of repo)
│   │   └── .gitkeep
│   ├── eval_set.json                     # 15–20 hand-written Q&A pairs: question, expected_answer, source_file, relevant_chunk_text
│   └── run_eval.py                       # Score system: retrieval recall@5 + LLM-as-judge answer quality (1–5)
│
├── tests/
│   ├── conftest.py                       # Shared fixtures: mock OpenAI client, in-memory ChromaDB, sample chunks, test client
│   ├── test_document_loader.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store_client.py
│   ├── test_retriever.py
│   ├── test_prompt_builder.py
│   ├── test_llm_client.py
│   ├── test_doc_registry.py
│   └── test_api.py
│
├── .env.example
├── requirements.txt
├── Makefile
└── README.md
```

**Full project guide (to-do list):**

- [x] Create project root directory named `ragnar`
- [x] Initialize a Git repository inside `ragnar`
- [x] Create a `.gitignore` that ignores `venv/`, `__pycache__/`, `.env`, `data/documents/*`, `!data/documents/.gitkeep`, `vector_store/*`, `!vector_store/.gitkeep`, `*.db`, and IDE settings — note that `eval/documents/` is **not** gitignored since those files are part of the repo
- [x] Create a virtual environment with `python -m venv venv`
- [x] Activate the virtual environment
- [x] Install `openai` package
- [x] Install `chromadb` package
- [x] Install `pypdf` package
- [x] Install `langchain-text-splitters` package
- [x] Install `tiktoken` package
- [x] Install `fastapi` package
- [x] Install `uvicorn` package
- [x] Install `python-multipart` package (required for FastAPI multipart file upload)
- [x] Install `pydantic-settings` package
- [x] Install `streamlit` package
- [x] Install `httpx` package
- [x] Install `pytest` package
- [x] Install `python-dotenv` package
- [x] Freeze installed package versions into `requirements.txt`
- [x] Create a `.env.example` file with `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `VECTOR_STORE_PATH`, `DOCUMENT_STORE_PATH`, `REGISTRY_DB_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SIMILARITY`, `MAX_CONTEXT_TOKENS`, `CORS_ALLOWED_ORIGINS`
- [x] Copy `.env.example` to `.env` and fill in real local development values (use `text-embedding-3-small` for `EMBEDDING_MODEL` and `gpt-4o-mini` for `CHAT_MODEL`)
- [x] Create `data/` directory
- [x] Create `data/documents/` directory with a `.gitkeep` file
- [x] Create `vector_store/` directory with a `.gitkeep` file
- [x] Create `backend/` directory
- [x] Create `backend/__init__.py`
- [x] Create `backend/core/` directory
- [x] Create `backend/core/__init__.py`
- [x] Create `backend/models/` directory
- [x] Create `backend/models/__init__.py`
- [x] Create `backend/routes/` directory
- [x] Create `backend/routes/__init__.py`
- [x] Create `backend/services/` directory
- [x] Create `backend/services/__init__.py`
- [x] Create `frontend/` directory
- [x] Create `frontend/components/` directory
- [x] Create `eval/` directory
- [x] Create `eval/documents/` directory with a `.gitkeep` file
- [x] Create `tests/` directory
- [x] Create `tests/__init__.py`
- [x] Download two or three diverse public-domain documents to use as development test documents — recommended sources include [Project Gutenberg](https://www.gutenberg.org/) (plain-text books), [arXiv.org](https://arxiv.org/) (PDFs of research papers), or [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) (annual reports); save them to `data/documents/` locally (gitignored)
- [x] Also place two or three smaller, simpler documents in `eval/documents/` — these must be documents whose content you know well enough to hand-write 15–20 specific factual questions about them
- [x] Create `backend/services/document_loader.py`
- [x] Import `pypdf.PdfReader` for PDF loading
- [x] Define a dataclass `LoadedDocument` with fields: `text: str`, `source_file: str`, `file_type: str` ("pdf" or "txt")
- [x] Implement `sanitize_filename(filename: str) -> str` that lowercases the name, replaces spaces and special characters with underscores, and strips any path components — only the base filename should survive; this sanitized name becomes the stable identifier stored in ChromaDB metadata and the document registry
- [x] Implement `load_pdf(file_path: str, source_file: str) -> LoadedDocument` that opens the file with `PdfReader`, concatenates `page.extract_text()` across all pages with a newline separator, and returns a `LoadedDocument`; if the concatenated text is empty or contains only whitespace after extraction, raise `EmptyDocumentError` — this is the most common failure mode for scanned PDFs that contain images rather than selectable text
- [x] Implement `load_txt(file_path: str, source_file: str) -> LoadedDocument` that reads the file as UTF-8 text with a fallback to `latin-1` if UTF-8 decoding fails, and returns a `LoadedDocument`
- [x] Implement `load_document(file_path: str, original_filename: str) -> LoadedDocument` that dispatches to `load_pdf` or `load_txt` based on the file extension (`.pdf` or `.txt`), raises `UnsupportedFileTypeError` for any other extension, and uses `sanitize_filename(original_filename)` as the `source_file` identifier
- [x] Create `backend/services/chunker.py`
- [x] Import `RecursiveCharacterTextSplitter` from `langchain_text_splitters`
- [x] Define a dataclass `TextChunk` with fields: `text: str`, `source_file: str`, `chunk_index: int`, `char_start: int`, `char_end: int`, `total_chunks: int`
- [x] Implement `chunk_document(doc: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[TextChunk]` that instantiates `RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)`, calls `split_text(doc.text)`, and wraps each split into a `TextChunk` with correct `chunk_index` (0-based) and `total_chunks`
- [x] Note in a code comment that `chunk_size` and `chunk_overlap` are passed in (not read from config directly) so chunker.py is independently testable without a settings object
- [x] Implement `estimate_char_offsets(chunks: list[str], full_text: str) -> list[tuple[int, int]]` that uses `str.find` with a running offset to approximate `(char_start, char_end)` for each chunk — used for the `TextChunk.char_start` / `char_end` fields; note in a comment that overlapping chunks mean char ranges will overlap, which is expected behavior
- [x] Create `backend/services/embedder.py`
- [x] Import the `openai.OpenAI` client
- [x] Implement `embed_texts(texts: list[str], client: openai.OpenAI, model: str) -> list[list[float]]` that calls `client.embeddings.create(model=model, input=texts)` and returns the list of embedding vectors; if `texts` is empty, return an empty list immediately without making an API call
- [x] Implement `embed_query(query: str, client: openai.OpenAI, model: str) -> list[float]` that calls `embed_texts([query], client, model)` and returns the single embedding vector
- [x] Note in a code comment that `text-embedding-3-small` produces 1536-dimensional vectors; changing to a different model (e.g., `text-embedding-ada-002` or `text-embedding-3-large`) produces vectors in a different space — existing ChromaDB entries become incompatible and the collection must be wiped and re-ingested
- [x] Implement batch-safe embedding: if `len(texts) > EMBED_BATCH_SIZE` (default 100, configurable), split into batches and call the API once per batch, then flatten results — the OpenAI embeddings endpoint has a max input limit per request
- [x] Wrap each batch call in a retry loop: catch `openai.RateLimitError`, back off with `time.sleep(2 ** attempt)`, retry up to 3 times, then raise `EmbeddingCallError` (add this to `errors.py`) with a safe client message rather than letting the raw OpenAI exception surface through `POST /ingest`
- [x] Catch `openai.APIConnectionError` and `openai.APITimeoutError` separately, applying the same retry/backoff — network blips during a large-document ingest should not fail the whole batch on the first transient error
- [x] Create `backend/services/vector_store_client.py`
- [x] Import `chromadb` and instantiate a `chromadb.PersistentClient(path=settings.vector_store_path)`
- [x] Define the constant `COLLECTION_NAME = "documents"`
- [x] Implement `get_or_create_collection(client: chromadb.Client, embedding_model: str, chunk_size: int, chunk_overlap: int) -> chromadb.Collection` that calls `client.get_or_create_collection(name=COLLECTION_NAME, metadata={"embedding_model": embedding_model, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})`; on first creation this stores the config; on subsequent calls it retrieves the existing collection with its stored metadata
- [x] Implement `validate_collection_config(collection: chromadb.Collection, embedding_model: str, chunk_size: int, chunk_overlap: int)` that reads `collection.metadata` and raises `EmbeddingModelMismatchError` if `metadata["embedding_model"] != embedding_model`; raise `ChunkConfigMismatchError` if chunk parameters differ — neither mismatch should ever silently pass because both would corrupt retrieval quality
- [x] Implement `add_chunks(collection: chromadb.Collection, chunks: list[TextChunk], embeddings: list[list[float]])` that calls `collection.add(documents=[c.text for c in chunks], embeddings=embeddings, ids=[f"{c.source_file}__{c.chunk_index}" for c in chunks], metadatas=[{"source_file": c.source_file, "chunk_index": c.chunk_index, "total_chunks": c.total_chunks} for c in chunks])`; note that ChromaDB IDs must be unique strings — the `source_file__chunk_index` format guarantees uniqueness within a collection
- [x] Implement `source_exists(collection: chromadb.Collection, source_file: str) -> bool` that calls `collection.get(where={"source_file": source_file}, limit=1)` and returns `True` if any results exist — used to enforce re-ingestion protection
- [x] Implement `delete_by_source(collection: chromadb.Collection, source_file: str)` that calls `collection.delete(where={"source_file": source_file})` to remove all chunks belonging to a document
- [x] Implement `search(collection: chromadb.Collection, query_embedding: list[float], top_k: int) -> list[dict]` that calls `collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])` and returns a list of result dicts with `text`, `source_file`, `chunk_index`, and `distance` fields
- [x] Implement `list_sources(collection: chromadb.Collection) -> list[str]` that calls `collection.get(include=["metadatas"])` and returns the deduplicated set of unique `source_file` metadata values
- [x] Create `backend/services/retriever.py`
- [x] Define a dataclass `SourceChunk` with fields: `text: str`, `source_file: str`, `chunk_index: int`, `similarity_score: float` (note: ChromaDB returns L2 distance by default; convert to a similarity score by computing `1 / (1 + distance)` so that lower distance = higher similarity on a 0–1 scale)
- [x] Implement `retrieve(query: str, collection, client: openai.OpenAI, embedding_model: str, top_k: int, min_similarity: float) -> list[SourceChunk]` that: (1) calls `embedder.embed_query`, (2) calls `vector_store_client.search`, (3) converts distances to similarity scores, (4) filters out any chunk with `similarity_score < min_similarity`, (5) returns the filtered list of `SourceChunk` objects sorted by similarity score descending
- [x] Add a grounding check in `retrieve`: if the filtered list is empty (all chunks were below `min_similarity`), return an empty list — **do not raise**; an empty result is handled upstream by the LLM client to produce a grounded "no relevant content found" response
- [x] Call `validate_collection_config` at the start of `retrieve` to confirm the embedding model matches before computing the query embedding — this is the retriever's responsibility, not the caller's
- [x] Create `backend/services/prompt_builder.py`
- [x] Import `tiktoken` and load the encoding for the chat model using `tiktoken.encoding_for_model(settings.chat_model)`
- [x] Define the grounded-only system prompt as a module-level constant: `"You are a helpful assistant. Answer the user's question based ONLY on the context provided below. If the context does not contain enough information to answer the question, respond with exactly: 'I could not find the answer in the uploaded documents.' Do not use prior knowledge. Cite the relevant context items by number, e.g. [1], [2]."`
- [x] Implement `format_context(chunks: list[SourceChunk]) -> str` that formats the chunk list as numbered citations: `"[1] (source: filename.pdf, chunk 3/12)\n{chunk.text}\n\n[2] (source: ...)..."` — this numbered format is what lets the LLM reference sources by index in its answer
- [x] Implement `count_tokens(text: str, encoding) -> int` using `len(encoding.encode(text))`
- [x] Implement `build_prompt(question: str, chunks: list[SourceChunk], max_context_tokens: int) -> list[dict]` that: (1) formats the full context string, (2) counts the combined token usage of system prompt + context + question, (3) if the total exceeds `max_context_tokens`, drops the lowest-similarity chunk and recounts, repeating until the budget is met or only one chunk remains, (4) returns the final message list `[{"role": "system", "content": SYSTEM_PROMPT + "\n\nContext:\n" + formatted_context}, {"role": "user", "content": question}]`
- [x] Note in a code comment that `max_context_tokens` is a safety ceiling, not a target — for `gpt-4o-mini` with a 128k context window, a budget of 8 000 tokens leaves ample room for the output while keeping latency and cost low
- [x] Implement a special case in `build_prompt`: if `chunks` is empty (all chunks filtered by similarity threshold), return messages with the system prompt only and the user question — the LLM will see no context and must reply with the grounded "no relevant content found" response as instructed by the system prompt
- [x] Create `backend/services/llm_client.py`
- [x] Define a dataclass `LLMResponse` with fields: `answer: str`, `prompt_tokens: int`, `completion_tokens: int`
- [x] Implement `generate_answer(messages: list[dict], client: openai.OpenAI, model: str) -> LLMResponse` that calls `client.chat.completions.create(model=model, messages=messages, temperature=0)` with `temperature=0` to maximize answer determinism (important for grounded Q&A — you want consistent, citation-following behavior, not creative variation)
- [x] Extract the answer text from `response.choices[0].message.content`
- [x] Extract token usage from `response.usage.prompt_tokens` and `response.usage.completion_tokens`
- [x] Return an `LLMResponse` with all three fields
- [x] Wrap the API call in a `try/except openai.OpenAIError`; raise `LLMCallError` (defined in `errors.py`) with a safe client message — never expose the raw OpenAI error string in the API response
- [x] Create `backend/services/doc_registry.py`
- [x] Use `sqlite3` (stdlib) only — no SQLAlchemy for this simple append-only registry
- [x] Implement `initialize_registry(db_path: str)` that creates a `documents` table with columns: `id INTEGER PRIMARY KEY AUTOINCREMENT`, `original_filename TEXT`, `sanitized_filename TEXT UNIQUE`, `ingestion_timestamp TEXT`, `chunk_count INTEGER` — `UNIQUE` on `sanitized_filename` enforces the single-document-per-name constraint at the DB level as well as in application logic
- [x] Implement `register_document(db_path: str, original_filename: str, sanitized_filename: str, chunk_count: int)` that inserts a new row with the current ISO-8601 timestamp
- [x] Implement `list_documents(db_path: str) -> list[dict]` that returns all rows as a list of dicts with all five fields
- [x] Implement `get_document_by_id(db_path: str, doc_id: int) -> dict | None` that returns the row with the given ID or `None` if not found
- [x] Implement `delete_document(db_path: str, doc_id: int) -> str | None` that deletes the row and returns the `sanitized_filename` of the deleted document (needed by the route to also delete chunks from ChromaDB), or returns `None` if the ID was not found
- [x] Create `backend/core/config.py` using `pydantic_settings.BaseSettings`
- [x] Add settings fields: `openai_api_key: SecretStr`, `embedding_model: str` (default `"text-embedding-3-small"`), `chat_model: str` (default `"gpt-4o-mini"`), `vector_store_path: str`, `document_store_path: str`, `registry_db_path: str`, `chunk_size: int` (default 800), `chunk_overlap: int` (default 100), `top_k: int` (default 5), `min_similarity: float` (default 0.3), `max_context_tokens: int` (default 8000), `cors_allowed_origins: list[str]`
- [x] Type `openai_api_key` as `SecretStr` — access the underlying string inside services with `settings.openai_api_key.get_secret_value()`; this prevents accidental logging of the key in stack traces or `repr()` output
- [x] Set `model_config = SettingsConfigDict(env_file=".env")` to load values from `.env`
- [x] Expose a module-level `settings = Settings()` singleton
- [x] Create `backend/core/errors.py` with these custom exception classes:
- [x] `UnsupportedFileTypeError` — raised when an uploaded file is not PDF or TXT
- [x] `EmptyDocumentError` — raised when a PDF yields no extractable text (likely a scanned image PDF)
- [x] `DocumentAlreadyIngestedError` — raised when a document with the same sanitized filename already exists in the collection and `force=False`
- [x] `EmbeddingModelMismatchError` — raised when the current config's embedding model differs from the one stored in ChromaDB collection metadata
- [x] `ChunkConfigMismatchError` — raised when chunk_size or chunk_overlap in the current config differs from the values stored in ChromaDB collection metadata
- [x] `LLMCallError` — raised when the OpenAI chat completion call fails
- [x] `DocumentNotFoundError` — raised when a requested document ID does not exist in the registry
- [x] `FileTooLargeError` — raised when an uploaded file exceeds `MAX_UPLOAD_SIZE_MB`
- [x] `EmbeddingCallError` — raised when the OpenAI embeddings call exhausts its retry budget (rate limit or connection failure)
- [x] Create `backend/models/requests.py`
- [x] Define `AskRequest` as a Pydantic `BaseModel` with fields: `question: str` (with `Field(min_length=3, max_length=1000)`) and `top_k: int | None` (optional override for the default `settings.top_k`, with `Field(default=None, ge=1, le=20)`)
- [x] Create `backend/models/responses.py`
- [x] Define `SourceChunkResponse` with fields: `source_file: str`, `chunk_index: int`, `text: str`, `similarity_score: float`
- [x] Define `AskResponse` with fields: `answer: str`, `sources: list[SourceChunkResponse]`, `grounded: bool` (True if at least one chunk was retrieved above the similarity threshold; False if the system answered from empty context with the "I could not find..." response), `prompt_tokens: int`, `completion_tokens: int`
- [x] Define `IngestResponse` with fields: `message: str`, `sanitized_filename: str`, `chunk_count: int`
- [x] Define `DocumentRecord` with fields: `id: int`, `original_filename: str`, `sanitized_filename: str`, `ingestion_timestamp: str`, `chunk_count: int`
- [x] Define `DocumentListResponse` with field: `documents: list[DocumentRecord]`
- [x] Define `HealthResponse` with fields: `status: str`, `vector_store_connected: bool`, `registry_connected: bool`, `embedding_model: str`, `chat_model: str`
- [x] Create `backend/dependencies.py`
- [x] Declare module-level singletons `_openai_client = None`, `_chroma_client = None`, `_collection = None`
- [x] Implement `initialize_clients()` that creates the `openai.OpenAI` client (with `api_key=settings.openai_api_key.get_secret_value()`), creates the `chromadb.PersistentClient`, calls `get_or_create_collection`, calls `validate_collection_config`, and calls `doc_registry.initialize_registry` — to be called once at FastAPI startup
- [x] Implement `get_openai_client()`, `get_chroma_collection()` that return the cached singletons or raise if not initialized
- [x] Create `backend/routes/health.py`
- [x] Define a FastAPI `APIRouter`
- [x] Implement `GET /health` that attempts `get_chroma_collection()` and `doc_registry.list_documents()` to confirm both stores are reachable, sets `vector_store_connected` and `registry_connected` booleans accordingly, and returns `HealthResponse` — if a store is unreachable, still return 200 with the relevant boolean set to `False` rather than raising (health is diagnostic, not a gate)
- [x] Create `backend/routes/ingest.py`
- [x] Define a FastAPI `APIRouter`
- [ ] Implement `POST /ingest` as an async endpoint that accepts `file: UploadFile = File(...)` and `force: bool = Query(default=False)` as parameters
- [x] Add `MAX_UPLOAD_SIZE_MB` (default 25) to `backend/core/config.py`; check `file.size` (or stream and count bytes if `size` is unavailable) before saving, and raise `FileTooLargeError` (HTTP 413) if exceeded — an unbounded upload is both a DoS vector and a way to silently blow the chunking/embedding pipeline's memory on a single request
- [x] Save the uploaded file to `settings.document_store_path / sanitize_filename(file.filename)` using async file I/O (`aiofiles` or `Path.write_bytes`)
- [x] Note that `chunker.chunk_document` and `embedder.embed_texts` are both CPU/network-bound blocking calls; wrap them with `starlette.concurrency.run_in_threadpool` (or make the route `def` instead of `async def`, letting FastAPI run it in its own threadpool) so a large document being processed doesn't block the event loop from serving concurrent `/ask` or `/health` requests
- [x] Call `document_loader.load_document` on the saved file
- [x] Call `vector_store_client.source_exists` to check if this document is already ingested; if `True` and `force=False`, raise `DocumentAlreadyIngestedError` with a message explaining the `force=True` option; if `force=True`, call `vector_store_client.delete_by_source` and `doc_registry.delete_document` before proceeding
- [x] Call `chunker.chunk_document(doc, settings.chunk_size, settings.chunk_overlap)` to split the document
- [x] Call `embedder.embed_texts` on all chunk texts (in batches if needed)
- [x] Call `vector_store_client.add_chunks` to store chunks with embeddings
- [x] Call `doc_registry.register_document` to record the document in the SQLite registry
- [x] Return `IngestResponse` with a success message, sanitized filename, and chunk count
- [x] Handle `UnsupportedFileTypeError` and `EmptyDocumentError` with HTTP 422; handle `DocumentAlreadyIngestedError` with HTTP 409; handle `FileTooLargeError` with HTTP 413; handle `EmbeddingCallError` with HTTP 502; always return only safe client messages, never the raw exception string
- [x] Create `backend/routes/ask.py`
- [x] Define a FastAPI `APIRouter`
- [x] Implement `POST /ask` that accepts `AskRequest` as the JSON request body
- [x] Resolve `effective_top_k = request.top_k or settings.top_k`
- [x] Call `retriever.retrieve(question, collection, openai_client, settings.embedding_model, effective_top_k, settings.min_similarity)` to get the filtered `SourceChunk` list
- [x] Call `prompt_builder.build_prompt(question, chunks, settings.max_context_tokens)` to assemble the LLM messages
- [x] Call `llm_client.generate_answer(messages, openai_client, settings.chat_model)` to get the `LLMResponse`
- [x] Set `grounded = len(chunks) > 0` — if no chunks passed the similarity threshold, the answer was produced from an empty context and should be flagged as ungrounded in the response
- [x] Build and return an `AskResponse` with answer, sources (converted from `SourceChunk` to `SourceChunkResponse`), grounded flag, and token usage
- [x] Handle `EmbeddingModelMismatchError` and `ChunkConfigMismatchError` with HTTP 409 and a message explaining that the vector store must be wiped and re-ingested; handle `LLMCallError` with HTTP 502
- [x] Create `backend/routes/documents.py`
- [x] Define a FastAPI `APIRouter`
- [x] Implement `GET /documents` that calls `doc_registry.list_documents` and returns `DocumentListResponse`
- [x] Implement `DELETE /documents/{doc_id}` that calls `doc_registry.get_document_by_id` to look up the document, raises `DocumentNotFoundError` (HTTP 404) if not found, then calls `vector_store_client.delete_by_source` to remove all chunks from ChromaDB, then calls `doc_registry.delete_document` to remove the registry entry, and returns a 200 with a success message — always delete from both stores or neither; if the ChromaDB deletion fails, do not proceed with the registry deletion
- [x] Create `backend/main.py`
- [x] Instantiate `FastAPI(title="Ragnar", description="RAG-powered document Q&A API", version="1.0.0")`
- [x] Add a `@app.on_event("startup")` handler that calls `dependencies.initialize_clients()` — fail fast at startup if the OpenAI API key is missing or the ChromaDB path is inaccessible
- [x] Add `CORSMiddleware` with `allow_origins=settings.cors_allowed_origins`, `allow_methods=["GET", "POST", "DELETE"]` — never use `["*"]` for origins
- [x] Include routers from `health`, `ingest`, `ask`, and `documents` with appropriate prefixes (e.g., `/api/v1`)
- [x] Add global exception handlers for all custom error classes in `errors.py`, each returning only the sanitized client-facing message and the appropriate HTTP status code
- [x] Test locally with `uvicorn backend.main:app --reload`
- [x] Visit `/docs` to verify all four endpoint groups appear with correct schemas
- [x] Upload a development document via `POST /ingest` using the Swagger UI file upload
- [x] Ask a question about that document via `POST /ask` and verify the answer is grounded in the correct source chunk
- [ ] Create `frontend/app.py`
- [ ] Set page configuration: `st.set_page_config(page_title="Ragnar – Document Q&A", layout="wide")`
- [ ] Initialize session state keys: `chat_history` (list of `{role, content, sources}` dicts), `documents` (list of ingested document names fetched from the API)
- [ ] On startup, call `GET /api/v1/documents` and populate `st.session_state.documents` — if the call fails, show a warning but do not crash
- [ ] Import and call the `uploader`, `chat`, and `source_viewer` components in the appropriate layout regions
- [ ] Create `frontend/components/uploader.py`
- [ ] Render a `st.sidebar` section titled "Uploaded Documents"
- [ ] Use `st.file_uploader(accept_multiple_files=False, type=["pdf", "txt"])` to accept one file at a time
- [ ] On upload, send a `multipart/form-data` POST to `/api/v1/ingest` using `httpx`; show `st.spinner("Ingesting document...")` during the call
- [ ] On success, show `st.success(f"Ingested {chunk_count} chunks from {filename}")` and refresh the document list
- [ ] On `DocumentAlreadyIngestedError` (HTTP 409), show an `st.warning` with a "Force re-ingest?" checkbox; if checked, resend the request with `?force=true`
- [ ] On `EmptyDocumentError` (HTTP 422), show `st.error("This PDF appears to contain scanned images rather than selectable text and cannot be processed.")` — a specific, actionable error message, not a generic "something went wrong"
- [ ] List all currently ingested documents from `st.session_state.documents` with a delete button next to each; on delete, call `DELETE /api/v1/documents/{doc_id}` and refresh the list
- [ ] Create `frontend/components/chat.py`
- [ ] Render the full `st.session_state.chat_history` — user messages aligned right, assistant messages aligned left using `st.chat_message`
- [ ] For each assistant message, if `sources` is non-empty, render a collapsed `st.expander("View sources")` beneath the answer that calls `source_viewer.render_sources(sources)`
- [ ] Use `st.chat_input("Ask a question about your documents...")` for the question input
- [ ] On submit, append the user message to `chat_history`, call `POST /api/v1/ask`, append the assistant response to `chat_history`, and rerender
- [ ] If `grounded=False` in the response, show a `st.info` notice: "This answer was generated without any matching context — the document may not contain information relevant to your question."
- [ ] Create `frontend/components/source_viewer.py`
- [ ] Implement `render_sources(sources: list[dict])` that iterates over the source list and for each chunk renders: the source filename and chunk position (`"filename.pdf — chunk 3 of 12"`), the chunk text in a `st.text_area` (read-only, small height), and the similarity score as a `st.progress` bar scaled to 0–1
- [ ] Create `eval/eval_set.json` with 15–20 hand-written question-answer pairs for the documents in `eval/documents/`; each entry should be a JSON object with fields: `question` (the natural language question), `expected_answer` (the correct answer in one or two sentences), `source_file` (the sanitized filename of the document containing the answer), and `relevant_chunk_text` (a short substring that should appear in the correct retrieval result)
- [ ] Ensure questions vary in difficulty: some answerable from a single chunk, some requiring synthesis across two chunks, and at least two that are deliberately unanswerable from the documents (to test the grounding check)
- [ ] Create `eval/run_eval.py`
- [ ] At the start of `run_eval.py`, ingest all documents in `eval/documents/` via the API (or directly via the service layer), skipping already-ingested ones — the eval harness must work on a clean state as well as an existing one
- [ ] Load `eval/eval_set.json`
- [ ] For each question: (1) call `retriever.retrieve` to get top-5 chunks, (2) check retrieval recall — does any chunk's text contain `relevant_chunk_text` as a substring? — record 1 if yes, 0 if no; for unanswerable questions, check that zero chunks are returned above the similarity threshold and record the grounding check result instead
- [ ] For each question: (3) call `prompt_builder.build_prompt` and `llm_client.generate_answer` to get the generated answer; (4) call the LLM again as a judge with the prompt: `"Question: {question}\nExpected Answer: {expected_answer}\nGenerated Answer: {generated_answer}\nRate the quality of the generated answer on a scale of 1–5 where 5 is perfect. Respond with only the integer score."` — parse the response as an integer
- [ ] Implement `parse_judge_score(raw_response: str) -> int` that strips whitespace and attempts `int(raw_response)`; if parsing fails (the judge occasionally adds a trailing period or a word), fall back to a regex extracting the first standalone digit 1–5; if that also fails, log the raw response and record the score as `None` for that question rather than crashing the whole eval run
- [ ] Clamp any parsed score outside the 1–5 range to the nearest boundary and log a warning — a judge hallucinating "7" should not silently corrupt the mean `answer_quality` statistic
- [ ] Exclude `None` scores from the mean `answer_quality` calculation but report the count of unparseable judge responses separately in the summary, so a spike in judge-parsing failures is visible rather than quietly averaged away
- [ ] Print a results table with one row per question: question text (truncated), retrieval recall (✓/✗), answer quality score (1–5)
- [ ] Print summary statistics: aggregate `retrieval_recall@5` (proportion of questions with correct retrieval), mean `answer_quality`, and `grounding_accuracy` (proportion of unanswerable questions that correctly returned empty context rather than hallucinating)
- [ ] Record the baseline scores from `eval/run_eval.py` in the README once the system is built
- [ ] Create `tests/conftest.py`
- [ ] Create a `mock_openai_client` fixture that returns a `MagicMock` whose `embeddings.create()` returns a mock with `.data = [MagicMock(embedding=[0.1] * 1536)]` for a single input, and whose `chat.completions.create()` returns a mock with `.choices[0].message.content = "Mock answer."` and `.usage.prompt_tokens = 100` and `.usage.completion_tokens = 50` — tests must never call the real OpenAI API
- [ ] Create an `in_memory_chroma` fixture that instantiates `chromadb.EphemeralClient()` (in-memory, no disk persistence) and creates a test collection with the default config — used by vector store and retriever tests so they don't write to the real `vector_store/` directory
- [ ] Create a `sample_chunks` fixture returning a list of three `TextChunk` objects with short text content, `source_file="test_doc.pdf"`, sequential `chunk_index` values, and `total_chunks=3`
- [ ] Create a `test_client` fixture that patches `backend.dependencies._openai_client`, `backend.dependencies._collection`, and `backend.dependencies._registry_db_path` with mocks, and returns a FastAPI `TestClient` — startup lifecycle is not called in tests
- [ ] Create a `tmp_pdf` fixture that writes a minimal valid PDF to a temp file using `pypdf` and yields the path — used in document loader tests to confirm PDF loading works without a real uploaded file
- [ ] Create `tests/test_document_loader.py`
- [ ] Test that `sanitize_filename("My Report (Final).PDF")` returns a lowercase, special-char-free string
- [ ] Test that `sanitize_filename` strips path components so `"../../etc/passwd"` becomes a safe filename with no slashes
- [ ] Test that `load_txt` with a valid UTF-8 file returns a `LoadedDocument` with non-empty `text` and the correct `source_file`
- [ ] Test that `load_document` raises `UnsupportedFileTypeError` for a `.docx` extension
- [ ] Test that `load_pdf` with the `tmp_pdf` fixture returns a `LoadedDocument` with non-empty `text`
- [ ] Create `tests/test_chunker.py`
- [ ] Test that `chunk_document` with a short text (under `chunk_size`) returns exactly one chunk
- [ ] Test that `chunk_document` with a long text returns multiple chunks
- [ ] Test that chunk text lengths are all ≤ `chunk_size + chunk_overlap` (overlap can push a chunk slightly over)
- [ ] Test that `chunk_index` values are sequential starting from 0 and `total_chunks` equals the length of the returned list
- [ ] Test that adjacent chunks share at least some overlapping text when `chunk_overlap > 0`
- [ ] Create `tests/test_embedder.py`
- [ ] Test that `embed_texts([], mock_openai_client, "text-embedding-3-small")` returns an empty list without calling the API
- [ ] Test that `embed_texts(["hello"], mock_openai_client, ...)` returns a list of length 1 containing a 1536-dimensional vector
- [ ] Test that `embed_query("hello", mock_openai_client, ...)` returns a single list of floats (not a list of lists)
- [ ] Test that `embed_texts` with 150 input texts and `EMBED_BATCH_SIZE=100` calls the mocked client exactly twice (batches of 100 and 50) and returns a flattened list of 150 vectors
- [ ] Test that `embed_texts` retries on a mocked `openai.RateLimitError` on the first call and succeeds on the second, without raising
- [ ] Test that `embed_texts` raises `EmbeddingCallError` (not the raw `openai.RateLimitError`) after exhausting all 3 retry attempts
- [ ] Create `tests/test_vector_store_client.py`
- [ ] Test that `get_or_create_collection` on the `in_memory_chroma` fixture creates a collection with the correct metadata
- [ ] Test that `validate_collection_config` raises `EmbeddingModelMismatchError` when the embedding model name does not match the stored metadata
- [ ] Test that `source_exists` returns `False` for an empty collection and `True` after `add_chunks`
- [ ] Test that `delete_by_source` removes all chunks for a given source file but leaves other source files' chunks untouched
- [ ] Test that `add_chunks` with a duplicate ID (same `source_file__chunk_index`) raises or overwrites without crashing — verify ChromaDB's `upsert` vs `add` behavior and document the chosen approach
- [ ] Create `tests/test_retriever.py`
- [ ] Test that `retrieve` with a query that has no matching chunks in the collection returns an empty list (not an error)
- [ ] Test that `retrieve` filters out chunks below `min_similarity` — seed the collection with known embeddings, use a query embedding that is close to one chunk and distant from others, and verify only the close chunk is returned
- [ ] Test that the returned `SourceChunk` list is sorted by `similarity_score` descending
- [ ] Test that `retrieve` raises `EmbeddingModelMismatchError` if the collection was created with a different embedding model (mock the collection metadata)
- [ ] Create `tests/test_prompt_builder.py`
- [ ] Test that `build_prompt` with an empty chunk list returns a message list with the grounded-only system prompt and the user question, but no context section
- [ ] Test that `build_prompt` with three chunks formats them as `[1]`, `[2]`, `[3]` in the system message
- [ ] Test that `build_prompt` drops the lowest-similarity chunk when the token count exceeds `max_context_tokens` — use a very small `max_context_tokens` value to force truncation
- [ ] Test that the returned message list has exactly two messages: `{"role": "system", ...}` and `{"role": "user", ...}`
- [ ] Create `tests/test_llm_client.py`
- [ ] Test that `generate_answer` with `mock_openai_client` returns an `LLMResponse` with `answer="Mock answer."`, `prompt_tokens=100`, `completion_tokens=50`
- [ ] Test that `generate_answer` calls `client.chat.completions.create` with `temperature=0` explicitly passed (assert on the mock's call args) — a regression here would silently reintroduce non-deterministic answers
- [ ] Test that `generate_answer` wraps a mocked `openai.OpenAIError` into `LLMCallError` rather than letting it propagate
- [ ] Create `tests/test_doc_registry.py`
- [ ] Use a temp SQLite file path (`tmp_path` pytest fixture) for every test in this file — never touch the real `registry_db_path`
- [ ] Test that `initialize_registry` creates the `documents` table and it's idempotent (calling it twice doesn't raise or duplicate the table)
- [ ] Test that `register_document` followed by `list_documents` returns exactly one record with the correct `sanitized_filename` and `chunk_count`
- [ ] Test that inserting two documents with the same `sanitized_filename` raises (verifies the `UNIQUE` constraint is actually enforced, not just documented)
- [ ] Test that `get_document_by_id` returns `None` for a non-existent ID rather than raising
- [ ] Test that `delete_document` returns the `sanitized_filename` on success and `None` when the ID doesn't exist
- [ ] Create `tests/test_api.py`
- [ ] Test `GET /health` returns 200 with `status: "ok"` (using mocked dependencies)
- [ ] Test `POST /ask` with a mocked retriever returning two source chunks returns 200 with `grounded=True`, a non-empty answer, and two entries in `sources`
- [ ] Test `POST /ask` with a mocked retriever returning an empty list returns 200 with `grounded=False` and the grounding fallback message in `answer`
- [ ] Test `POST /ask` with `question` shorter than 3 characters returns 422
- [ ] Test `POST /ingest` with a non-PDF, non-TXT file (e.g., `.docx`) returns 422
- [ ] Test `POST /ingest` when `source_exists` returns `True` and `force=False` returns 409
- [ ] Test `GET /documents` returns 200 and a list (may be empty) with the correct response shape
- [ ] Test `DELETE /documents/{doc_id}` with a non-existent ID returns 404
- [ ] Run `pytest tests/ -v` and confirm all tests pass before building the frontend
- [ ] Create `Makefile`
- [ ] Add `install` target: `pip install -r requirements.txt`
- [ ] Add `run-api` target: `uvicorn backend.main:app --reload`
- [ ] Add `run-ui` target: `streamlit run frontend/app.py`
- [ ] Add `test` target: `pytest tests/ -v`
- [ ] Add `eval` target: `python eval/run_eval.py`
- [ ] Add `wipe-vector-store` target: `rm -rf vector_store/* && touch vector_store/.gitkeep` — required before changing embedding model or chunk parameters; include a comment in the Makefile warning that this permanently deletes all ingested chunk embeddings and requires re-ingesting all documents
- [ ] Add `ingest-eval-docs` target that loops over files in `eval/documents/` and calls `POST /ingest` for each using `curl` — useful for resetting the eval environment
- [ ] Create `README.md`
- [ ] Write project title (`Ragnar: RAG-Powered Document Q&A`) and a one-line description
- [ ] List prerequisites: Python 3.11+, an OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini`
- [ ] Provide setup instructions: clone repo, create venv, `make install`, copy `.env.example` to `.env`, add OpenAI API key
- [ ] Provide local run instructions: `make run-api` → `make run-ui` → upload a document → ask a question
- [ ] Include an architecture diagram as a plain-text ASCII diagram:
  ```
  POST /ingest (PDF or TXT)
        │
        ▼
  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
  │ Document Loader  │──▶│    Chunker        │──▶│  Embedder           │
  │ (pypdf / text)   │   │ (RecursiveChar    │   │ (text-embedding-    │
  └──────────────────┘   │  TextSplitter)    │   │  3-small)           │
                         └──────────────────┘   └──────────┬──────────┘
                                                            │
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │  ChromaDB            │
                                                 │  (PersistentClient)  │
                                                 └──────────────────────┘

  POST /ask (question)
        │
        ▼
  ┌──────────────┐   ┌──────────────────────┐   ┌──────────────────┐
  │   Embedder   │──▶│   Retriever          │──▶│  Prompt Builder  │
  │ (query embed)│   │ (top-k + threshold)  │   │ (grounded system │
  └──────────────┘   └──────────────────────┘   │  prompt + budget)│
                                                 └────────┬─────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │   LLM Client         │
                                               │   (gpt-4o-mini,      │
                                               │    temperature=0)    │
                                               └──────────────────────┘
                                                          │
                                               answer + sources + grounded flag
  ```
- [ ] Document the embedding model consistency rule: once you ingest documents, do not change `EMBEDDING_MODEL` in `.env` without running `make wipe-vector-store` and re-ingesting all documents
- [ ] Document the similarity threshold: `MIN_SIMILARITY=0.3` is the default; lower values return more (potentially less relevant) chunks; higher values are stricter and may increase the rate of "no relevant content found" responses — tune this by inspecting similarity scores in `/ask` responses on your specific document set
- [ ] Record the eval baseline scores (`retrieval_recall@5`, mean `answer_quality`, `grounding_accuracy`) once the system is built
- [ ] Add a screenshot of the Streamlit chat interface with a sample question, answer, and expanded source viewer
- [ ] Note in the README that scanned/image PDFs are not supported — the document loader detects empty extraction and returns a clear error; users must use OCR-processed PDFs with selectable text
- [ ] (Optional) Add support for `.md` and `.html` file types in `document_loader.py`: strip HTML tags with `BeautifulSoup` for HTML files; load Markdown as plain text
- [ ] (Optional) Implement streaming answers in `llm_client.py` using `stream=True` in the OpenAI call, and surface the token stream via a FastAPI `StreamingResponse` (`text/event-stream`) so the Streamlit frontend can display tokens as they arrive using `st.write_stream`
- [ ] (Optional) Add multi-document filtering to `POST /ask`: accept an optional `source_files: list[str]` parameter in `AskRequest` and pass a ChromaDB `where` filter to restrict retrieval to only those documents
- [ ] (Optional) Implement Maximal Marginal Relevance (MMR) re-ranking in `retriever.py`: after the initial top-k retrieval, apply MMR to reduce redundancy among returned chunks while maintaining relevance — improves answer quality on long, repetitive documents
- [ ] (Optional) Replace the OpenAI embedding model with a local `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`) controlled by a `EMBEDDING_BACKEND=openai|local` config flag — eliminates per-embedding API costs and makes the system fully offline-capable; requires wiping and re-ingesting the vector store when switching backends
- [ ] (Optional) Add a `GET /ask/history` endpoint that logs each question, answer, grounded flag, and token usage to the SQLite registry so past Q&A sessions are reviewable without rebuilding context
- [ ] (Optional) Deploy the FastAPI backend to Render, Railway, or Fly.io; persist the `vector_store/` directory on a mounted volume; persist `documents.db` on the same volume; set `OPENAI_API_KEY` as an environment secret
- [ ] (Optional) Set up a `.github/workflows/ci.yml` GitHub Actions workflow that runs `pytest tests/ -v` on every push to `main` — safe because all tests mock the OpenAI client and use in-memory ChromaDB, so CI runs without any API key or disk state
