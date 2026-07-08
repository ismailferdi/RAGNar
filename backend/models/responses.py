from pydantic import BaseModel, Field


class SourceChunkResponse(BaseModel):
    source_file: str
    chunk_index: int
    text: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunkResponse] = []
    grounded: bool
    prompt_tokens: int
    completion_tokens: int


class IngestResponse(BaseModel):
    message: str
    sanitized_filename: str
    chunk_count: int


class DocumentRecord(BaseModel):
    id: int
    original_filename: str
    sanitized_filename: str
    ingestion_timestamp: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]


class HealthResponse(BaseModel):
    status: str
    vector_store_connected: bool
    registry_connected: bool
    embedding_model: str
    chat_model: str
