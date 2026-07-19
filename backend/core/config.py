from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from pathlib import Path

ENV_PATH = Path(__file__).parent.parent.parent / ".env"
MAX_UPLOAD_SIZE_MB: int = 25


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
    )

    openrouter_api_key: SecretStr
    base_url: str
    embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
    embed_batch_size: int
    chat_model: str = "openai/gpt-oss-20b"
    judge_model: str = "meta/llama-3.1-8b-instruct"
    tiktoken_model: str = "gpt-oss-20b"
    tiktoken_encoder: str = "o200k_harmony"
    vector_store_path: str
    document_store_path: str
    registry_db_path: str
    chunk_size: int = 3000
    chunk_overlap: int = 375
    max_context_tokens: int = 8000
    top_k: int = 5
    min_similarity: float = 0.30
    grounding_threshold: float = 0.55
    core_allowed_origins: list[str]
    allowed_methods: list[str]
    ragnar_api_url: str


settings = Settings()
