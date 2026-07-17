from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://tokenopt:tokenopt@localhost:5432/tokenopt"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "conversation_memory"
    redis_url: str = "redis://localhost:6379/0"

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    reranker_model: str = "BAAI/bge-reranker-base"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_mock: bool = False

    similarity_threshold: float = 0.72
    retrieval_top_k: int = 3
    retrieval_candidates: int = 20
    context_token_budget: int = 4096
    short_term_max_tokens: int = 1200
    summarize_threshold: int = 3000
    summarize_turn_threshold: int = 10
    short_term_max_messages: int = 8

    system_prompt_tokens: int = 300
    user_facts_tokens: int = 150
    session_summary_tokens: int = 400
    retrieved_memories_tokens: int = 600

    temporal_decay_lambda: float = 0.01
    mmr_lambda: float = 0.5
    dedup_similarity_threshold: float = 0.95

    api_port: int = 9200
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
