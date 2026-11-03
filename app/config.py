from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    embedding_model: str = "models/text-embedding-004"
    embedding_dimensions: int = 768
    llm_model: str = "gemini-2.0-flash"
    chunk_size: int = 512
    chunk_overlap: int = 50
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    postgres_url: str = "postgresql+asyncpg://rag:rag_secret@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"


settings = Settings()
