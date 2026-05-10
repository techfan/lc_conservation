from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "智能问答系统"
    VERSION: str = "1.0.0"
    
    REDIS_URL: str = "redis://admin:Admin@123@localhost:6379/0"
    REDIS_TTL: int = 7 * 24 * 60 * 60
    
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "rag_documents"
    
    POSTGRES_URL: str = "postgresql+asyncpg://admin:Admin%40123@localhost:5432/ai-conservation"
    
    ZHIPU_API_KEY: str = "60c3230638b343869bfba9e6ceb744da.FfQPmLl505bMJnrm"
    LLM_MODEL: str = "glm-4.7-flash"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    
    MAX_CONTEXT_TOKENS: int = 8000
    MAX_HISTORY_LENGTH: int = 20
    COMPRESSION_THRESHOLD: int = 10
    
    class Config:
        env_file = ".env"


settings = Settings()
