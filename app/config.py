from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    
    # Pinecone
    pinecone_api_key: str
    pinecone_index: str
    
    # Cohere
    cohere_api_key: str
        
    # LangSmith
    langsmith_api_key: str
    langsmith_tracing: str = "true"
    langsmith_project: str = "document-qna-rag"
    
    # RAG settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"

# Even if get_settings gets called multiple times, it returns the cached result as opposed to re-reading the file
@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()