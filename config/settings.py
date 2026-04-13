from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    persist_dir: str = "./chroma_db"
    collection_name: str = "default"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    llm_model_path: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

settings = Settings()
