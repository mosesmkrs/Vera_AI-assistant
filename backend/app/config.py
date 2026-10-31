from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://vera:vera@localhost:5433/vera"
    llm_api_key: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-6-luna"
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_smtp_username: str = ""
    email_smtp_password: str = ""
    email_smtp_use_ssl: bool = False
    email_from_name: str = "Vera"
    email_from: str = ""
    email_reply_to: str = ""
    jwt_secret: str = "development-only-change-me"
    embedding_model_name: str = "intfloat/multilingual-e5-base"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    evidence_confidence_threshold: float = 0.55
    answer_confidence_medium_threshold: float = 0.65
    answer_confidence_high_threshold: float = 0.85

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

