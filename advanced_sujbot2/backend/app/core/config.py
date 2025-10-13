"""Application configuration using Pydantic settings."""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API
    API_TITLE: str = "SUJBOT2 API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 4

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080"
    ]

    # Claude API
    CLAUDE_API_KEY: str | None = None  # Optional for testing without API key
    MAIN_AGENT_MODEL: str = "claude-sonnet-4-5-20250929"
    SUBAGENT_MODEL: str = "claude-3-5-haiku-20241022"
    MAX_PARALLEL_AGENTS: int = 10

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # PostgreSQL Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "sujbot2"
    POSTGRES_USER: str = "sujbot_app"
    POSTGRES_PASSWORD: str = ""

    @field_validator('POSTGRES_PASSWORD')
    @classmethod
    def validate_postgres_password(cls, v: str) -> str:
        """Validate that PostgreSQL password is set."""
        if not v or v.strip() == "":
            raise ValueError(
                "POSTGRES_PASSWORD must be set. "
                "Set it in environment variables or .env file. "
                "Never use empty or default passwords in production."
            )
        if v == "change_this_password" or v == "password" or v == "postgres":
            raise ValueError(
                f"POSTGRES_PASSWORD cannot be a common weak password like '{v}'. "
                "Use a strong, unique password."
            )
        return v

    # File storage
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md", ".odt", ".rtf"]

    # Indexing
    INDEX_DIR: str = "indexes"

    # Authentication (future)
    SECRET_KEY: str = "change-me-in-production-use-openssl-rand-hex-32"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Logging
    VERBOSE_LOGGING: bool = False

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
