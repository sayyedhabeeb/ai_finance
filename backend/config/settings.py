from functools import lru_cache
import logging

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Keep explicit module constants for existing imports.
PATCHTST_MODEL_PATH: str = "models/patchtst"
_DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/ai_financial_brain"
_DEFAULT_WEAVIATE_URL = "http://localhost:8080"

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = Field(
        default="AI Financial Brain",
        validation_alias=AliasChoices("APP_NAME", "app_name"),
    )
    app_version: str = Field(
        default="1.0.0",
        validation_alias=AliasChoices("APP_VERSION", "app_version"),
    )
    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("DEBUG", "debug", "APP_ENV"),
    )
    environment: str = Field(
        default="production",
        validation_alias=AliasChoices("ENVIRONMENT", "APP_ENV", "environment"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )

    api_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("API_HOST", "api_host"),
    )
    api_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("API_PORT", "BACKEND_PORT", "api_port"),
    )
    api_workers: int = Field(
        default=4,
        validation_alias=AliasChoices("API_WORKERS", "api_workers"),
    )

    database_url: str = Field(
        default=_DEFAULT_DATABASE_URL,
        validation_alias=AliasChoices("DATABASE_URL", "database_url"),
    )
    redis_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("REDIS_HOST", "redis_host"),
    )
    redis_port: int = Field(
        default=6379,
        validation_alias=AliasChoices("REDIS_PORT", "redis_port"),
    )
    redis_password: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REDIS_PASSWORD", "redis_password"),
    )
    redis_url: str = Field(
        default="",
        validation_alias=AliasChoices("REDIS_URL", "redis_url"),
    )
    weaviate_url: str = Field(
        default=_DEFAULT_WEAVIATE_URL,
        validation_alias=AliasChoices("WEAVIATE_URL", "weaviate_url"),
    )

    llm_provider: str = Field(
        default="groq",
        validation_alias=AliasChoices("LLM_PROVIDER", "llm_provider"),
    )
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "groq_api_key"),
    )
    groq_model: str = Field(
        default="llama3-70b-8192",
        validation_alias=AliasChoices("GROQ_MODEL", "groq_model"),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("OPENAI_MODEL", "openai_model"),
    )

    jwt_secret_key: str = Field(
        default="CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR",
        validation_alias=AliasChoices("JWT_SECRET_KEY", "jwt_secret_key"),
    )
    jwt_algorithm: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALGORITHM", "jwt_algorithm"),
    )

    critic_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("CRITIC_ENABLED", "critic_enabled"),
    )
    critic_max_revisions: int = Field(
        default=3,
        validation_alias=AliasChoices("CRITIC_MAX_REVISIONS", "critic_max_revisions"),
    )
    min_quality_score: float = Field(
        default=0.75,
        validation_alias=AliasChoices("MIN_QUALITY_SCORE", "min_quality_score"),
    )

    patchtst_model_path: str = Field(
        default=PATCHTST_MODEL_PATH,
        validation_alias=AliasChoices("PATCHTST_MODEL_PATH", "patchtst_model_path"),
    )

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            return normalized in {"1", "true", "yes", "on", "debug", "development"}
        return bool(value)

    @model_validator(mode="after")
    def _normalize_urls(self) -> "Settings":
        if not self.redis_url:
            if self.redis_password:
                self.redis_url = f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
            else:
                self.redis_url = f"redis://{self.redis_host}:{self.redis_port}/0"

        if not self.weaviate_url:
            self.weaviate_url = _DEFAULT_WEAVIATE_URL

        return self

    @property
    def DATABASE_URL(self) -> str:
        return self.database_url

    @property
    def REDIS_HOST(self) -> str:
        return self.redis_host

    @property
    def REDIS_PORT(self) -> int:
        return self.redis_port

    @property
    def REDIS_PASSWORD(self) -> str | None:
        return self.redis_password

    @property
    def GROQ_API_KEY(self) -> str | None:
        return self.groq_api_key

    @property
    def WEAVIATE_URL(self) -> str:
        return self.weaviate_url


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    if not settings.database_url or settings.database_url == _DEFAULT_DATABASE_URL:
        logger.warning("DATABASE_URL is missing or defaulted; database connection may be non-production.")
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set.")
    return settings
