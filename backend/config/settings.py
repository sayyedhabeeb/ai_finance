from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover
    from pydantic import BaseModel

    class BaseSettings(BaseModel):
        pass
from pydantic import field_validator

# Keep explicit module constants for existing imports.
PATCHTST_MODEL_PATH: str = "models/patchtst"


class Settings(BaseSettings):
    app_name: str = "AI Financial Brain"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    llm_provider: str = "groq"
    groq_api_key: str = ""

    critic_enabled: bool = True
    critic_max_revisions: int = 3
    min_quality_score: float = 0.75

    patchtst_model_path: str = PATCHTST_MODEL_PATH

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {"1", "true", "yes", "on", "debug", "development"}
        return bool(value)

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
