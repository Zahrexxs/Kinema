from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    """Runtime configuration, overridable with ``KINEMA_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="KINEMA_", env_file=".env", extra="ignore")

    model_complexity: int = 1
    upload_dir: Path = DATA_DIR / "uploads"
    output_dir: Path = DATA_DIR / "outputs"
    max_upload_mb: float = 200.0
    job_ttl_hours: float = 24.0
    visibility_threshold: float = 0.5
    log_level: str = "INFO"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
