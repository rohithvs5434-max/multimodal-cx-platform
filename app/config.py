from pathlib import Path
import re

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    redis_url: str = "redis://localhost:6379/0"
    redis_stream_key: str = "cx:events"
    event_retention: int = 1000

    # Selected ASR model: 'tiny' for fast inference and 512MB RAM compatibility.
    whisper_model: str = "tiny"

    # Selected acoustic emotion model from Phase-2 comparison.
    emotion_model_path: Path = Path("models/emotion_cnn.pt")

    # Selected linguistic sentiment model: 'fast' for instant 512MB RAM execution without OOM.
    sentiment_model: str = "fast"

    model_cache_dir: Path = Path("models/cache")
    max_upload_mb: int = 25

    @field_validator("redis_url", mode="before")
    @classmethod
    def clean_redis_url(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        match = re.search(r"(rediss?://[^\s\"\'\)]+)", v)
        if match:
            return match.group(1)
        return v.strip().strip("\"'")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
