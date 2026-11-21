from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    redis_url: str = "redis://localhost:6379/0"
    redis_stream_key: str = "cx:events"
    event_retention: int = 1000

    # Selected ASR model from Phase-2 comparison.
    whisper_model: str = "base"

    # Selected acoustic emotion model from Phase-2 comparison.
    emotion_model_path: Path = Path("models/emotion_cnn.pt")

    # Selected linguistic sentiment model from Phase-2 comparison.
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment"

    model_cache_dir: Path = Path("models/cache")
    max_upload_mb: int = 25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
