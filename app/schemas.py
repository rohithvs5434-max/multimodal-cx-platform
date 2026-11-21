from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class EmotionResult(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    status: Literal["ready", "unavailable"]


class SentimentResult(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class FusionResult(BaseModel):
    state: Literal["positive", "mixed", "negative"]
    score: float = Field(ge=-1, le=1)
    sentiment_score: float = Field(ge=-1, le=1)
    emotion_score: float = Field(ge=-1, le=1)


class Attention(BaseModel):
    severity: Literal["medium", "high", "critical"]
    reason: str
    reason_label: str
    summary: str
    actions: list[str]


class FollowUp(BaseModel):
    previous_event_id: str
    previous_score: float
    new_score: float
    outcome: Literal["improved", "unchanged", "worse"]


class CXEvent(BaseModel):
    event_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["audio", "simulated"]
    transcript: str
    emotion: EmotionResult
    sentiment: SentimentResult
    fusion: FusionResult | None = None
    alert: str | None = None
    attention: Attention | None = None
    followup: FollowUp | None = None
    customer_id: str | None = None
    handled: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    redis: bool
    emotion_model_ready: bool


class DemoSeedResponse(BaseModel):
    events: list[CXEvent]
