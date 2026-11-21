"""Deterministic output-level multimodal fusion.

The project currently has no supervised customer-service dataset with labels
for learning a fusion classifier. Therefore this module implements an
explicit, reproducible decision-level (late) fusion rule.
"""

from dataclasses import dataclass


# Interpretable acoustic valence mapping.
# These are not claims about universal human emotion; they are an
# operational mapping used to combine model outputs consistently.
EMOTION_VALENCE = {
    "happy": 1.0,
    "calm": 0.20,
    "neutral": 0.0,
    "surprised": 0.25,
    "sad": -1.0,
    "angry": -1.0,
    "fearful": -0.80,
    "disgust": -1.0,
}

DEFAULT_ALPHA = 0.5
POSITIVE_THRESHOLD = 0.25
NEGATIVE_THRESHOLD = -0.25


@dataclass(frozen=True)
class FusionResult:
    score: float
    state: str
    sentiment_score: float
    emotion_score: float


def sentiment_to_score(label: str, confidence: float) -> float:
    """Convert sentiment output to a signed confidence score."""
    normalized = label.lower().replace("_", " ").strip()
    if normalized.startswith("positive"):
        return float(confidence)
    if normalized.startswith("negative"):
        return -float(confidence)
    return 0.0


def emotion_to_score(label: str, confidence: float) -> float:
    """Convert emotion output to a signed valence contribution."""
    valence = EMOTION_VALENCE.get(label.lower().strip(), 0.0)
    return float(valence * confidence)


def fuse(
    emotion_label: str,
    emotion_confidence: float,
    sentiment_label: str,
    sentiment_confidence: float,
    alpha: float = DEFAULT_ALPHA,
) -> FusionResult:
    """Combine acoustic emotion and linguistic sentiment at output level.

    F = alpha*S + (1-alpha)*E

    alpha=0.5 gives equal contribution to sentiment and emotion.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    sentiment_score = sentiment_to_score(sentiment_label, sentiment_confidence)
    emotion_score = emotion_to_score(emotion_label, emotion_confidence)
    score = alpha * sentiment_score + (1.0 - alpha) * emotion_score

    if score >= POSITIVE_THRESHOLD:
        state = "positive"
    elif score <= NEGATIVE_THRESHOLD:
        state = "negative"
    else:
        state = "mixed"

    return FusionResult(
        score=round(float(score), 6),
        state=state,
        sentiment_score=round(float(sentiment_score), 6),
        emotion_score=round(float(emotion_score), 6),
    )
