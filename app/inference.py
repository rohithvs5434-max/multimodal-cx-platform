from pathlib import Path

import torch

from app.config import Settings
from app.fusion import fuse
from app.schemas import EmotionResult, FusionResult, SentimentResult
from emotion_model.audio import log_mel
from emotion_model.model import EmotionCNN


class MultimodalAnalyzer:
    """Lazy model loading prevents a slow or network-dependent API startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self._emotion_model: EmotionCNN | None = None
        self._labels: list[str] = []
        self._whisper = None
        self._sentiment = None

        # General device: used by Whisper when MPS is available.
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Emotion CNN runs on CPU because AdaptiveAvgPool2d
        # has an MPS limitation for non-divisible input sizes.
        self._emotion_device = torch.device("cpu")

    @property
    def emotion_model_ready(self) -> bool:
        return self.settings.emotion_model_path.is_file()

    @property
    def device(self) -> str:
        return str(self._device)

    def _load_emotion_model(self) -> None:
        if self._emotion_model is not None:
            return

        checkpoint = torch.load(
            self.settings.emotion_model_path,
            map_location=self._emotion_device,
            weights_only=False,
        )

        self._labels = checkpoint["labels"]

        model = EmotionCNN(classes=len(self._labels))
        model.load_state_dict(checkpoint["state_dict"])

        # Emotion CNN runs on CPU.
        model.to(self._emotion_device).eval()

        self._emotion_model = model

    def emotion(self, audio_path: Path) -> EmotionResult:
        if not self.emotion_model_ready:
            return EmotionResult(
                label="train_model_first",
                confidence=0,
                status="unavailable",
            )

        self._load_emotion_model()

        assert self._emotion_model is not None

        # Keep emotion features on CPU to avoid the MPS
        # AdaptiveAvgPool2d limitation.
        features = (
            log_mel(audio_path)
            .unsqueeze(0)
            .to(self._emotion_device)
        )

        with torch.inference_mode():
            probabilities = torch.softmax(
                self._emotion_model(features),
                dim=1,
            )[0]

        score, index = torch.max(probabilities, dim=0)

        return EmotionResult(
            label=self._labels[index.item()],
            confidence=float(score),
            status="ready",
        )

    def transcript(self, audio_path: Path) -> str:
        if self._whisper is None:
            import whisper

            whisper_cache = self.settings.model_cache_dir / "whisper"
            whisper_cache.mkdir(parents=True, exist_ok=True)

            self._whisper = whisper.load_model(
                self.settings.whisper_model,
                device=self._device,
                download_root=str(whisper_cache),
            )

        result = self._whisper.transcribe(
            str(audio_path),
            fp16=False,
        )

        return result["text"].strip()

    def sentiment(self, text: str) -> SentimentResult:
        if not text:
            return SentimentResult(
                label="neutral",
                confidence=0,
            )

        if self._sentiment is None:
            from transformers import pipeline

            transformer_cache = self.settings.model_cache_dir / "huggingface"
            transformer_cache.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._sentiment = pipeline(
                "sentiment-analysis",
                model=self.settings.sentiment_model,
                cache_dir=str(transformer_cache),
            )

        result = self._sentiment(text[:512])[0]

        label = result["label"].lower().replace("_", " ")

        label_map = {
            "label 0": "negative",
            "label 1": "neutral",
            "label 2": "positive",
        }

        label = label_map.get(label, label)

        return SentimentResult(
            label=label,
            confidence=float(result["score"]),
        )

    def analyze(
        self,
        audio_path: Path,
    ) -> tuple[str, EmotionResult, SentimentResult, FusionResult]:
        transcript = self.transcript(audio_path)
        emotion = self.emotion(audio_path)
        sentiment = self.sentiment(transcript)
        fused = fuse(
            emotion.label,
            emotion.confidence,
            sentiment.label,
            sentiment.confidence,
        )
        fusion = FusionResult(
            state=fused.state,
            score=fused.score,
            sentiment_score=fused.sentiment_score,
            emotion_score=fused.emotion_score,
        )

        return transcript, emotion, sentiment, fusion
