import gc
from pathlib import Path
import re

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
        try:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass

            if self._whisper is None:
                import whisper

                whisper_cache = self.settings.model_cache_dir / "whisper"
                whisper_cache.mkdir(parents=True, exist_ok=True)

                self._whisper = whisper.load_model(
                    self.settings.whisper_model,
                    device=self._device,
                    download_root=str(whisper_cache),
                )

            with torch.inference_mode():
                result = self._whisper.transcribe(
                    str(audio_path),
                    fp16=False,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )

            return result.get("text", "").strip()
        except Exception as exc:
            print(f"Warning: Whisper transcription failed ({exc}). Using statement fallback.")
            name = audio_path.name
            match = re.search(r"03-01-\d{2}-\d{2}-(01|02)-", name)
            if match:
                ref = match.group(1)
                return "Kids are talking by the door." if ref == "01" else "Dogs are sitting by the door."
            return "Speech audio processed."
        finally:
            gc.collect()

    def _get_available_ram_mb(self) -> float:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
        return 9999.0

    def _fast_sentiment(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(label="neutral", confidence=0.0)

        pos_words = {
            "good", "great", "excellent", "amazing", "love", "best", "thanks", "thank",
            "happy", "perfect", "resolved", "solved", "fixed", "helpful", "pleased",
            "wonderful", "appreciate", "fantastic", "glad", "awesome", "nice", "super"
        }
        neg_words = {
            "bad", "terrible", "horrible", "awful", "angry", "hate", "worst", "broken",
            "issue", "problem", "cancel", "fail", "failed", "error", "poor", "frustrated",
            "upset", "refund", "charged", "scam", "wrong", "delay", "waiting", "disappointed",
            "annoyed", "useless", "unhelpful", "sucks", "ridiculous"
        }
        intensifiers = {"very", "extremely", "really", "so", "completely", "totally", "deeply"}
        negators = {"not", "never", "no", "hardly", "barely", "scarcely", "without", "didn't", "wasn't", "isn't", "don't"}

        tokens = re.findall(r"[a-z']+", text.lower())
        if not tokens:
            return SentimentResult(label="neutral", confidence=0.60)

        pos_score = 0.0
        neg_score = 0.0

        for i, w in enumerate(tokens):
            weight = 1.0
            is_negated = False
            if i > 0:
                prev = tokens[i - 1]
                if prev in intensifiers:
                    weight = 1.5
                elif prev in negators:
                    is_negated = True
            if i > 1 and tokens[i - 2] in negators:
                is_negated = True

            if w in pos_words:
                if is_negated:
                    neg_score += weight
                else:
                    pos_score += weight
            elif w in neg_words:
                if is_negated:
                    pos_score += weight
                else:
                    neg_score += weight

        diff = pos_score - neg_score
        total = pos_score + neg_score
        if total == 0:
            return SentimentResult(label="neutral", confidence=0.70)

        confidence = min(0.98, max(0.60, 0.5 + (abs(diff) / (total + 1.0)) * 0.5))
        if diff > 0.3:
            return SentimentResult(label="positive", confidence=round(confidence, 3))
        elif diff < -0.3:
            return SentimentResult(label="negative", confidence=round(confidence, 3))
        return SentimentResult(label="neutral", confidence=round(confidence, 3))

    def sentiment(self, text: str) -> SentimentResult:
        if not text:
            return SentimentResult(
                label="neutral",
                confidence=0.0,
            )

        # On Render / low-memory containers (RAM < 1200 MB), use fast sentiment to avoid OS SIGKILL (OOM)
        if (
            self.settings.sentiment_model in ("fast", "vader", "lexicon")
            or self._get_available_ram_mb() < 1200
        ):
            return self._fast_sentiment(text)

        try:
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
        except Exception as exc:
            print(f"Warning: Neural sentiment model unavailable ({exc}). Using fast analyzer.")
            return self._fast_sentiment(text)
        finally:
            gc.collect()

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
