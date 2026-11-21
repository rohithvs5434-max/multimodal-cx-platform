import re
from dataclasses import dataclass
from pathlib import Path


RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
FILENAME = re.compile(r"^03-01-(?P<emotion>\d{2})-\d{2}-\d{2}-\d{2}-(?P<speaker>\d{2})\.wav$")


@dataclass(frozen=True)
class RavdessRecord:
    path: Path
    emotion: str
    speaker: int


def parse_ravdess(path: str | Path) -> RavdessRecord:
    path = Path(path)
    match = FILENAME.match(path.name)
    if not match:
        raise ValueError(f"Not a RAVDESS speech filename: {path.name}")
    code = match.group("emotion")
    if code not in RAVDESS_EMOTIONS:
        raise ValueError(f"Unsupported RAVDESS emotion code: {code}")
    return RavdessRecord(path=path, emotion=RAVDESS_EMOTIONS[code], speaker=int(match.group("speaker")))


def speaker_split(speaker: int) -> str:
    """Deterministic, gender-balanced speaker-independent 16/4/4 split."""
    validation_speakers = {3, 6, 12, 15}
    test_speakers = {9, 18, 21, 24}
    if speaker in validation_speakers:
        return "validation"
    if speaker in test_speakers:
        return "test"
    return "train"
