from pathlib import Path

import librosa
import numpy as np
import torch


SAMPLE_RATE = 16_000
N_MELS = 128
MAX_SECONDS = 4.0


def log_mel(path: str | Path) -> torch.Tensor:
    """Load audio consistently for both training and serving."""
    samples, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    target = int(SAMPLE_RATE * MAX_SECONDS)
    samples = np.pad(samples, (0, max(0, target - len(samples))))[:target]
    mel = librosa.feature.melspectrogram(y=samples, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel_spec = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return torch.from_numpy(log_mel_spec).unsqueeze(0)
