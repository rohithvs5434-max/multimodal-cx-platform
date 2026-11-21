from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from emotion_model.model import EmotionCNN


SAMPLE_RATE = 16_000
N_MELS = 128
MAX_SECONDS = 4.0


def log_mel_128(path: str | Path) -> torch.Tensor:
    """Same preprocessing as production audio.py, but with 128 Mel bins."""

    samples, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=MAX_SECONDS,
    )

    target = int(SAMPLE_RATE * MAX_SECONDS)

    samples = np.pad(
        samples,
        (0, max(0, target - len(samples))),
    )[:target]

    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
    )

    log_mel_spec = librosa.power_to_db(
        mel,
        ref=np.max,
    ).astype(np.float32)

    return torch.from_numpy(log_mel_spec).unsqueeze(0)


class ManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        labels: list[str],
    ) -> None:
        self.rows = rows
        self.labels = labels
        self.to_index = {
            label: index
            for index, label in enumerate(labels)
        }
        self.feature_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:

        row = self.rows[index]

        if index not in self.feature_cache:
            self.feature_cache[index] = log_mel_128(
                row["path"]
            )

        return (
            self.feature_cache[index],
            self.to_index[row["emotion"]],
        )


def evaluate(
    model: EmotionCNN,
    loader: DataLoader,
    device: str,
    labels: list[str],
) -> tuple[float, dict, np.ndarray]:

    model.eval()

    targets = []
    predictions = []

    with torch.inference_mode():
        for features, target in loader:
            output = model(
                features.to(device)
            )

            predictions.extend(
                output.argmax(dim=1)
                .cpu()
                .tolist()
            )

            targets.extend(
                target.tolist()
            )

    if not targets:
        return 0.0, {}, np.empty((0, 0), dtype=int)

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(labels))),
    )

    return weighted_f1, report, matrix


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/manifest.csv"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/emotion_cnn_128mel.pt"
        ),
    )

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Keep training on CPU so this experiment is
    # comparable with the frozen baseline training.
    device = "cpu"

    # Load manifest
    with args.manifest.open() as file:
        rows = list(csv.DictReader(file))

    labels = sorted(
        {
            row["emotion"]
            for row in rows
        }
    )

    train_rows = [
        row for row in rows
        if row["split"] == "train"
    ]

    validation_rows = [
        row for row in rows
        if row["split"] == "validation"
    ]

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    if not all(
        (
            train_rows,
            validation_rows,
            test_rows,
        )
    ):
        raise SystemExit(
            "Manifest needs non-empty "
            "train, validation, and test splits."
        )

    print("Experiment 4: 128-Mel representation")
    print("---------------------------------------")
    print(f"Device: {device}")
    print(f"Mel bins: {N_MELS}")
    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Max seconds: {MAX_SECONDS}")
    print(f"Train samples: {len(train_rows)}")
    print(f"Validation samples: {len(validation_rows)}")
    print(f"Test samples: {len(test_rows)}")
    print()

    # Verify feature shape before training
    sample_feature = log_mel_128(
        train_rows[0]["path"]
    )

    print(
        f"Feature shape: {tuple(sample_feature.shape)}"
    )
    print(
        f"Feature dtype: {sample_feature.dtype}"
    )
    print()

    train_loader = DataLoader(
        ManifestDataset(
            train_rows,
            labels,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        ManifestDataset(
            validation_rows,
            labels,
        ),
        batch_size=args.batch_size,
    )

    test_loader = DataLoader(
        ManifestDataset(
            test_rows,
            labels,
        ),
        batch_size=args.batch_size,
    )

    # SAME compact CNN architecture as baseline.
    model = EmotionCNN(
        len(labels)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        epoch_loss = 0.0
        batches = 0

        for features, target in train_loader:

            optimizer.zero_grad()

            output = model(
                features.to(device)
            )

            loss = loss_fn(
                output,
                target.to(device),
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        validation_f1, _, _ = evaluate(
            model,
            validation_loader,
            device,
            labels,
        )

        mean_loss = (
            epoch_loss / batches
            if batches
            else 0.0
        )

        print(
            f"epoch={epoch:02d} "
            f"loss={mean_loss:.4f} "
            f"validation_weighted_f1="
            f"{validation_f1:.4f}"
        )

        if validation_f1 > best_f1:

            best_f1 = validation_f1
            best_epoch = epoch

            best_state = {
                name: value.cpu().clone()
                for name, value
                in model.state_dict().items()
            }

    assert best_state is not None

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "labels": labels,
            "n_mels": N_MELS,
            "sample_rate": SAMPLE_RATE,
            "max_seconds": MAX_SECONDS,
            "state_dict": best_state,
        },
        args.output,
    )

    # Load best validation checkpoint
    model.load_state_dict(
        best_state
    )

    test_f1, report, matrix = evaluate(
        model,
        test_loader,
        device,
        labels,
    )

    test_accuracy = report["accuracy"]
    macro_f1 = report["macro avg"]["f1-score"]

    metrics = {
        "experiment": "128-mel representation",
        "device": device,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "max_seconds": MAX_SECONDS,
        "feature_shape": list(
            sample_feature.shape
        ),
        "best_epoch": best_epoch,
        "validation_weighted_f1": best_f1,
        "test_accuracy": test_accuracy,
        "test_weighted_f1": test_f1,
        "test_macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }

    metrics_path = (
        args.output.with_suffix(
            ".metrics.json"
        )
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()
    print("RESULT")
    print("------")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Validation weighted F1: "
        f"{best_f1:.4f}"
    )
    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )
    print(
        f"Test weighted F1: "
        f"{test_f1:.4f}"
    )
    print(
        f"Test macro F1: "
        f"{macro_f1:.4f}"
    )
    print()
    print(
        f"Model saved: {args.output}"
    )
    print(
        f"Metrics saved: {metrics_path}"
    )
    print()
    print("Confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
