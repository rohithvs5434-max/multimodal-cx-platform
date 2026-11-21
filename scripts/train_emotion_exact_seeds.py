from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from emotion_model.audio import log_mel
from emotion_model.model import EmotionCNN


SEEDS = [42, 43, 44, 45, 46]


class ManifestDataset(Dataset):
    """
    Exact dataset behavior from scripts/train_emotion.py:
    pre-compute all log-Mel features before training.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        labels: list[str],
    ) -> None:

        self.rows = rows

        self.label_to_id = {
            label: i
            for i, label in enumerate(labels)
        }

        self.features = []

        for row in rows:

            path = Path(row["path"])

            feature = log_mel(path)

            target = self.label_to_id[
                row["emotion"]
            ]

            self.features.append(
                (feature, target)
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():

        for features, target in loader:

            features = features.to(device)
            target = target.to(device)

            output = model(features)

            prediction = output.argmax(
                dim=1
            )

            y_true.extend(
                target.cpu().tolist()
            )

            y_pred.extend(
                prediction.cpu().tolist()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return (
        accuracy,
        weighted_f1,
        macro_f1,
    )


def train_one_seed(
    seed: int,
    train_rows,
    validation_rows,
    test_rows,
    labels,
    epochs: int,
    batch_size: int,
):

    print()
    print("=" * 60)
    print(f"SEED {seed}")
    print("=" * 60)

    # EXACT ordering from original script:
    # seed BEFORE dataset precomputation.
    set_seed(seed)

    print("Preparing training dataset...")

    train_dataset = ManifestDataset(
        train_rows,
        labels,
    )

    print("Preparing validation dataset...")

    validation_dataset = ManifestDataset(
        validation_rows,
        labels,
    )

    print("Preparing test dataset...")

    test_dataset = ManifestDataset(
        test_rows,
        labels,
    )

    print()

    # EXACT original DataLoader construction.
    # No explicit generator.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # EXACT original model creation.
    model = EmotionCNN(
        classes=len(labels)
    ).to("cpu")

    device = "cpu"

    # EXACT original optimizer.
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
        epochs + 1,
    ):

        model.train()

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

        (
            validation_accuracy,
            validation_weighted_f1,
            validation_macro_f1,
        ) = evaluate(
            model,
            validation_loader,
            device,
        )

        print(
            f"epoch={epoch:02d} "
            f"validation_weighted_f1="
            f"{validation_weighted_f1:.4f}"
        )

        # EXACT original best-checkpoint criterion.
        if validation_weighted_f1 > best_f1:

            best_f1 = (
                validation_weighted_f1
            )

            best_epoch = epoch

            best_state = {
                name: value.cpu().clone()
                for name, value
                in model.state_dict().items()
            }

    assert best_state is not None

    # Evaluate exactly the selected validation checkpoint.
    model.load_state_dict(
        best_state
    )

    (
        test_accuracy,
        test_weighted_f1,
        test_macro_f1,
    ) = evaluate(
        model,
        test_loader,
        device,
    )

    print()
    print(f"Seed {seed} result:")
    print(
        f"  Best epoch: {best_epoch}"
    )
    print(
        f"  Validation weighted F1: "
        f"{best_f1:.4f}"
    )
    print(
        f"  Test accuracy: "
        f"{test_accuracy:.4f}"
    )
    print(
        f"  Test weighted F1: "
        f"{test_weighted_f1:.4f}"
    )
    print(
        f"  Test macro F1: "
        f"{test_macro_f1:.4f}"
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_weighted_f1": best_f1,
        "test_accuracy": test_accuracy,
        "test_weighted_f1": test_weighted_f1,
        "test_macro_f1": test_macro_f1,
    }


def summarize(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    mean = float(
        values.mean()
    )

    std = float(
        values.std(ddof=1)
    )

    # Student-t critical value:
    # t(0.975, df=4) = 2.776445
    margin = (
        2.776445
        * std
        / np.sqrt(len(values))
    )

    return {
        "mean": mean,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def main():

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
        "--output",
        type=Path,
        default=Path(
            "models/"
            "emotion_cnn_exact_seeds.metrics.json"
        ),
    )

    args = parser.parse_args()

    device = "cpu"

    with args.manifest.open() as file:

        rows = list(
            csv.DictReader(file)
        )

    labels = sorted(
        {
            row["emotion"]
            for row in rows
        }
    )

    train_rows = [
        row
        for row in rows
        if row["split"] == "train"
    ]

    validation_rows = [
        row
        for row in rows
        if row["split"] == "validation"
    ]

    test_rows = [
        row
        for row in rows
        if row["split"] == "test"
    ]

    print(
        "Experiment 6A: Exact-protocol "
        "repeated seeds"
    )

    print(
        "---------------------------------------"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Train: {len(train_rows)}"
    )

    print(
        f"Validation: {len(validation_rows)}"
    )

    print(
        f"Test: {len(test_rows)}"
    )

    results = []

    for seed in SEEDS:

        result = train_one_seed(
            seed,
            train_rows,
            validation_rows,
            test_rows,
            labels,
            args.epochs,
            args.batch_size,
        )

        results.append(result)

    accuracies = [
        result["test_accuracy"]
        for result in results
    ]

    weighted_f1s = [
        result["test_weighted_f1"]
        for result in results
    ]

    macro_f1s = [
        result["test_macro_f1"]
        for result in results
    ]

    summary = {
        "accuracy": summarize(
            accuracies
        ),
        "weighted_f1": summarize(
            weighted_f1s
        ),
        "macro_f1": summarize(
            macro_f1s
        ),
    }

    output = {
        "experiment": (
            "Exact-protocol repeated "
            "64-Mel CNN seeds"
        ),
        "seeds": SEEDS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device,
        "train_samples": len(train_rows),
        "validation_samples": len(
            validation_rows
        ),
        "test_samples": len(
            test_rows
        ),
        "per_seed_results": results,
        "summary": summary,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("=" * 65)
    print(
        "EXACT-PROTOCOL REPEATED-SEED SUMMARY"
    )
    print("=" * 65)

    print(
        f"Accuracy: "
        f"{summary['accuracy']['mean']:.4f} "
        f"± "
        f"{summary['accuracy']['std']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{summary['weighted_f1']['mean']:.4f} "
        f"± "
        f"{summary['weighted_f1']['std']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{summary['macro_f1']['mean']:.4f} "
        f"± "
        f"{summary['macro_f1']['std']:.4f}"
    )

    print()

    print(
        "95% CI weighted F1: "
        f"["
        f"{summary['weighted_f1']['ci95_lower']:.4f}, "
        f"{summary['weighted_f1']['ci95_upper']:.4f}"
        f"]"
    )

    print()

    print(
        f"Metrics saved: {args.output}"
    )


if __name__ == "__main__":
    main()
