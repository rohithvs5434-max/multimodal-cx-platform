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
    def __init__(
        self,
        rows: list[dict[str, str]],
        labels: list[str],
    ) -> None:
        self.rows = rows
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
            self.feature_cache[index] = log_mel(
                row["path"]
            )

        return (
            self.feature_cache[index],
            self.to_index[row["emotion"]],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: EmotionCNN,
    loader: DataLoader,
    device: str,
) -> tuple[float, float, float]:

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

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    return (
        accuracy,
        weighted_f1,
        macro_f1,
    )


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
        "--output",
        type=Path,
        default=Path(
            "models/emotion_cnn_repeated_seeds.metrics.json"
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

    print("Experiment 6: Repeated-seed 64-Mel CNN")
    print("-----------------------------------------")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Train: {len(train_rows)}")
    print(f"Validation: {len(validation_rows)}")
    print(f"Test: {len(test_rows)}")
    print()

    results = []

    for seed in SEEDS:

        print("=" * 55)
        print(f"SEED {seed}")
        print("=" * 55)

        set_seed(seed)

        train_dataset = ManifestDataset(
            train_rows,
            labels,
        )

        validation_dataset = ManifestDataset(
            validation_rows,
            labels,
        )

        test_dataset = ManifestDataset(
            test_rows,
            labels,
        )

        # Explicit generator makes the shuffle order
        # reproducible for this seed.
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        model = EmotionCNN(
            len(labels)
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )

        loss_fn = nn.CrossEntropyLoss()

        best_validation_f1 = -1.0
        best_state = None
        best_epoch = None

        for epoch in range(
            1,
            args.epochs + 1,
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

            if validation_weighted_f1 > best_validation_f1:

                best_validation_f1 = (
                    validation_weighted_f1
                )

                best_epoch = epoch

                best_state = {
                    name: value.cpu().clone()
                    for name, value
                    in model.state_dict().items()
                }

            print(
                f"epoch={epoch:02d} "
                f"validation_weighted_f1="
                f"{validation_weighted_f1:.4f}"
            )

        assert best_state is not None

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

        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_weighted_f1": (
                best_validation_f1
            ),
            "test_accuracy": test_accuracy,
            "test_weighted_f1": test_weighted_f1,
            "test_macro_f1": test_macro_f1,
        }

        results.append(result)

        print()
        print(f"Seed {seed} result:")
        print(
            f"  Best epoch: {best_epoch}"
        )
        print(
            f"  Validation weighted F1: "
            f"{best_validation_f1:.4f}"
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
        print()

    # Aggregate results
    accuracies = np.array(
        [
            result["test_accuracy"]
            for result in results
        ],
        dtype=float,
    )

    weighted_f1s = np.array(
        [
            result["test_weighted_f1"]
            for result in results
        ],
        dtype=float,
    )

    macro_f1s = np.array(
        [
            result["test_macro_f1"]
            for result in results
        ],
        dtype=float,
    )

    def summarize(values: np.ndarray) -> dict:
        mean = float(values.mean())
        std = float(values.std(ddof=1))

        # t-based 95% CI for n=5.
        # t_(0.975,4) = 2.776445.
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

    summary = {
        "accuracy": summarize(accuracies),
        "weighted_f1": summarize(weighted_f1s),
        "macro_f1": summarize(macro_f1s),
    }

    output = {
        "experiment": (
            "Repeated-seed 64-Mel compact CNN"
        ),
        "seeds": SEEDS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "test_samples": len(test_rows),
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
    print("FINAL REPEATED-SEED SUMMARY")
    print("=" * 65)

    print(
        f"Accuracy: "
        f"{summary['accuracy']['mean']:.4f} "
        f"± {summary['accuracy']['std']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{summary['weighted_f1']['mean']:.4f} "
        f"± {summary['weighted_f1']['std']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{summary['macro_f1']['mean']:.4f} "
        f"± {summary['macro_f1']['std']:.4f}"
    )

    print()
    print(
        f"95% CI weighted F1: "
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
