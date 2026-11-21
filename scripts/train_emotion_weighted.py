import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from emotion_model.audio import log_mel
from emotion_model.model import EmotionCNN


class ManifestDataset(Dataset):
    def __init__(self, rows, labels):
        self.rows = rows
        self.labels = labels
        self.to_index = {label: index for index, label in enumerate(labels)}
        self.features = [log_mel(row["path"]) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return (
            self.features[index],
            self.to_index[self.rows[index]["emotion"]],
        )


def evaluate(model, loader, device, labels):
    model.eval()

    targets = []
    predictions = []

    with torch.inference_mode():
        for features, target in loader:
            output = model(features.to(device))

            predictions.extend(
                output.argmax(dim=1).cpu().tolist()
            )
            targets.extend(target.tolist())

    if not targets:
        return 0.0, {}, targets, predictions

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

    return weighted_f1, report, targets, predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
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
        default=Path("models/emotion_cnn_weighted.pt"),
    )

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use CPU to match the verified baseline evaluation environment.
    device = "cpu"

    print("=" * 70)
    print("EXPERIMENT 1 — CLASS-WEIGHTED CROSS-ENTROPY")
    print("=" * 70)
    print(f"Device:      {device}")
    print(f"Seed:        {args.seed}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print()

    with args.manifest.open() as file:
        rows = list(csv.DictReader(file))

    labels = sorted({row["emotion"] for row in rows})

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

    if not all((train_rows, validation_rows, test_rows)):
        raise SystemExit(
            "Manifest needs non-empty train, validation, and test splits."
        )

    print(f"Labels:      {labels}")
    print(f"Train:       {len(train_rows)}")
    print(f"Validation:  {len(validation_rows)}")
    print(f"Test:        {len(test_rows)}")
    print()

    # ---------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------
    train_counts = {
        label: sum(
            row["emotion"] == label
            for row in train_rows
        )
        for label in labels
    }

    total_train = len(train_rows)

    # Inverse-frequency weighting:
    # weight_i = total_samples / (number_of_classes * class_count_i)
    class_weights = torch.tensor(
        [
            total_train / (
                len(labels) * train_counts[label]
            )
            for label in labels
        ],
        dtype=torch.float32,
    )

    print("Training class counts:")
    for label in labels:
        print(
            f"  {label:10s}: "
            f"{train_counts[label]:4d} "
            f"weight={class_weights[labels.index(label)]:.4f}"
        )

    print()

    train_loader = DataLoader(
        ManifestDataset(train_rows, labels),
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        ManifestDataset(validation_rows, labels),
        batch_size=args.batch_size,
    )

    test_loader = DataLoader(
        ManifestDataset(test_rows, labels),
        batch_size=args.batch_size,
    )

    model = EmotionCNN(len(labels)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    # ONLY intentional change from baseline:
    # class-weighted CrossEntropyLoss.
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):

        model.train()

        epoch_losses = []

        for features, target in train_loader:

            optimizer.zero_grad()

            output = model(
                features.to(device)
            )

            loss = loss_fn(
                output,
                target.to(device)
            )

            loss.backward()
            optimizer.step()

            epoch_losses.append(
                float(loss.detach())
            )

        validation_f1, _, _, _ = evaluate(
            model,
            validation_loader,
            device,
            labels,
        )

        mean_loss = np.mean(epoch_losses)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={mean_loss:.4f} | "
            f"validation_weighted_f1={validation_f1:.4f}"
        )

        if validation_f1 > best_f1:

            best_f1 = validation_f1
            best_epoch = epoch

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    assert best_state is not None

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "labels": labels,
            "state_dict": best_state,
            "experiment": "class_weighted_cross_entropy",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
            "class_weights": class_weights.tolist(),
        },
        args.output,
    )

    # Load best checkpoint
    model.load_state_dict(best_state)

    test_f1, report, targets, predictions = evaluate(
        model,
        test_loader,
        device,
        labels,
    )

    metrics_path = args.output.with_suffix(
        ".metrics.json"
    )

    metrics = {
        "experiment": "class_weighted_cross_entropy",
        "device": device,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "validation_weighted_f1": best_f1,
        "test_weighted_f1": test_f1,
        "classification_report": report,
        "class_weights": {
            label: float(
                class_weights[i]
            )
            for i, label in enumerate(labels)
        },
    }

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("EXPERIMENT 1 RESULTS")
    print("=" * 70)

    print(
        f"Best epoch:              {best_epoch}"
    )

    print(
        f"Validation weighted F1:  "
        f"{best_f1:.4f}"
    )

    print(
        f"Test weighted F1:        "
        f"{test_f1:.4f}"
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            targets,
            predictions,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    print(
        f"Saved model:   {args.output}"
    )

    print(
        f"Saved metrics: {metrics_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
