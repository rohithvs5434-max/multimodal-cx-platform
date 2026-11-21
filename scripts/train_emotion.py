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
        self.label_to_id = {
            label: i for i, label in enumerate(labels)
        }

        # Pre-compute log-Mel features.
        # log_mel() expects the AUDIO FILE PATH.
        self.features = []

        for row in rows:
            path = Path(row["path"])

            feature = log_mel(path)

            target = self.label_to_id[row["emotion"]]

            self.features.append(
                (feature, target)
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)

            output = model(features)

            prediction = output.argmax(dim=1)

            y_true.extend(
                target.cpu().tolist()
            )

            y_pred.extend(
                prediction.cpu().tolist()
            )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return weighted_f1, (y_true, y_pred)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/emotion_cnn.pt"
        )
    )

    args = parser.parse_args()

    # =========================================================
    # REPRODUCIBILITY
    # =========================================================

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # =========================================================
    # DEVICE
    #
    # IMPORTANT:
    # EmotionCNN contains AdaptiveAvgPool2d((4,4)).
    #
    # The current MPS backend fails for the feature-map shape
    # produced by this architecture.
    #
    # Therefore the emotion CNN is trained on CPU.
    # =========================================================

    device = "cpu"

    print()
    print("=" * 60)
    print("EMOTION CNN TRAINING")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")

    print("=" * 60)
    print()

    # =========================================================
    # LOAD MANIFEST
    # =========================================================

    with args.manifest.open() as f:
        rows = list(
            csv.DictReader(f)
        )

    labels = sorted(
        {
            row["emotion"]
            for row in rows
        }
    )

    print(
        f"Total samples: {len(rows)}"
    )

    print(
        f"Labels: {labels}"
    )

    # =========================================================
    # SPLITS
    # =========================================================

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
        f"Training samples: {len(train_rows)}"
    )

    print(
        f"Validation samples: {len(validation_rows)}"
    )

    print(
        f"Test samples: {len(test_rows)}"
    )

    print()

    # =========================================================
    # DATASETS
    # =========================================================

    print(
        "Preparing training dataset..."
    )

    train_dataset = ManifestDataset(
        train_rows,
        labels
    )

    print(
        "Preparing validation dataset..."
    )

    validation_dataset = ManifestDataset(
        validation_rows,
        labels
    )

    print(
        "Preparing test dataset..."
    )

    test_dataset = ManifestDataset(
        test_rows,
        labels
    )

    print()

    # =========================================================
    # DATALOADERS
    # =========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # =========================================================
    # MODEL
    # =========================================================

    model = EmotionCNN(
        classes=len(labels)
    ).to(device)

    # =========================================================
    # OPTIMIZER
    # =========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    # =========================================================
    # LOSS
    # =========================================================

    loss_fn = nn.CrossEntropyLoss()

    # =========================================================
    # BEST MODEL
    # =========================================================

    best_f1 = -1.0
    best_state = None
    best_epoch = 0

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        running_loss = 0.0
        batches = 0

        for features, target in train_loader:

            features = features.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(features)

            loss = loss_fn(
                output,
                target
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            batches += 1

        train_loss = (
            running_loss / batches
        )

        # =====================================================
        # VALIDATION
        # =====================================================

        validation_f1, _ = evaluate(
            model,
            validation_loader,
            device
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Weighted F1: "
            f"{validation_f1:.4f}"
        )

        # =====================================================
        # BEST CHECKPOINT
        # =====================================================

        if validation_f1 > best_f1:

            best_f1 = validation_f1
            best_epoch = epoch

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    # =========================================================
    # RESTORE BEST MODEL
    # =========================================================

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    print()
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation weighted F1: "
        f"{best_f1:.4f}"
    )

    print("=" * 60)
    print()

    # =========================================================
    # TEST
    # =========================================================

    test_f1, (y_true, y_pred) = evaluate(
        model,
        test_loader,
        device
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(labels))
        ),
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    # =========================================================
    # SAVE MODEL
    # =========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "labels": labels,
            "state_dict": best_state
        },
        args.output
    )

    # =========================================================
    # SAVE METRICS
    # =========================================================

    metrics_output = (
        args.output.with_suffix(
            ".metrics.json"
        )
    )

    metrics = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation_weighted_f1": best_f1,
        "test_weighted_f1": test_f1,
        "classification_report": report
    }

    with metrics_output.open("w") as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    # =========================================================
    # FINAL RESULTS
    # =========================================================

    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Test Weighted F1: "
        f"{test_f1:.4f}"
    )

    print()
    print("Classification Report:")
    print()

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(
                range(len(labels))
            ),
            target_names=labels,
            zero_division=0
        )
    )

    print("=" * 60)

    print(
        f"Model saved: {args.output}"
    )

    print(
        f"Metrics saved: {metrics_output}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
