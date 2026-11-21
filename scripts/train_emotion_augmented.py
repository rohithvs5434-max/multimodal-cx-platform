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


class AugmentedManifestDataset(Dataset):
    def __init__(self, rows, labels, augment=False):
        self.rows = rows
        self.labels = labels
        self.to_index = {label: i for i, label in enumerate(labels)}
        self.augment = augment

        # Precompute the original features.
        self.features = [
            log_mel(row["path"])
            for row in rows
        ]

    def __len__(self):
        return len(self.rows)

    def augment_feature(self, feature):
        """
        Conservative training-only augmentation on log-Mel features.

        feature shape: [1, n_mels, time]
        """

        x = feature.clone()

        # --------------------------------------------------
        # 1. Small time shift
        # --------------------------------------------------
        if random.random() < 0.5:
            max_shift = max(1, x.shape[-1] // 20)
            shift = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=shift, dims=-1)

        # --------------------------------------------------
        # 2. Random gain
        # --------------------------------------------------
        if random.random() < 0.5:
            gain = random.uniform(0.90, 1.10)
            x = x * gain

        # --------------------------------------------------
        # 3. Low-level Gaussian noise
        # --------------------------------------------------
        if random.random() < 0.5:
            noise_std = random.uniform(0.005, 0.02)
            noise = torch.randn_like(x) * noise_std
            x = x + noise

        return x

    def __getitem__(self, index):
        feature = self.features[index]

        if self.augment:
            feature = self.augment_feature(feature)

        label = self.to_index[
            self.rows[index]["emotion"]
        ]

        return feature, label


def evaluate(model, loader, device, labels):
    model.eval()

    targets = []
    predictions = []

    with torch.inference_mode():
        for features, target in loader:
            output = model(
                features.to(device)
            )

            predictions.extend(
                output.argmax(dim=1).cpu().tolist()
            )

            targets.extend(
                target.tolist()
            )

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

    return (
        weighted_f1,
        report,
        targets,
        predictions,
    )


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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/emotion_cnn_augmented.pt"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cpu"

    print("=" * 70)
    print("EXPERIMENT 2 — AUDIO AUGMENTATION")
    print("=" * 70)

    print(f"Device:      {device}")
    print(f"Seed:        {args.seed}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print()

    # ------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------

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

    print(f"Labels:      {labels}")
    print(f"Train:       {len(train_rows)}")
    print(f"Validation:  {len(validation_rows)}")
    print(f"Test:        {len(test_rows)}")
    print()

    # ------------------------------------------------------
    # IMPORTANT:
    # Only TRAIN receives augmentation.
    # Validation/test remain untouched.
    # ------------------------------------------------------

    train_dataset = AugmentedManifestDataset(
        train_rows,
        labels,
        augment=True,
    )

    validation_dataset = AugmentedManifestDataset(
        validation_rows,
        labels,
        augment=False,
    )

    test_dataset = AugmentedManifestDataset(
        test_rows,
        labels,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = EmotionCNN(
        len(labels)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    # Same loss as baseline.
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = 0
    best_state = None

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        args.epochs + 1
    ):

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
                float(
                    loss.detach()
                )
            )

        validation_f1, _, _, _ = evaluate(
            model,
            validation_loader,
            device,
            labels,
        )

        mean_loss = np.mean(
            epoch_losses
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={mean_loss:.4f} | "
            f"validation_weighted_f1="
            f"{validation_f1:.4f}"
        )

        if validation_f1 > best_f1:

            best_f1 = validation_f1
            best_epoch = epoch

            best_state = {
                name:
                value.detach()
                .cpu()
                .clone()

                for name, value
                in model.state_dict().items()
            }

    assert best_state is not None

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "labels": labels,
            "state_dict": best_state,
            "experiment":
                "audio_augmentation",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
            "augmentations": [
                "small_time_shift",
                "random_gain",
                "gaussian_noise",
            ],
        },
        args.output,
    )

    # ------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------

    model.load_state_dict(
        best_state
    )

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
        "experiment":
            "audio_augmentation",

        "device":
            device,

        "seed":
            args.seed,

        "epochs":
            args.epochs,

        "batch_size":
            args.batch_size,

        "best_epoch":
            best_epoch,

        "validation_weighted_f1":
            best_f1,

        "test_weighted_f1":
            test_f1,

        "augmentations": [
            "small_time_shift",
            "random_gain",
            "gaussian_noise",
        ],

        "classification_report":
            report,
    }

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 70)

    print(
        f"Best epoch:              "
        f"{best_epoch}"
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
        f"Saved model:   "
        f"{args.output}"
    )

    print(
        f"Saved metrics: "
        f"{metrics_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
