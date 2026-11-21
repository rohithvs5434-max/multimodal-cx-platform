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


class ImprovedEmotionCNN(nn.Module):
    """
    Experiment 3:
    Deeper CNN with residual blocks.

    Input:
        [batch, 1, mel_bins, time]

    Output:
        emotion logits
    """

    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.block1 = self._block(32, 32)
        self.block2 = self._block(32, 64)
        self.block3 = self._block(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def _block(self, in_channels, out_channels):

        if in_channels != out_channels:
            shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            shortcut = nn.Identity()

        return nn.ModuleDict({
            "conv1": nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            "bn1": nn.BatchNorm2d(out_channels),
            "conv2": nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            "bn2": nn.BatchNorm2d(out_channels),
            "shortcut": shortcut,
            "relu": nn.ReLU(),
        })

    def _forward_block(self, x, block):

        identity = block["shortcut"](x)

        x = block["conv1"](x)
        x = block["bn1"](x)
        x = block["relu"](x)

        x = block["conv2"](x)
        x = block["bn2"](x)

        x = x + identity
        x = block["relu"](x)

        return x

    def forward(self, x):

        x = self.stem(x)

        x = self._forward_block(
            x,
            self.block1,
        )
        x = self.pool(x)

        x = self._forward_block(
            x,
            self.block2,
        )
        x = self.pool(x)

        x = self._forward_block(
            x,
            self.block3,
        )

        x = self.global_pool(x)

        return self.classifier(x)


class ManifestDataset(Dataset):

    def __init__(self, rows, labels):

        self.rows = rows
        self.labels = labels

        self.to_index = {
            label: i
            for i, label in enumerate(labels)
        }

        self.features = [
            log_mel(row["path"])
            for row in rows
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.to_index[
                self.rows[index]["emotion"]
            ],
        )


def evaluate(
    model,
    loader,
    device,
    labels,
):

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
            "models/emotion_cnn_improved.pt"
        ),
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cpu"

    print("=" * 70)
    print("EXPERIMENT 3 — IMPROVED CNN ARCHITECTURE")
    print("=" * 70)

    print(f"Device:      {device}")
    print(f"Seed:        {args.seed}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print()

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
        shuffle=False,
    )

    test_loader = DataLoader(
        ManifestDataset(
            test_rows,
            labels,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ImprovedEmotionCNN(
        len(labels)
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = 0
    best_state = None

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        losses = []

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

            losses.append(
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

        print(
            f"Epoch {epoch:02d} | "
            f"loss={np.mean(losses):.4f} | "
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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "labels": labels,
            "state_dict": best_state,
            "experiment":
                "improved_cnn_architecture",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
            "parameter_count": parameter_count,
        },
        args.output,
    )

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
            "improved_cnn_architecture",

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

        "parameter_count":
            parameter_count,

        "validation_weighted_f1":
            best_f1,

        "test_weighted_f1":
            test_f1,

        "classification_report":
            report,
    }

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("EXPERIMENT 3 RESULTS")
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
