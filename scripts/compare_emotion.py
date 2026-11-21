"""Train and compare Wav2Vec2 and WavLM emotion classifiers on the same RAVDESS split.

Both candidates use the project's deterministic speaker-independent split and the
same 8 RAVDESS emotion labels. This script intentionally trains the classification
head/backbone on the project's data rather than importing reported scores from a
third-party checkpoint, so the comparison is project-specific.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
MODELS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "wavlm": "microsoft/wavlm-base-plus",
}
SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path, limit_train: int, limit_val: int, limit_test: int) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups = {"train": [], "validation": [], "test": []}
    for row in rows:
        if row["split"] in groups:
            groups[row["split"]].append(row)
    for split, limit in (("train", limit_train), ("validation", limit_val), ("test", limit_test)):
        if limit:
            groups[split] = groups[split][:limit]
    missing = [row["path"] for rows_ in groups.values() for row in rows_ if not Path(row["path"]).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} audio files referenced by the manifest are missing. "
            "Regenerate data/processed/manifest.csv on the machine that contains RAVDESS."
        )
    return groups


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target = int(SAMPLE_RATE * MAX_SECONDS)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return audio.astype(np.float32)


class RavdessAudio(Dataset):
    def __init__(self, rows: list[dict[str, str]], label_to_id: dict[str, int]):
        self.rows = rows
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        row = self.rows[index]
        return load_audio(row["path"]), self.label_to_id[row["emotion"]]


def make_collate(extractor):
    def collate(batch):
        audio = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        inputs = extractor(audio, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
        inputs["labels"] = labels
        return inputs
    return collate


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate(model, loader, device) -> tuple[float, float, dict]:
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            y_true.extend(batch["labels"].detach().cpu().tolist())
            y_pred.extend(outputs.logits.argmax(dim=-1).detach().cpu().tolist())
    accuracy = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=list(range(len(LABELS))), zero_division=0)
    report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    output_dict=True,
    zero_division=0,
)
    return accuracy, macro, report


def freeze_feature_extractor(model) -> None:
    for name in ("wav2vec2", "wavlm"):
        base = getattr(model, name, None)
        if base is not None and hasattr(base, "feature_extractor"):
            for parameter in base.feature_extractor.parameters():
                parameter.requires_grad = False
            return


def train_one(name: str, model_id: str, groups, args, device):
    print(f"\n=== {name}: {model_id} ===")
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=len(LABELS),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_extractor:
        freeze_feature_extractor(model)
    model.to(device)

    train_loader = DataLoader(
        RavdessAudio(groups["train"], label_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(extractor),
        num_workers=0,
    )
    val_loader = DataLoader(
        RavdessAudio(groups["validation"], label_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(extractor),
        num_workers=0,
    )
    test_loader = DataLoader(
        RavdessAudio(groups["test"], label_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(extractor),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_val = -1.0
    best_state = None
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accumulation
            loss.backward()
            if step % args.grad_accumulation == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += outputs.loss.item()
        val_acc, val_macro, _ = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d}/{args.epochs} loss={running_loss/max(1,len(train_loader)):.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro:.4f}")
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    training_time = time.perf_counter() - training_start
    assert best_state is not None
    model.load_state_dict(best_state)
    test_start = time.perf_counter()
    test_acc, test_macro, report = evaluate(model, test_loader, device)
    test_time = time.perf_counter() - test_start

    output_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{name}.pt"
    torch.save({"model_id": model_id, "labels": LABELS, "state_dict": best_state}, checkpoint)

    return {
        "model_id": model_id,
        "train_samples": len(groups["train"]),
        "validation_samples": len(groups["validation"]),
        "test_samples": len(groups["test"]),
        "best_validation_macro_f1": best_val,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro,
        "test_weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
        "training_time_s": training_time,
        "test_eval_time_s": test_time,
        "checkpoint": str(checkpoint),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--models", nargs="+", choices=MODELS.keys(), default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-feature-extractor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-validation", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/comparison"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/emotion_comparison.json"))
    args = parser.parse_args()

    seed_everything(args.seed)
    groups = load_manifest(args.manifest, args.limit_train, args.limit_validation, args.limit_test)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    results = {
        "protocol": "speaker-independent RAVDESS; same split and 8 labels for both pretrained backbones",
        "seed": args.seed,
        "device": str(device),
        "models": {},
    }
    for name in args.models:
        results["models"][name] = train_one(name, MODELS[name], groups, args, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
