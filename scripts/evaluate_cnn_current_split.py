import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from emotion_model.audio import log_mel
from emotion_model.model import EmotionCNN


MANIFEST = Path("data/processed/manifest.csv")
CHECKPOINT = Path("models/emotion_cnn.pt")
OUTPUT = Path("data/processed/cnn_current_split.json")

device = "cpu"

# Load manifest
with MANIFEST.open() as f:
    rows = list(csv.DictReader(f))

test_rows = [r for r in rows if r["split"] == "test"]

labels = sorted({r["emotion"] for r in rows})
label_to_id = {label: i for i, label in enumerate(labels)}

print("=" * 70)
print("CNN EVALUATION — CURRENT PHASE-2 TEST SPLIT")
print("=" * 70)
print(f"Device: {device}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Test samples: {len(test_rows)}")
print(f"Labels: {labels}")
print("=" * 70)

# Load model
model = EmotionCNN(classes=len(labels))
checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.inference_mode():
    for i, row in enumerate(test_rows, 1):
        features = log_mel(Path(row["path"])).unsqueeze(0).to(device)

        logits = model(features)
        prediction = logits.argmax(dim=1).item()

        true_id = label_to_id[row["emotion"]]

        y_true.append(true_id)
        y_pred.append(prediction)

        print(
            f"[{i:02d}/{len(test_rows)}] "
            f"true={row['emotion']:<10} "
            f"pred={labels[prediction]}"
        )

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)
weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(labels))),
    target_names=labels,
    output_dict=True,
    zero_division=0,
)

results = {
    "protocol": "current Phase-2 speaker-independent RAVDESS test split",
    "checkpoint": str(CHECKPOINT),
    "device": device,
    "test_samples": len(test_rows),
    "labels": labels,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "classification_report": report,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(results, indent=2))

print()
print("=" * 70)
print("CNN RESULTS")
print("=" * 70)
print(f"Accuracy:    {accuracy:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print()
print("Saved:", OUTPUT)
print("=" * 70)
