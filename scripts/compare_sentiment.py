"""Compare 3-class DistilBERT and RoBERTa sentiment models.

The script evaluates both candidates on exactly the same CSV rows. CSV format:
text,label where label is negative, neutral, or positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

MODELS = {
    "distilbert": "Ido-shraga/distilbert-base-tweeteval-sentiment",
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment",
}
LABELS = {"negative", "neutral", "positive"}


def load_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = row["label"].lower()
        if row["label"] not in LABELS:
            raise ValueError(f"Unsupported label: {row['label']}")
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/sentiment_demo.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/processed/sentiment_comparison.json"))
    args = parser.parse_args()

    rows = load_rows(args.input, args.limit)
    device = 0 if torch.cuda.is_available() else -1
    results = {"protocol": "same text rows for both models", "n": len(rows), "models": {}}

    from transformers import pipeline

    for name, model_id in MODELS.items():
        print(f"\nLoading {name}: {model_id}")
        classifier = pipeline("text-classification", model=model_id, device=device)
        y_true: list[str] = []
        y_pred: list[str] = []
        latencies: list[float] = []
        predictions = []
        for index, row in enumerate(rows, start=1):
            start = time.perf_counter()
            result = classifier(row["text"], truncation=True, max_length=256)[0]
            elapsed = time.perf_counter() - start
            label = result["label"].lower()
            if label.startswith("label_"):
                mapping = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
                label = mapping[label]
            y_true.append(row["label"])
            y_pred.append(label)
            latencies.append(elapsed)
            predictions.append({"text": row["text"], "true": row["label"], "predicted": label, "confidence": float(result["score"]), "latency_s": elapsed})
            print(f"  {index:>3}/{len(rows)}  {row['label']:>8} -> {label:<8} {result['score']:.3f}")

        results["models"][name] = {
            "model_id": model_id,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=sorted(LABELS), average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, labels=sorted(LABELS), average="weighted", zero_division=0),
            "weighted_precision": precision_score(y_true, y_pred, labels=sorted(LABELS), average="weighted", zero_division=0),
            "weighted_recall": recall_score(y_true, y_pred, labels=sorted(LABELS), average="weighted", zero_division=0),
            "mean_latency_s": sum(latencies) / max(1, len(latencies)),
            "classification_report": classification_report(y_true, y_pred, labels=sorted(LABELS), output_dict=True, zero_division=0),
            "predictions": predictions,
        }
        del classifier

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
