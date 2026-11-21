"""Compare Whisper Base vs Small on the speaker-independent RAVDESS test split.

RAVDESS speech uses two fixed lexical statements. The statement id is encoded in
field 5 of each filename: 01 = "Kids are talking by the door." and
02 = "Dogs are sitting by the door.". WER is therefore computed against the
known utterance rather than against Whisper output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path



REFERENCES = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}
MODELS = ("base", "small")


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (r != h),
            ))
        prev = cur
    return prev[-1] / max(1, len(ref))


def statement_reference(path: str) -> str:
    match = re.match(r"^03-01-\d{2}-\d{2}-(?P<statement>\d{2})-\d{2}-\d{2}\.wav$", Path(path).name)
    if not match:
        raise ValueError(f"Unexpected RAVDESS filename: {path}")
    return REFERENCES[match.group("statement")]


def load_manifest(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "test"]
    if limit:
        rows = rows[:limit]
    missing = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} test audio files are missing. The manifest contains "
            "absolute paths from the original machine; regenerate the manifest "
            "on this Mac with the local RAVDESS directory."
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--limit", type=int, default=0, help="0 = full test split")
    parser.add_argument("--output", type=Path, default=Path("data/processed/asr_comparison.json"))
    args = parser.parse_args()

    rows = load_manifest(args.manifest, args.limit)
    results: dict[str, object] = {
        "protocol": "speaker-independent RAVDESS test split",
        "n": len(rows),
        "models": {},
    }

    import whisper

    for model_name in args.models:
        print(f"\nLoading Whisper {model_name}...")
        model = whisper.load_model(model_name)
        predictions = []
        total_latency = 0.0
        total_errors = 0
        total_ref_words = 0
        for index, row in enumerate(rows, start=1):
            reference = statement_reference(row["path"])
            start = time.perf_counter()
            output = model.transcribe(row["path"], fp16=False, language="en")
            elapsed = time.perf_counter() - start
            hypothesis = output["text"].strip()
            total_latency += elapsed
            ref_tokens = normalize(reference)
            total_errors += round(wer(reference, hypothesis) * len(ref_tokens))
            total_ref_words += len(ref_tokens)
            predictions.append({
                "path": row["path"],
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer(reference, hypothesis),
                "latency_s": elapsed,
            })
            print(f"  {index:>3}/{len(rows)}  WER={predictions[-1]['wer']:.3f}  {hypothesis!r}")

        results["models"][model_name] = {
            "wer": total_errors / max(1, total_ref_words),
            "mean_latency_s": total_latency / max(1, len(rows)),
            "total_time_s": total_latency,
            "predictions": predictions,
        }
        del model

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
