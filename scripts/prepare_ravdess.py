"""Create a speaker-independent RAVDESS manifest from extracted audio."""

import argparse
import csv
from pathlib import Path

from emotion_model.ravdess import parse_ravdess, speaker_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Extracted RAVDESS Speech_Actor_01..24 directory")
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--limit-per-emotion", type=int, default=0, help="Optional small-dataset limit (0 = all)")
    args = parser.parse_args()

    by_emotion: dict[str, list] = {}
    for path in sorted(args.source.rglob("*.wav")):
        try:
            record = parse_ravdess(path)
        except ValueError:
            continue
        by_emotion.setdefault(record.emotion, []).append(record)
    records = []
    counts: dict[str, int] = {}
    for emotion, candidates in sorted(by_emotion.items()):
        # Filename ordering cycles through all actors for one stimulus before the
        # next stimulus, preserving train/validation/test speaker coverage in a
        # small smoke-test subset.
        chosen = sorted(candidates, key=lambda record: record.path.name)
        if args.limit_per_emotion:
            chosen = chosen[: args.limit_per_emotion]
        records.extend(chosen)
        counts[emotion] = len(chosen)
    if not records:
        raise SystemExit("No RAVDESS speech WAV files found. Check the extracted source path.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "emotion", "speaker", "split"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "path": record.path.resolve(),
                "emotion": record.emotion,
                "speaker": record.speaker,
                "split": speaker_split(record.speaker),
            })
    print(f"Wrote {len(records)} files to {args.output}")
    print("Counts:", counts)


if __name__ == "__main__":
    main()
