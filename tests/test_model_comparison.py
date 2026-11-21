from pathlib import Path

from scripts.compare_asr import statement_reference, wer
from scripts.compare_sentiment import load_rows


def test_ravdess_statement_references() -> None:
    assert statement_reference("03-01-05-01-01-01-16.wav") == "Kids are talking by the door."
    assert statement_reference("03-01-03-02-02-01-16.wav") == "Dogs are sitting by the door."


def test_wer_identity_and_single_error() -> None:
    assert wer("Kids are talking by the door.", "Kids are talking by the door.") == 0.0
    assert wer("Kids are talking by the door.", "Kids are talking near the door.") > 0.0


def test_sentiment_demo_has_expected_three_classes() -> None:
    path = Path("data/processed/sentiment_demo.csv")
    rows = load_rows(path, 0)
    assert len(rows) == 20
    assert {row["label"] for row in rows} == {"negative", "neutral", "positive"}
