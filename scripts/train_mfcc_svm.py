from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0
N_MFCC = 40


def extract_mfcc_features(path: str | Path) -> np.ndarray:
    """Extract fixed-length MFCC + delta + delta-delta features."""

    samples, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=MAX_SECONDS,
    )

    target = int(SAMPLE_RATE * MAX_SECONDS)

    samples = np.pad(
        samples,
        (0, max(0, target - len(samples))),
    )[:target]

    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
    )

    delta = librosa.feature.delta(mfcc)

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2,
    )

    # Aggregate over time.
    features = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta_delta.mean(axis=1),
            delta_delta.std(axis=1),
        ]
    )

    return features.astype(np.float32)


def load_split(
    rows: list[dict[str, str]],
    split: str,
    labels: list[str],
):
    split_rows = [
        row for row in rows
        if row["split"] == split
    ]

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    X = []
    y = []

    for i, row in enumerate(split_rows, start=1):

        X.append(
            extract_mfcc_features(
                row["path"]
            )
        )

        y.append(
            label_to_index[row["emotion"]]
        )

        if i % 100 == 0:
            print(
                f"  {split}: extracted "
                f"{i}/{len(split_rows)}"
            )

    return (
        np.stack(X),
        np.array(y),
    )


def evaluate(
    model,
    X,
    y,
    labels,
):
    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=list(range(len(labels))),
    )

    return (
        accuracy,
        weighted_f1,
        macro_f1,
        report,
        matrix,
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
        "--output",
        type=Path,
        default=Path(
            "models/mfcc_svm.metrics.json"
        ),
    )

    args = parser.parse_args()

    print("Experiment 5: MFCC + SVM")
    print("-------------------------")
    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Maximum duration: {MAX_SECONDS}s")
    print(f"MFCC coefficients: {N_MFCC}")
    print("Features: MFCC + delta + delta-delta")
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

    print(f"Labels: {labels}")
    print()

    print("Extracting training features...")
    X_train, y_train = load_split(
        rows,
        "train",
        labels,
    )

    print()
    print("Extracting validation features...")
    X_validation, y_validation = load_split(
        rows,
        "validation",
        labels,
    )

    print()
    print("Extracting test features...")
    X_test, y_test = load_split(
        rows,
        "test",
        labels,
    )

    print()
    print("Feature shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test: {X_test.shape}")
    print()

    # Expected:
    # 40 MFCC means
    # 40 MFCC std
    # 40 delta means
    # 40 delta std
    # 40 delta-delta means
    # 40 delta-delta std
    # = 240 features

    # IMPORTANT:
    # Scaling is fitted only on training data.
    candidate_C = [
        0.1,
        1.0,
        10.0,
        100.0,
    ]

    validation_results = []

    print("Validation model selection")
    print("--------------------------")

    best_C = None
    best_validation_f1 = -1.0

    for C in candidate_C:

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=C,
                        gamma="scale",
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        (
            val_accuracy,
            val_weighted_f1,
            val_macro_f1,
            _,
            _,
        ) = evaluate(
            model,
            X_validation,
            y_validation,
            labels,
        )

        validation_results.append(
            {
                "C": C,
                "accuracy": val_accuracy,
                "weighted_f1": val_weighted_f1,
                "macro_f1": val_macro_f1,
            }
        )

        print(
            f"C={C:<5} "
            f"accuracy={val_accuracy:.4f} "
            f"weighted_f1={val_weighted_f1:.4f} "
            f"macro_f1={val_macro_f1:.4f}"
        )

        if val_weighted_f1 > best_validation_f1:

            best_validation_f1 = val_weighted_f1
            best_C = C

    print()
    print(
        f"Selected C={best_C} "
        f"using validation weighted F1="
        f"{best_validation_f1:.4f}"
    )
    print()

    # Refit selected model on TRAINING DATA ONLY.
    final_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=best_C,
                    gamma="scale",
                ),
            ),
        ]
    )

    final_model.fit(
        X_train,
        y_train,
    )

    print("Final test evaluation")
    print("---------------------")

    (
        test_accuracy,
        test_weighted_f1,
        test_macro_f1,
        report,
        matrix,
    ) = evaluate(
        final_model,
        X_test,
        y_test,
        labels,
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test weighted F1: "
        f"{test_weighted_f1:.4f}"
    )

    print(
        f"Test macro F1: "
        f"{test_macro_f1:.4f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            final_model.predict(X_test),
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    metrics = {
        "experiment": "MFCC + RBF SVM",
        "sample_rate": SAMPLE_RATE,
        "max_seconds": MAX_SECONDS,
        "n_mfcc": N_MFCC,
        "feature_dimensions": int(
            X_train.shape[1]
        ),
        "feature_description": (
            "MFCC mean/std + "
            "delta mean/std + "
            "delta-delta mean/std"
        ),
        "train_samples": len(y_train),
        "validation_samples": len(y_validation),
        "test_samples": len(y_test),
        "candidate_C": candidate_C,
        "validation_results": validation_results,
        "selected_C": best_C,
        "validation_weighted_f1": (
            best_validation_f1
        ),
        "test_accuracy": test_accuracy,
        "test_weighted_f1": test_weighted_f1,
        "test_macro_f1": test_macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()
    print(
        f"Metrics saved: {args.output}"
    )


if __name__ == "__main__":
    main()
