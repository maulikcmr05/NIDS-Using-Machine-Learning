from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_utils import load_dataset, split_features_target


def build_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=None,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=150,
                        n_jobs=-1,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        n_jobs=-1,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def save_metrics(output_dir: Path, model_name: str, y_true, y_pred, labels: list[str]) -> dict:
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix,
    }

    metrics_file = output_dir / f"{model_name}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for Network Intrusion Detection.")
    parser.add_argument("--data", default="dataset", help="CSV file or folder containing CSV files.")
    parser.add_argument("--output", default="model/artifacts", help="Folder where trained models will be saved.")
    parser.add_argument("--max-rows-per-file", type=int, default=50000, help="Rows sampled from each CSV. Use 0 for full files.")
    parser.add_argument("--include-merged", action="store_true", help="Also use Merged.csv when a dataset folder has other CSV files.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_rows = None if args.max_rows_per_file == 0 else args.max_rows_per_file
    data = load_dataset(
        args.data,
        max_rows_per_file=max_rows,
        random_state=args.random_state,
        include_merged=args.include_merged,
    )
    print(f"Final dataset shape: {data.shape}")
    print("Class distribution:")
    print(data["Label"].value_counts())

    x, y_text = split_features_target(data)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    labels = list(label_encoder.classes_)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    models = build_models(args.random_state)
    results: list[dict] = []
    best_name = ""
    best_score = -1.0
    best_model = None

    for name, pipeline in models.items():
        print(f"\nTraining {name} ...")
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = save_metrics(output_dir, name, y_test, predictions, labels)
        score = metrics["accuracy"]
        results.append({"model": name, "accuracy": score})
        with (output_dir / f"{name}.pkl").open("wb") as file:
            pickle.dump(pipeline, file)
        print(f"{name} accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_name = name
            best_model = pipeline

    model_package = {
        "model": best_model,
        "model_name": best_name,
        "accuracy": best_score,
        "label_encoder": label_encoder,
        "feature_columns": list(x.columns),
        "labels": labels,
    }
    best_model_path = output_dir / "best_nids_model.pkl"
    with best_model_path.open("wb") as file:
        pickle.dump(model_package, file)

    pd.DataFrame(results).sort_values("accuracy", ascending=False).to_csv(
        output_dir / "model_comparison.csv", index=False
    )
    print(f"\nBest model: {best_name} ({best_score:.4f})")
    print(f"Saved: {best_model_path}")


if __name__ == "__main__":
    main()
