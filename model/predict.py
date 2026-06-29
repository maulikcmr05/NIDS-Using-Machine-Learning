from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from data_utils import align_prediction_columns


def predict_file(model_path: str | Path, input_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    with Path(model_path).open("rb") as file:
        package = pickle.load(file)
    model = package["model"]
    label_encoder = package["label_encoder"]
    feature_columns = package["feature_columns"]

    raw = pd.read_csv(input_csv, low_memory=False)
    x = align_prediction_columns(raw, feature_columns)
    encoded_predictions = model.predict(x)
    predictions = label_encoder.inverse_transform(encoded_predictions)

    result = raw.copy()
    result["Predicted_Label"] = predictions
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict attack labels for network-flow CSV.")
    parser.add_argument("--model", default="model/artifacts/best_nids_model.pkl")
    parser.add_argument("--input", required=True, help="CSV file for prediction.")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    result = predict_file(args.model, args.input, args.output)
    print(result[["Predicted_Label"]].value_counts())
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
