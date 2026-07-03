import pickle
from pathlib import Path
import pandas as pd

MODEL_DIR = Path(__file__).parent / "ml_artifacts"

class ModelHandler:
    def __init__(self):
        with (MODEL_DIR / "best_nids_model.pkl").open("rb") as f:
            package = pickle.load(f)

        self.model           = package["model"]
        self.model_name      = package["model_name"]
        self.accuracy        = package["accuracy"]
        self.label_encoder   = package["label_encoder"]
        self.feature_columns = package["feature_columns"]
        self.labels          = package["labels"]

        print(f"Model loaded: {self.model_name} | Accuracy: {self.accuracy:.4f}")
        print(f"Classes: {self.labels}")
        print(f"Features: {len(self.feature_columns)}")

    def predict(self, df: pd.DataFrame):
        encoded_preds = self.model.predict(df)
        labels = self.label_encoder.inverse_transform(encoded_preds)

        confidences = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df)
            confidences = probs.max(axis=1)

        return labels, confidences

# Loaded once when the app starts
model_handler = ModelHandler()