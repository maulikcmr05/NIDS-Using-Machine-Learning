from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "model"))

from data_utils import align_prediction_columns


st.set_page_config(page_title="NIDS ML Detector", page_icon="NIDS", layout="wide")
st.title("Network Intrusion Detection System")

model_path = ROOT / "model" / "artifacts" / "best_nids_model.pkl"

if not model_path.exists():
    st.error("Model not found. Train first: python model/train_model.py --data dataset")
    st.stop()

with model_path.open("rb") as file:
    package = pickle.load(file)
model = package["model"]
label_encoder = package["label_encoder"]
feature_columns = package["feature_columns"]

uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded, low_memory=False)
    x = align_prediction_columns(raw, feature_columns)
    encoded = model.predict(x)
    raw["Predicted_Label"] = label_encoder.inverse_transform(encoded)

    st.subheader("Prediction Summary")
    st.dataframe(raw["Predicted_Label"].value_counts().rename("count"))

    st.subheader("Predicted Rows")
    st.dataframe(raw.head(200), use_container_width=True)

    csv = raw.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "nids_predictions.csv", "text/csv")
else:
    st.info("Upload a CSV file with CICIDS-style flow features.")
