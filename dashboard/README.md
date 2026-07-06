# Enterprise NIDS ML Dashboard

This Streamlit dashboard is a professional control-center layer for the NIDS machine learning project.

It works directly with local CSV files and `.pkl` model artifacts. It does not require a separate backend service.

## Features

- CSV upload and dataset preview
- Local dataset CSV explorer
- Schema, missing value, duplicate row, and label distribution views
- `.pkl` model registry
- Model metadata inspection
- Feature column inspection
- Model comparison graph
- Class distribution graph
- Confusion matrix graph
- Classification report table
- Train new model from dashboard
- Expected training time before starting a run
- Live elapsed timer while training is running
- Stop training button for a running dashboard training job
- Training history with actual time, expected time, status, and rows used
- Score uploaded CSV files
- Download scored prediction CSV
- Custom one-row scoring using model feature columns
- Dashboard result section with readiness checks and artifact inventory

## Run

From the project root:

```powershell
pip install -r requirements.txt
streamlit run dashboard\nids_dashboard.py
```

Open:

```text
http://localhost:8501
```

## Required Artifacts

Train first if artifacts are missing:

```powershell
python model\train_model.py --data dataset --max-rows-per-file 50000
```

The dashboard reads:

```text
model/artifacts/best_nids_model.pkl
model/artifacts/model_comparison.csv
model/artifacts/*_metrics.json
model/artifacts/training_metadata.json
model/artifacts/training_history.json
```

## Notes

- Large datasets should be sampled during training.
- Model artifacts are local and ignored by Git.
- Dataset files are local and ignored by Git.
