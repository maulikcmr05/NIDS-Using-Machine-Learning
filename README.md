# Enterprise Network Intrusion Detection ML System

This repository contains a machine learning focused Network Intrusion Detection System (NIDS). It is structured like an enterprise security analytics project for a large organization, but it is not affiliated with any real company.

The core project is ML-first: train, evaluate, save, and use intrusion detection models. It also includes a professional Streamlit dashboard for ML operations: CSV inspection, PKL model inspection, graphs, training controls, prediction, and custom row scoring. The dashboard works directly with local files and does not require a separate backend API.

## Objective

Train supervised machine learning models on network-flow CSV data and save the best performing model as a `.pkl` artifact for later prediction.

## What This Project Contains

- Data loading and cleaning utilities
- Large CSV sampling support
- Feature and target separation
- Label encoding
- Multiple ML classifiers
- Model evaluation reports
- Best model selection
- `.pkl` model artifact generation
- CLI-based batch prediction
- Enterprise-style model metadata
- Professional dashboard for CSV, PKL, graphs, training, prediction, and custom scoring

## What This Project Does Not Contain

- No backend server
- No database layer
- No live network packet capture
- No generated model artifacts committed to GitHub
- No dataset committed to GitHub

## Project Structure

```text
NIDS-Using-Machine-Learning/
  dataset/                         # Local only, ignored by Git
  docs/                            # Project documentation
  dashboard/
    nids_dashboard.py              # Streamlit dashboard for ML operations
    README.md
  model/
    data_utils.py                  # Data loading, cleaning, feature alignment
    train_model.py                 # ML training and .pkl artifact creation
    predict.py                     # Batch prediction using trained .pkl model
  .gitignore
  requirements.txt
  README.md
```

## Dataset

Place CICIDS-style CSV files inside the local `dataset/` folder.

Example:

```text
dataset/Friday11.csv
dataset/Friday21.csv
dataset/Monday1.csv
dataset/Tuesday1.csv
dataset/Wednesday1.csv
dataset/Thursday11.csv
dataset/Thursday21.csv
```

The `Label` column is required because it is the supervised learning target.

The `dataset/` folder is intentionally ignored by Git because these files are large.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train Models

Fast training:

```powershell
python model\train_model.py --data dataset --max-rows-per-file 50000
```

More training data:

```powershell
python model\train_model.py --data dataset --max-rows-per-file 150000
```

Full files:

```powershell
python model\train_model.py --data dataset --max-rows-per-file 0
```

## Training Outputs

Generated files are saved locally under `model/artifacts/`.

```text
model/artifacts/best_nids_model.pkl
model/artifacts/random_forest.pkl
model/artifacts/extra_trees.pkl
model/artifacts/logistic_regression.pkl
model/artifacts/model_comparison.csv
model/artifacts/*_metrics.json
model/artifacts/training_metadata.json
```

These files are ignored by Git because trained model artifacts can be large and should be regenerated locally.

## Predict

```powershell
python model\predict.py --input dataset\Friday11.csv --output predictions.csv
```

The output CSV contains:

```text
Predicted_Label
```

## Dashboard

The dashboard is an enterprise-style local ML command center.

```powershell
streamlit run dashboard\nids_dashboard.py
```

Dashboard features:

- View and profile CSV files
- Inspect `.pkl` model artifacts
- View model metadata and feature columns
- Compare trained models
- View confusion matrix and classification report
- Train models on new data
- Score uploaded CSV files
- Download prediction output
- Score one custom row using model features

## Models Used

- Random Forest Classifier
- Extra Trees Classifier
- Logistic Regression

The best model is selected by test accuracy and saved as `best_nids_model.pkl`.

## Model Artifact Metadata

The saved `.pkl` package contains:

- Trained model
- Model name
- Accuracy
- Label encoder
- Feature column list
- Output labels
- Project metadata
- Training row count
- Feature count
- Class distribution
- Training timestamp

## GitHub Policy

The repository excludes:

- `dataset/`
- `.venv/`
- `model/artifacts/`
- `__pycache__/`
- prediction output CSV files

This keeps the repository clean and suitable for GitHub.
