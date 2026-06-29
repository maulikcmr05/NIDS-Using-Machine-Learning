# ML-Only Enterprise Project Overview

## Project Positioning

This project is presented as an enterprise-style machine learning solution for network intrusion detection. It follows a practical structure that a large organization could use for an internal security analytics proof of concept.

It does not claim affiliation with any real MNC or commercial organization.

## ML Scope

The project scope is limited to:

- Dataset preparation
- Feature cleaning
- Model training
- Model evaluation
- Model selection
- `.pkl` artifact generation
- Batch prediction

Backend and frontend implementation are intentionally excluded.

## ML Pipeline

```mermaid
flowchart TD
    A[Network Flow CSV Files] --> B[Load Dataset]
    B --> C[Clean Columns and Values]
    C --> D[Drop Non-Feature Columns]
    D --> E[Separate Features and Label]
    E --> F[Encode Target Labels]
    F --> G[Train/Test Split]
    G --> H[Train Candidate Models]
    H --> I[Evaluate Accuracy and Reports]
    I --> J[Select Best Model]
    J --> K[Save best_nids_model.pkl]
```

## Artifact Flow

```mermaid
flowchart LR
    A[train_model.py] --> B[random_forest.pkl]
    A --> C[extra_trees.pkl]
    A --> D[logistic_regression.pkl]
    A --> E[best_nids_model.pkl]
    A --> F[training_metadata.json]
    A --> G[model_comparison.csv]
```

## Prediction Flow

```mermaid
flowchart LR
    A[Input CSV] --> B[predict.py]
    C[best_nids_model.pkl] --> B
    B --> D[Feature Alignment]
    D --> E[Model Prediction]
    E --> F[Output CSV with Predicted_Label]
```

## Enterprise Quality Notes

- The dataset and model artifacts are not committed to GitHub.
- The final model package includes metadata for traceability.
- The code supports deterministic training through `random_state`.
- The training script compares multiple candidate models.
- The prediction script aligns input columns with training columns.
