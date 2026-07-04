# NIDS Dashboard

A Streamlit dashboard that visualizes the model artifacts produced by `train_model.py` and `predict.py`. It doesn't retrain anything or touch your dataset — it just reads what training already produced.

---

## 1. Generate the artifacts first

The dashboard is empty until these exist. From the repo root:

```bash
pip install -r requirements.txt

# Train all three models, save comparison + confusion matrices + metadata
python model/train_model.py --data dataset --max-rows-per-file 50000

# Score a CSV with the best saved model
python model/predict.py --input dataset/Friday11.csv --output predictions.csv
```

This produces:
```
model/artifacts/
  training_metadata.json        # best model, accuracy, class distribution, etc.
  model_comparison.csv          # accuracy per model
  random_forest_metrics.json    # confusion matrix + classification report
  extra_trees_metrics.json
  logistic_regression_metrics.json
predictions.csv                 # per-row Predicted_Label from predict.py
```

---

## 2. Run the dashboard

```bash
pip install streamlit plotly
streamlit run dashboard/nids_dashboard.py
```

Opens at `http://localhost:8501`. Use the sidebar to point at `model/artifacts` and `predictions.csv` if they're not in the default relative locations, or upload a predictions CSV directly.

**Tabs:**
- **Overview** — best model, accuracy, training rows, feature count, class distribution
- **Model Comparison** — accuracy across Random Forest / Extra Trees / Logistic Regression, plus macro/weighted F1
- **Confusion Matrix & Report** — interactive heatmap and per-class precision/recall/F1, model selectable
- **Predictions Explorer** — attack rate, breakdown charts, filterable table, downloadable flagged-traffic CSV

---

## Notes

- Re-run `train_model.py` / `predict.py` any time you retrain; the dashboard just re-reads whatever's currently in `model/artifacts/` and `predictions.csv`.
- Hit the **🔄 Refresh** button in the sidebar after retraining — file reads are cached.
