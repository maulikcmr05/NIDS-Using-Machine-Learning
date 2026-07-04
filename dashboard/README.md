# NIDS Dashboards

Two ways to visualize the model artifacts produced by `train_model.py` and `predict.py` — pick whichever fits your workflow.

| | `nids_dashboard.py` | `flowscope.html` |
|---|---|---|
| Requires | Python + Streamlit | Just a browser |
| Run with | `streamlit run` | Double-click, or `python -m http.server` |
| Best for | Quick local iteration while training | A polished, shareable view |

Both read the **same artifact files** — nothing here retrains a model or touches your dataset.

---

## 1. Generate the artifacts first

Both dashboards are empty until these exist. From the repo root:

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

## 2. Option A — Streamlit dashboard

```bash
pip install streamlit plotly
streamlit run dashboard/nids_dashboard.py
```
Opens at `http://localhost:8501`. Use the sidebar to point at `model/artifacts` and `predictions.csv` if they're not in the default relative locations, or upload a predictions CSV directly.

Tabs: **Overview** · **Model Comparison** · **Confusion Matrix & Report** · **Predictions Explorer**

---

## 3. Option B — FLOWSCOPE (standalone HTML)

**Quick look (no auto-load):**
Just double-click `dashboard/flowscope.html`. It opens with demo data and lets you load your real files manually via the **Data Source** tab.

**Auto-load your project's artifacts (recommended):**
```bash
cd NIDS-Using-Machine-Learning
python -m http.server 8000
```
Then visit `http://localhost:8000/dashboard/flowscope.html` — it automatically fetches `model/artifacts/*.json`, `model_comparison.csv`, and `predictions.csv` with no clicking required.

> Auto-load only works when served over `http://`, not when opened directly (`file://`) — that's a browser security restriction, not a bug. Opened directly, it falls back to the manual file picker.

Runs entirely client-side (Chart.js + PapaParse via CDN) — your data never leaves the browser.

---

## Notes

- Re-run `train_model.py` / `predict.py` any time you retrain; both dashboards just re-read whatever's currently in `model/artifacts/` and `predictions.csv`.
- For Streamlit, hit the **🔄 Refresh** button in the sidebar after retraining (it caches file reads).
- For FLOWSCOPE, just refresh the browser tab.
