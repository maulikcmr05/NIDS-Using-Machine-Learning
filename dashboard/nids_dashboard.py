"""
NIDS Dashboard — companion visualization layer for
maulikcmr05/NIDS-Using-Machine-Learning

This app does NOT add a backend, live packet capture, or a database.
It reads the artifacts that the repo's own scripts already produce:

    model/artifacts/training_metadata.json
    model/artifacts/model_comparison.csv
    model/artifacts/<model_name>_metrics.json   (per model, incl. confusion matrix)
    predictions.csv (or any file produced by model/predict.py)

Run it from the repo root after training a model:

    pip install -r requirements.txt
    pip install streamlit plotly
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")

ARTIFACTS_DEFAULT = "model/artifacts"
BENIGN_LABELS = {"benign", "normal", "normal.", "no attack"}


# --------------------------------------------------------------------------
# Loaders — every one fails soft, so a partially-trained project still renders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def discover_metric_files(artifacts_dir: Path) -> dict[str, dict]:
    """Find every <model_name>_metrics.json in the artifacts folder."""
    out = {}
    for f in sorted(artifacts_dir.glob("*_metrics.json")):
        model_name = f.name.replace("_metrics.json", "")
        data = load_json(f)
        if data:
            out[model_name] = data
    return out


def is_attack_label(label: str) -> bool:
    return str(label).strip().lower() not in BENIGN_LABELS


# --------------------------------------------------------------------------
# Sidebar — point at wherever the artifacts / predictions actually live
# --------------------------------------------------------------------------
st.sidebar.title("🛡️ NIDS Dashboard")
st.sidebar.caption("Reads outputs from train_model.py and predict.py")

artifacts_dir = Path(st.sidebar.text_input("Artifacts folder", ARTIFACTS_DEFAULT))
predictions_path = Path(
    st.sidebar.text_input("Predictions CSV (from predict.py)", "predictions.csv")
)
uploaded_predictions = st.sidebar.file_uploader(
    "...or upload a predictions CSV", type=["csv"]
)

if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()

metadata = load_json(artifacts_dir / "training_metadata.json")
comparison = load_csv(artifacts_dir / "model_comparison.csv")
per_model_metrics = discover_metric_files(artifacts_dir)

st.sidebar.divider()
if metadata:
    st.sidebar.success(f"Loaded artifacts for: {metadata.get('project_name', 'NIDS')}")
else:
    st.sidebar.warning(
        "No training_metadata.json found yet.\n\n"
        "Run:\n`python model/train_model.py --data dataset`"
    )

st.title("Network Intrusion Detection — Model & Traffic Dashboard")

tab_overview, tab_models, tab_confusion, tab_predictions = st.tabs(
    ["📊 Overview", "🏆 Model Comparison", "🧩 Confusion Matrix & Report", "🚨 Predictions Explorer"]
)

# --------------------------------------------------------------------------
# TAB 1 — Overview
# --------------------------------------------------------------------------
with tab_overview:
    if not metadata:
        st.info(
            "Train a model first, then point the sidebar's **Artifacts folder** at "
            "`model/artifacts` to populate this dashboard."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", metadata.get("best_model_name", "—"))
        c2.metric("Best accuracy", f"{metadata.get('best_accuracy', 0):.2%}")
        c3.metric("Training rows", f"{metadata.get('training_rows', 0):,}")
        c4.metric("Feature count", metadata.get("feature_count", "—"))

        c5, c6, c7 = st.columns(3)
        c5.metric("Data source", metadata.get("data_source", "—"))
        c6.metric("Max rows / file", metadata.get("max_rows_per_file", "—"))
        c7.metric("Trained at (UTC)", str(metadata.get("trained_at_utc", "—"))[:19])

        st.divider()
        st.subheader("Class distribution in training data")
        dist = metadata.get("class_distribution", {})
        if dist:
            dist_df = (
                pd.DataFrame(list(dist.items()), columns=["Label", "Count"])
                .sort_values("Count", ascending=False)
            )
            dist_df["Type"] = dist_df["Label"].apply(
                lambda x: "Attack" if is_attack_label(x) else "Benign"
            )
            fig = px.bar(
                dist_df,
                x="Label",
                y="Count",
                color="Type",
                color_discrete_map={"Benign": "#2E7D32", "Attack": "#C62828"},
                log_y=True,
                title="Training class distribution (log scale)",
            )
            fig.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)

            benign_n = dist_df.loc[dist_df["Type"] == "Benign", "Count"].sum()
            attack_n = dist_df.loc[dist_df["Type"] == "Attack", "Count"].sum()
            total = benign_n + attack_n
            if total:
                st.caption(
                    f"Benign: {benign_n:,} ({benign_n/total:.1%})  |  "
                    f"Attack: {attack_n:,} ({attack_n/total:.1%})  |  "
                    f"Attack classes: {(dist_df['Type'] == 'Attack').sum()}"
                )

# --------------------------------------------------------------------------
# TAB 2 — Model comparison
# --------------------------------------------------------------------------
with tab_models:
    if comparison is None or comparison.empty:
        st.info("No `model_comparison.csv` found yet — run training to generate it.")
    else:
        st.subheader("Accuracy by model")
        comp_sorted = comparison.sort_values("accuracy", ascending=False)
        fig = px.bar(
            comp_sorted,
            x="model",
            y="accuracy",
            text_auto=".2%",
            color="accuracy",
            color_continuous_scale="Blues",
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comp_sorted.style.format({"accuracy": "{:.4f}"}), use_container_width=True)

    if per_model_metrics:
        st.divider()
        st.subheader("Macro / weighted F1 per model")
        rows = []
        for name, m in per_model_metrics.items():
            report = m.get("classification_report", {})
            rows.append(
                {
                    "model": name,
                    "accuracy": m.get("accuracy"),
                    "macro_f1": report.get("macro avg", {}).get("f1-score"),
                    "weighted_f1": report.get("weighted avg", {}).get("f1-score"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --------------------------------------------------------------------------
# TAB 3 — Confusion matrix + per-class report for a chosen model
# --------------------------------------------------------------------------
with tab_confusion:
    if not per_model_metrics:
        st.info("No `*_metrics.json` files found yet — run training to generate them.")
    else:
        chosen = st.selectbox("Model", list(per_model_metrics.keys()))
        m = per_model_metrics[chosen]
        labels = m.get("labels", [])
        matrix = m.get("confusion_matrix", [])

        st.subheader(f"Confusion matrix — {chosen} (accuracy {m.get('accuracy', 0):.2%})")
        if matrix and labels:
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    text=matrix,
                    texttemplate="%{text}",
                )
            )
            fig.update_layout(
                xaxis_title="Predicted label",
                yaxis_title="True label",
                yaxis_autorange="reversed",
                height=max(400, 40 * len(labels)),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-class precision / recall / F1")
        report = m.get("classification_report", {})
        if report:
            report_df = (
                pd.DataFrame(report).T.drop(index=["accuracy"], errors="ignore")
                .rename_axis("class")
                .reset_index()
            )
            st.dataframe(
                report_df.style.format(
                    {"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}
                ),
                use_container_width=True,
            )

# --------------------------------------------------------------------------
# TAB 4 — Explore a predictions.csv produced by predict.py
# --------------------------------------------------------------------------
with tab_predictions:
    preds = None
    if uploaded_predictions is not None:
        preds = pd.read_csv(uploaded_predictions, low_memory=False)
    else:
        preds = load_csv(predictions_path)

    if preds is None:
        st.info(
            "No predictions loaded. Run:\n\n"
            "`python model/predict.py --input dataset/Friday11.csv --output predictions.csv`\n\n"
            "then point the sidebar at the resulting CSV, or upload it directly."
        )
    elif "Predicted_Label" not in preds.columns:
        st.error("This CSV has no `Predicted_Label` column — is it the output of predict.py?")
    else:
        preds = preds.copy()
        preds["Traffic_Type"] = preds["Predicted_Label"].apply(
            lambda x: "Attack" if is_attack_label(x) else "Benign"
        )

        total = len(preds)
        attacks = int((preds["Traffic_Type"] == "Attack").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total flows scored", f"{total:,}")
        c2.metric("Flagged as attack", f"{attacks:,}")
        c3.metric("Attack rate", f"{attacks/total:.2%}" if total else "—")

        left, right = st.columns([1, 1])
        with left:
            label_counts = preds["Predicted_Label"].value_counts().reset_index()
            label_counts.columns = ["Predicted_Label", "Count"]
            fig = px.pie(
                label_counts,
                names="Predicted_Label",
                values="Count",
                title="Predicted label breakdown",
                hole=0.45,
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig2 = px.bar(
                label_counts.sort_values("Count", ascending=True),
                x="Count",
                y="Predicted_Label",
                orientation="h",
                title="Flow count per predicted label",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Flagged traffic (non-benign predictions)")
        alert_choices = sorted(preds.loc[preds["Traffic_Type"] == "Attack", "Predicted_Label"].unique())
        selected = st.multiselect("Filter by predicted attack type", alert_choices, default=alert_choices)
        alerts = preds[preds["Predicted_Label"].isin(selected)] if selected else preds.iloc[0:0]
        st.dataframe(alerts, use_container_width=True, height=350)
        st.download_button(
            "⬇️ Download flagged rows as CSV",
            alerts.to_csv(index=False).encode("utf-8"),
            file_name="flagged_traffic.csv",
            mime="text/csv",
        )

        with st.expander("Full prediction table"):
            st.dataframe(preds, use_container_width=True, height=400)
