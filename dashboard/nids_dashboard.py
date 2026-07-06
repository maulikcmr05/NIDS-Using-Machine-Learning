from __future__ import annotations

import json
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from data_utils import TARGET_COLUMN, align_prediction_columns


ARTIFACTS_DEFAULT = ROOT / "model" / "artifacts"
DATASET_DEFAULT = ROOT / "dataset"
MODEL_DEFAULT = ARTIFACTS_DEFAULT / "best_nids_model.pkl"
BENIGN_LABELS = {"benign", "normal", "normal.", "no attack"}


st.set_page_config(
    page_title="Enterprise NIDS ML Command Center",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)


ENTERPRISE_CSS = """
<style>
    .stApp {
        background: #0b0f17;
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #243244;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1520px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .hero {
        padding: 26px 30px;
        border: 1px solid #263244;
        background: linear-gradient(135deg, #101827 0%, #111827 50%, #0f172a 100%);
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 34px;
        line-height: 1.15;
    }
    .hero p {
        margin: 0;
        color: #a7b0c0;
        font-size: 15px;
    }
    .status-card {
        padding: 16px 18px;
        border: 1px solid #263244;
        background: #111827;
        border-radius: 8px;
        min-height: 112px;
    }
    .status-label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 8px;
    }
    .status-value {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 700;
        line-height: 1.2;
    }
    .status-subtle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 8px;
    }
    .section-note {
        padding: 12px 14px;
        border: 1px solid #334155;
        background: #0f172a;
        border-radius: 8px;
        color: #cbd5e1;
    }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stTabs"] button p {
        font-size: 15px;
        font-weight: 600;
    }
</style>
"""


st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def is_attack_label(label: Any) -> bool:
    return str(label).strip().lower() not in BENIGN_LABELS


@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict[str, Any] | None:
    path = Path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path_text: str, nrows: int | None = None) -> pd.DataFrame | None:
    path = Path(path_text)
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False, nrows=nrows)


@st.cache_resource(show_spinner=False)
def load_pickle_model(path_text: str) -> dict[str, Any] | None:
    path = Path(path_text)
    if not path.exists():
        return None
    with path.open("rb") as file:
        package = pickle.load(file)
    return package


def load_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file, low_memory=False)


def discover_csv_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.csv"))


def discover_pkl_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.pkl"))


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


@st.cache_data(show_spinner=False)
def estimate_csv_rows(path_text: str) -> int:
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return 0

    sample_size = min(path.stat().st_size, 1024 * 1024)
    with path.open("rb") as file:
        sample = file.read(sample_size)

    line_count = max(sample.count(b"\n"), 1)
    avg_bytes_per_line = max(sample_size / line_count, 1)
    return int(path.stat().st_size / avg_bytes_per_line)


def training_files(data_path: str, include_merged: bool) -> list[Path]:
    path = resolve_project_path(data_path)
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if not path.exists():
        return []
    files = sorted(path.glob("*.csv"))
    if not include_merged and len(files) > 1:
        files = [file for file in files if file.name.lower() != "merged.csv"]
    return files


def training_estimate(data_path: str, max_rows: int, include_merged: bool) -> dict[str, Any]:
    files = training_files(data_path, include_merged)
    rows = []
    for file in files:
        estimated_rows = estimate_csv_rows(str(file))
        selected_rows = estimated_rows if max_rows == 0 else min(estimated_rows, max_rows)
        rows.append(
            {
                "file": file.name,
                "size": file_size(file),
                "estimated_rows": estimated_rows,
                "selected_rows": selected_rows,
            }
        )

    total_rows = sum(item["selected_rows"] for item in rows)
    # Conservative laptop estimate for this project: data prep plus 3 sklearn models.
    low_seconds = int(45 + total_rows * 0.0015)
    high_seconds = int(90 + total_rows * 0.0035)
    return {
        "files": rows,
        "total_rows": total_rows,
        "low_seconds": low_seconds,
        "high_seconds": high_seconds,
    }


def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def discover_metrics(folder: Path) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    if not folder.exists():
        return metrics
    for path in sorted(folder.glob("*_metrics.json")):
        data = load_json(str(path))
        if data:
            metrics[path.name.replace("_metrics.json", "")] = data
    return metrics


def package_summary(package: dict[str, Any] | None) -> dict[str, Any]:
    if not package:
        return {}
    metadata = package.get("metadata", {})
    return {
        "model_name": package.get("model_name") or metadata.get("best_model_name") or "unknown",
        "accuracy": package.get("accuracy") or metadata.get("best_accuracy"),
        "feature_count": len(package.get("feature_columns", [])) or metadata.get("feature_count"),
        "labels": package.get("labels", []),
        "metadata": metadata,
    }


def predict_dataframe(package: dict[str, Any], raw: pd.DataFrame) -> pd.DataFrame:
    model = package["model"]
    label_encoder = package["label_encoder"]
    feature_columns = package["feature_columns"]
    aligned = align_prediction_columns(raw, feature_columns)
    encoded = model.predict(aligned)
    predictions = label_encoder.inverse_transform(encoded)
    result = raw.copy()
    result["Predicted_Label"] = predictions
    result["Traffic_Type"] = result["Predicted_Label"].apply(
        lambda value: "Attack" if is_attack_label(value) else "Benign"
    )
    return result


def run_training(data_path: str, max_rows: int, include_merged: bool, test_size: float) -> tuple[int, str]:
    command = [
        sys.executable,
        str(ROOT / "model" / "train_model.py"),
        "--data",
        data_path,
        "--max-rows-per-file",
        str(max_rows),
        "--test-size",
        str(test_size),
    ]
    if include_merged:
        command.append("--include-merged")

    process = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=None,
    )
    output = process.stdout
    if process.stderr:
        output = f"{output}\n\nSTDERR:\n{process.stderr}"
    return process.returncode, output


def build_training_command(data_path: str, max_rows: int, include_merged: bool, test_size: float) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "model" / "train_model.py"),
        "--data",
        data_path,
        "--max-rows-per-file",
        str(max_rows),
        "--test-size",
        str(test_size),
    ]
    if include_merged:
        command.append("--include-merged")
    return command


def load_training_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def save_training_history(history_path: Path, history: list[dict[str, Any]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history[-50:], indent=2), encoding="utf-8")


def append_training_history(history_path: Path, record: dict[str, Any]) -> None:
    history = load_training_history(history_path)
    history.append(record)
    save_training_history(history_path, history)


def run_training_with_live_status(
    data_path: str,
    max_rows: int,
    include_merged: bool,
    test_size: float,
    estimate: dict[str, Any],
    history_path: Path,
) -> tuple[int, str]:
    command = build_training_command(data_path, max_rows, include_merged, test_size)
    started = datetime.now()
    started_monotonic = time.monotonic()

    status_box = st.empty()
    progress_box = st.empty()
    metrics_box = st.empty()

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    high_seconds = max(int(estimate.get("high_seconds", 1)), 1)
    while process.poll() is None:
        elapsed = int(time.monotonic() - started_monotonic)
        remaining = max(high_seconds - elapsed, 0)
        progress = min(elapsed / high_seconds, 0.98)

        with metrics_box.container():
            live1, live2, live3, live4 = st.columns(4)
            live1.metric("Elapsed time", format_duration(elapsed))
            live2.metric("Expected range", f"{format_duration(estimate['low_seconds'])} - {format_duration(estimate['high_seconds'])}")
            live3.metric("ETA remaining", format_duration(remaining) if remaining else "finalizing")
            live4.metric("Training status", "Running")
        progress_box.progress(progress, text=f"Training in progress: {format_duration(elapsed)} elapsed")
        status_box.info("Training is running. Keep this browser tab open until the result appears.")
        time.sleep(1)

    output, _ = process.communicate()
    ended = datetime.now()
    elapsed_seconds = int(time.monotonic() - started_monotonic)
    code = process.returncode

    progress_box.progress(1.0, text=f"Training finished in {format_duration(elapsed_seconds)}")
    with metrics_box.container():
        done1, done2, done3, done4 = st.columns(4)
        done1.metric("Actual training time", format_duration(elapsed_seconds))
        done2.metric("Expected range", f"{format_duration(estimate['low_seconds'])} - {format_duration(estimate['high_seconds'])}")
        done3.metric("Exit code", code)
        done4.metric("Training status", "Success" if code == 0 else "Failed")

    append_training_history(
        history_path,
        {
            "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": ended.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed_seconds,
            "elapsed": format_duration(elapsed_seconds),
            "expected_low_seconds": estimate["low_seconds"],
            "expected_high_seconds": estimate["high_seconds"],
            "expected_range": f"{format_duration(estimate['low_seconds'])} - {format_duration(estimate['high_seconds'])}",
            "status": "success" if code == 0 else "failed",
            "exit_code": code,
            "data_path": data_path,
            "max_rows_per_file": max_rows,
            "include_merged": include_merged,
            "test_size": test_size,
            "csv_files": len(estimate["files"]),
            "estimated_rows_used": estimate["total_rows"],
            "command": " ".join(command),
        },
    )
    return code, output or ""


def training_log_text(log_path: Path, limit: int = 12000) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def start_background_training(
    data_path: str,
    max_rows: int,
    include_merged: bool,
    test_size: float,
    estimate: dict[str, Any],
    history_path: Path,
) -> None:
    command = build_training_command(data_path, max_rows, include_merged, test_size)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = history_path.parent / "active_training.log"
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log_file.close()
    started = datetime.now()
    st.session_state["active_training"] = {
        "process": process,
        "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "started_monotonic": time.monotonic(),
        "data_path": data_path,
        "max_rows_per_file": max_rows,
        "include_merged": include_merged,
        "test_size": test_size,
        "estimate": estimate,
        "history_path": str(history_path),
        "log_path": str(log_path),
        "command": " ".join(command),
    }


def training_record(active: dict[str, Any], status: str, exit_code: int | None, elapsed_seconds: int) -> dict[str, Any]:
    estimate = active["estimate"]
    return {
        "started_at": active["started_at"],
        "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed_seconds,
        "elapsed": format_duration(elapsed_seconds),
        "expected_low_seconds": estimate["low_seconds"],
        "expected_high_seconds": estimate["high_seconds"],
        "expected_range": f"{format_duration(estimate['low_seconds'])} - {format_duration(estimate['high_seconds'])}",
        "status": status,
        "exit_code": exit_code,
        "data_path": active["data_path"],
        "max_rows_per_file": active["max_rows_per_file"],
        "include_merged": active["include_merged"],
        "test_size": active["test_size"],
        "csv_files": len(estimate["files"]),
        "estimated_rows_used": estimate["total_rows"],
        "command": active["command"],
    }


def stop_active_training() -> None:
    active = st.session_state.get("active_training")
    if not active:
        return

    process = active["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    elapsed = int(time.monotonic() - active["started_monotonic"])
    append_training_history(
        Path(active["history_path"]),
        training_record(active, "stopped", process.returncode, elapsed),
    )
    st.session_state.pop("active_training", None)


def complete_active_training_if_finished() -> tuple[bool, str]:
    active = st.session_state.get("active_training")
    if not active:
        return False, ""

    process = active["process"]
    exit_code = process.poll()
    if exit_code is None:
        return False, ""

    elapsed = int(time.monotonic() - active["started_monotonic"])
    status = "success" if exit_code == 0 else "failed"
    append_training_history(
        Path(active["history_path"]),
        training_record(active, status, exit_code, elapsed),
    )
    log_text = training_log_text(Path(active["log_path"]))
    st.session_state.pop("active_training", None)
    return True, log_text


def render_active_training() -> None:
    active = st.session_state.get("active_training")
    if not active:
        return

    process = active["process"]
    estimate = active["estimate"]
    elapsed = int(time.monotonic() - active["started_monotonic"])
    high_seconds = max(int(estimate.get("high_seconds", 1)), 1)
    remaining = max(high_seconds - elapsed, 0)
    progress = min(elapsed / high_seconds, 0.98)

    live1, live2, live3, live4 = st.columns(4)
    live1.metric("Elapsed time", format_duration(elapsed))
    live2.metric("Expected range", f"{format_duration(estimate['low_seconds'])} - {format_duration(estimate['high_seconds'])}")
    live3.metric("ETA remaining", format_duration(remaining) if remaining else "finalizing")
    live4.metric("Training PID", process.pid)
    st.progress(progress, text=f"Training in progress: {format_duration(elapsed)} elapsed")
    st.code(training_log_text(Path(active["log_path"]), limit=6000) or "Training started. Waiting for log output...", language="text")

    if st.button("Stop training", type="secondary"):
        stop_active_training()
        st.warning("Training stopped. A stopped run was saved in Training History.")
        st.rerun()

    time.sleep(1)
    st.rerun()


def card(label: str, value: str, subtle: str = "") -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div class="status-subtle">{subtle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.title("NIDS Command Center")
    st.caption("Enterprise-style ML dashboard for CSV, PKL, training, prediction, and reporting.")

    artifacts_dir = Path(st.text_input("Artifacts folder", str(ARTIFACTS_DEFAULT)))
    dataset_dir = Path(st.text_input("Dataset folder", str(DATASET_DEFAULT)))
    model_path = Path(st.text_input("Default PKL model", str(MODEL_DEFAULT)))

    st.divider()
    sample_rows = st.slider("CSV preview rows", 100, 10000, 1000, step=100)
    if st.button("Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("Local paths")
    st.write(f"Root: `{ROOT}`")
    st.write(f"Artifacts: `{rel(artifacts_dir)}`")


metadata = load_json(str(artifacts_dir / "training_metadata.json"))
comparison = load_csv(str(artifacts_dir / "model_comparison.csv"))
metrics = discover_metrics(artifacts_dir)
model_package = load_pickle_model(str(model_path))
summary = package_summary(model_package)
csv_files = discover_csv_files(dataset_dir)
pkl_files = discover_pkl_files(artifacts_dir)
history_path = artifacts_dir / "training_history.json"
training_history = load_training_history(history_path)


st.markdown(
    """
    <div class="hero">
        <h1>Enterprise Network Intrusion Detection ML Dashboard</h1>
        <p>Operational view for dataset inspection, model registry, training control, prediction, and security analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


top1, top2, top3, top4 = st.columns(4)
with top1:
    card("Model status", "Loaded" if model_package else "Missing", rel(model_path))
with top2:
    accuracy = summary.get("accuracy")
    card("Best accuracy", f"{accuracy:.2%}" if isinstance(accuracy, float) else "N/A", summary.get("model_name", "No model"))
with top3:
    rows = metadata.get("training_rows") if metadata else None
    card("Training rows", f"{rows:,}" if isinstance(rows, int) else "N/A", "from training metadata")
with top4:
    card("Dataset CSV files", str(len(csv_files)), rel(dataset_dir))


tabs = st.tabs(
    [
        "Executive Overview",
        "CSV Data Explorer",
        "PKL Model Registry",
        "Model Graphs",
        "Train New Data",
        "Training History",
        "Prediction Studio",
        "Custom Row Scoring",
    ]
)


with tabs[0]:
    st.subheader("Operational summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("PKL artifacts", len(pkl_files))
    c2.metric("Metric reports", len(metrics))
    c3.metric("Feature count", summary.get("feature_count", "N/A"))

    st.subheader("Dashboard result")
    model_ready = bool(model_package)
    metadata_ready = bool(metadata)
    comparison_ready = comparison is not None and not comparison.empty
    metrics_ready = bool(metrics)
    dataset_ready = bool(csv_files)
    ready_score = sum([model_ready, metadata_ready, comparison_ready, metrics_ready, dataset_ready])
    result_cols = st.columns(4)
    result_cols[0].metric("Readiness score", f"{ready_score}/5")
    result_cols[1].metric("Model registry", "Ready" if model_ready else "Missing")
    result_cols[2].metric("Training reports", "Ready" if metrics_ready else "Missing")
    result_cols[3].metric("Dataset access", "Ready" if dataset_ready else "Missing")

    checklist = pd.DataFrame(
        [
            {"Check": "PKL model can be loaded", "Status": "Pass" if model_ready else "Missing", "Details": rel(model_path)},
            {"Check": "Training metadata exists", "Status": "Pass" if metadata_ready else "Missing", "Details": rel(artifacts_dir / "training_metadata.json")},
            {"Check": "Model comparison exists", "Status": "Pass" if comparison_ready else "Missing", "Details": rel(artifacts_dir / "model_comparison.csv")},
            {"Check": "Metrics JSON files exist", "Status": "Pass" if metrics_ready else "Missing", "Details": f"{len(metrics)} report(s)"},
            {"Check": "Dataset CSV files visible", "Status": "Pass" if dataset_ready else "Missing", "Details": f"{len(csv_files)} CSV file(s)"},
        ]
    )
    st.dataframe(checklist, use_container_width=True, hide_index=True)

    if pkl_files:
        st.subheader("Artifact inventory")
        artifact_rows = [
            {
                "file": file.name,
                "size": file_size(file),
                "modified": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
            for file in pkl_files
        ]
        st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)

    if metadata:
        st.markdown('<div class="section-note">Training metadata is available. This dashboard is reading the latest local training artifacts.</div>', unsafe_allow_html=True)
        st.json(metadata, expanded=False)
    else:
        st.warning("No training_metadata.json found. Train a model from the Train New Data tab.")

    if summary.get("labels"):
        labels = pd.DataFrame({"Model labels": summary["labels"]})
        st.dataframe(labels, use_container_width=True, height=260)


with tabs[1]:
    st.subheader("CSV data explorer")
    upload_csv = st.file_uploader("Upload CSV for inspection", type=["csv"], key="csv_inspect")

    selected_path: Path | None = None
    if csv_files:
        selected_path = st.selectbox(
            "Or select local dataset CSV",
            csv_files,
            format_func=lambda path: f"{path.name} ({file_size(path)})",
        )

    data_df = load_uploaded_csv(upload_csv)
    if data_df is None and selected_path:
        data_df = load_csv(str(selected_path), nrows=sample_rows)

    if data_df is None:
        st.info("Upload a CSV or place CSV files inside the dataset folder.")
    else:
        st.caption(f"Previewing {len(data_df):,} rows and {len(data_df.columns):,} columns.")
        st.dataframe(data_df.head(sample_rows), use_container_width=True, height=360)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rows loaded", f"{len(data_df):,}")
        s2.metric("Columns", f"{len(data_df.columns):,}")
        s3.metric("Missing cells", f"{int(data_df.isna().sum().sum()):,}")
        s4.metric("Duplicate rows", f"{int(data_df.duplicated().sum()):,}")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Schema")
            schema = pd.DataFrame(
                {
                    "column": data_df.columns,
                    "dtype": [str(dtype) for dtype in data_df.dtypes],
                    "missing": data_df.isna().sum().values,
                }
            )
            st.dataframe(schema, use_container_width=True, height=300)
        with right:
            if TARGET_COLUMN in data_df.columns:
                st.subheader("Label distribution")
                label_counts = data_df[TARGET_COLUMN].astype(str).value_counts().reset_index()
                label_counts.columns = ["Label", "Count"]
                fig = px.bar(label_counts, x="Label", y="Count", color="Label")
                fig.update_layout(showlegend=False, xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader("Numeric profile")
                numeric_cols = data_df.select_dtypes(include="number").columns.tolist()
                if numeric_cols:
                    col = st.selectbox("Numeric column", numeric_cols)
                    fig = px.histogram(data_df, x=col, nbins=50)
                    st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download current preview as CSV",
            data_df.head(sample_rows).to_csv(index=False).encode("utf-8"),
            file_name="nids_csv_preview.csv",
            mime="text/csv",
        )


with tabs[2]:
    st.subheader("PKL model registry")
    if not pkl_files:
        st.info("No `.pkl` files found in the artifacts folder.")
    else:
        selected_pkl = st.selectbox(
            "Select PKL artifact",
            pkl_files,
            index=pkl_files.index(model_path) if model_path in pkl_files else 0,
            format_func=lambda path: f"{path.name} ({file_size(path)})",
        )
        selected_package = load_pickle_model(str(selected_pkl))
        selected_summary = package_summary(selected_package)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("File", selected_pkl.name)
        p2.metric("Size", file_size(selected_pkl))
        p3.metric("Model", selected_summary.get("model_name", "unknown"))
        p4.metric(
            "Accuracy",
            f"{selected_summary.get('accuracy'):.2%}"
            if isinstance(selected_summary.get("accuracy"), float)
            else "N/A",
        )

        st.subheader("Artifact metadata")
        st.json(selected_summary.get("metadata", {}), expanded=False)

        feature_columns = selected_package.get("feature_columns", []) if selected_package else []
        if feature_columns:
            st.subheader("Feature columns")
            st.dataframe(pd.DataFrame({"feature": feature_columns}), use_container_width=True, height=300)


with tabs[3]:
    st.subheader("Model performance graphs")
    if comparison is not None and not comparison.empty:
        comp = comparison.sort_values("accuracy", ascending=False)
        fig = px.bar(comp, x="model", y="accuracy", text_auto=".2%", color="accuracy", color_continuous_scale="Blues")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comp, use_container_width=True)
    else:
        st.info("model_comparison.csv not found. Train first.")

    if metadata and metadata.get("class_distribution"):
        st.subheader("Training class distribution")
        dist = pd.DataFrame(list(metadata["class_distribution"].items()), columns=["Label", "Count"])
        dist["Traffic_Type"] = dist["Label"].apply(lambda value: "Attack" if is_attack_label(value) else "Benign")
        fig = px.bar(dist.sort_values("Count", ascending=False), x="Label", y="Count", color="Traffic_Type", log_y=True)
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    if metrics:
        st.subheader("Confusion matrix and report")
        selected_metric = st.selectbox("Metric file", list(metrics.keys()))
        report = metrics[selected_metric]
        labels = report.get("labels", [])
        matrix = report.get("confusion_matrix", [])
        if labels and matrix:
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
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        class_report = report.get("classification_report", {})
        if class_report:
            report_df = pd.DataFrame(class_report).T.drop(index=["accuracy"], errors="ignore")
            st.dataframe(report_df, use_container_width=True)


with tabs[4]:
    st.subheader("Train model on new data")
    st.markdown(
        '<div class="section-note">This runs the existing ML training script. It may take time on large CSV files. Artifacts are written to model/artifacts.</div>',
        unsafe_allow_html=True,
    )
    train_data_path = st.text_input("Training data path", str(dataset_dir), key="train_path")
    max_rows = st.number_input("Max rows per file", min_value=0, max_value=1_000_000, value=50_000, step=10_000)
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    include_merged = st.checkbox("Include Merged.csv", value=False)

    estimate = training_estimate(train_data_path, int(max_rows), include_merged)
    st.subheader("Expected training time")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("CSV files used", len(estimate["files"]))
    e2.metric("Estimated rows used", f"{estimate['total_rows']:,}")
    e3.metric("Expected time", f"{format_duration(estimate['low_seconds'])} - {format_duration(estimate['high_seconds'])}")
    e4.metric("Models trained", "3")
    st.caption("Estimate is based on CSV size, selected row limit, and three sklearn models. Actual time depends on CPU, RAM, and disk speed.")
    if estimate["files"]:
        st.dataframe(pd.DataFrame(estimate["files"]), use_container_width=True, hide_index=True)
    else:
        st.warning("No CSV files found for the selected training path.")

    finished, finished_log = complete_active_training_if_finished()
    if finished:
        st.success("Training finished. Refresh dashboard to reload artifacts and history.")
        st.code(finished_log, language="text")

    if st.session_state.get("active_training"):
        st.subheader("Running training job")
        render_active_training()
    elif st.button("Start training", type="primary"):
        start_background_training(
            train_data_path,
            int(max_rows),
            include_merged,
            float(test_size),
            estimate,
            history_path,
        )
        st.rerun()


with tabs[5]:
    st.subheader("Last training data history")
    if not training_history:
        st.info("No training runs recorded yet. Start training once from the Train New Data tab.")
    else:
        history_df = pd.DataFrame(training_history).sort_values("started_at", ascending=False)
        h1, h2, h3, h4 = st.columns(4)
        last_run = history_df.iloc[0].to_dict()
        h1.metric("Total runs", len(history_df))
        h2.metric("Last status", str(last_run.get("status", "unknown")).title())
        h3.metric("Last actual time", last_run.get("elapsed", "N/A"))
        h4.metric("Last rows used", f"{int(last_run.get('estimated_rows_used', 0)):,}")

        status_counts = history_df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Runs"]
        left, right = st.columns([1, 2])
        with left:
            fig = px.pie(status_counts, names="Status", values="Runs", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            chart_df = history_df.copy()
            chart_df["elapsed_minutes"] = chart_df["elapsed_seconds"] / 60
            fig = px.bar(
                chart_df.head(15),
                x="started_at",
                y="elapsed_minutes",
                color="status",
                hover_data=["data_path", "max_rows_per_file", "estimated_rows_used", "expected_range"],
                title="Recent training run duration",
            )
            fig.update_layout(xaxis_tickangle=-35, yaxis_title="Minutes")
            st.plotly_chart(fig, use_container_width=True)

        visible_cols = [
            "started_at",
            "ended_at",
            "status",
            "elapsed",
            "expected_range",
            "estimated_rows_used",
            "csv_files",
            "max_rows_per_file",
            "include_merged",
            "test_size",
            "data_path",
        ]
        st.dataframe(history_df[[col for col in visible_cols if col in history_df.columns]], use_container_width=True, hide_index=True)
        st.download_button(
            "Download training history JSON",
            json.dumps(training_history, indent=2).encode("utf-8"),
            file_name="training_history.json",
            mime="application/json",
        )


with tabs[6]:
    st.subheader("Prediction studio")
    if not model_package:
        st.warning("Load or train a PKL model first.")
    else:
        pred_upload = st.file_uploader("Upload CSV to score", type=["csv"], key="pred_upload")
        if pred_upload is not None:
            raw_pred = pd.read_csv(pred_upload, low_memory=False)
            with st.spinner("Scoring uploaded CSV"):
                scored = predict_dataframe(model_package, raw_pred)
            total = len(scored)
            attacks = int((scored["Traffic_Type"] == "Attack").sum())
            q1, q2, q3 = st.columns(3)
            q1.metric("Rows scored", f"{total:,}")
            q2.metric("Attack rows", f"{attacks:,}")
            q3.metric("Attack rate", f"{attacks / total:.2%}" if total else "N/A")

            counts = scored["Predicted_Label"].value_counts().reset_index()
            counts.columns = ["Predicted_Label", "Count"]
            left, right = st.columns([1, 1])
            with left:
                st.plotly_chart(px.pie(counts, names="Predicted_Label", values="Count", hole=0.45), use_container_width=True)
            with right:
                st.plotly_chart(px.bar(counts, x="Predicted_Label", y="Count", color="Predicted_Label"), use_container_width=True)

            attack_labels = sorted(scored.loc[scored["Traffic_Type"] == "Attack", "Predicted_Label"].unique())
            selected_attack_labels = st.multiselect("Filter flagged labels", attack_labels, default=attack_labels)
            flagged = scored[scored["Predicted_Label"].isin(selected_attack_labels)] if selected_attack_labels else scored.iloc[0:0]
            st.dataframe(flagged, use_container_width=True, height=340)
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False).encode("utf-8"),
                file_name=f"nids_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("Upload a CSV file to score with the selected PKL model.")


with tabs[7]:
    st.subheader("Custom single-row scoring")
    if not model_package:
        st.warning("Load or train a PKL model first.")
    else:
        feature_columns = model_package.get("feature_columns", [])
        if not feature_columns:
            st.warning("The PKL model does not contain feature_columns metadata.")
        else:
            st.caption("Edit one row of feature values. Missing or unchanged fields default to 0.")
            default_row = pd.DataFrame([{column: 0.0 for column in feature_columns}])
            edited = st.data_editor(default_row, use_container_width=True, num_rows="fixed", height=300)
            if st.button("Score custom row", type="primary"):
                scored_row = predict_dataframe(model_package, edited)
                label = scored_row.loc[0, "Predicted_Label"]
                traffic_type = scored_row.loc[0, "Traffic_Type"]
                if traffic_type == "Attack":
                    st.error(f"Predicted label: {label}")
                else:
                    st.success(f"Predicted label: {label}")
                st.dataframe(scored_row, use_container_width=True)
