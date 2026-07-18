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
    :root {
        --bg: #070b12;
        --panel: #0e1624;
        --panel-2: #111c2d;
        --border: #22314a;
        --border-soft: rgba(148, 163, 184, 0.18);
        --text: #edf3fb;
        --muted: #9caabc;
        --accent: #38bdf8;
        --accent-2: #14b8a6;
        --danger: #fb7185;
        --warning: #fbbf24;
        --success: #22c55e;
        --shadow: 0 18px 44px rgba(0, 0, 0, 0.30);
    }

    .stApp {
        background:
            radial-gradient(circle at 18% 0%, rgba(56, 189, 248, 0.13), transparent 28%),
            radial-gradient(circle at 82% 8%, rgba(20, 184, 166, 0.10), transparent 32%),
            linear-gradient(180deg, #080d16 0%, #070b12 45%, #05070d 100%);
        color: var(--text);
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(9, 14, 25, 0.98) 100%);
        border-right: 1px solid var(--border);
        box-shadow: 16px 0 34px rgba(0, 0, 0, 0.24);
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: .35rem;
    }

    [data-testid="stSidebar"] .stCaption {
        color: var(--muted);
    }

    .block-container {
        padding-top: 1.45rem;
        padding-bottom: 3.5rem;
        max-width: 1520px;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: #f8fafc;
        font-weight: 800;
    }

    p, label, span, div {
        letter-spacing: 0;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 34px;
        border: 1px solid var(--border);
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(13, 28, 48, 0.94) 48%, rgba(8, 13, 22, 0.98) 100%);
        border-radius: 14px;
        box-shadow: var(--shadow);
        margin-bottom: 22px;
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        pointer-events: none;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -120px;
        top: -90px;
        width: 360px;
        height: 360px;
        background: radial-gradient(circle, rgba(56, 189, 248, .20), transparent 62%);
        pointer-events: none;
    }

    .hero h1 {
        margin: 0 0 8px 0;
        font-size: clamp(30px, 3vw, 44px);
        line-height: 1.15;
        max-width: 1040px;
    }

    .hero p {
        margin: 0;
        color: #b7c4d6;
        font-size: 16px;
        max-width: 920px;
    }

    .status-card {
        padding: 18px 19px;
        border: 1px solid var(--border);
        background:
            linear-gradient(180deg, rgba(17, 28, 45, 0.98) 0%, rgba(11, 18, 31, 0.98) 100%);
        border-radius: 12px;
        min-height: 120px;
        box-shadow: 0 12px 26px rgba(0, 0, 0, .20);
        position: relative;
        overflow: hidden;
    }

    .status-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--accent), var(--accent-2));
    }

    .status-label {
        color: #9fb0c6;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .10em;
        margin-bottom: 10px;
        font-weight: 700;
    }

    .status-value {
        color: var(--text);
        font-size: 28px;
        font-weight: 850;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .status-subtle {
        color: #93a4b8;
        font-size: 13px;
        margin-top: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .section-note {
        padding: 14px 16px;
        border: 1px solid rgba(56, 189, 248, 0.28);
        background: linear-gradient(90deg, rgba(14, 116, 144, 0.18), rgba(15, 23, 42, 0.68));
        border-radius: 10px;
        color: #d7e4f4;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, .04);
    }

    div[data-testid="stMetric"] {
        background:
            linear-gradient(180deg, rgba(17, 28, 45, .96), rgba(10, 17, 29, .96));
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 15px 16px;
        box-shadow: 0 10px 22px rgba(0, 0, 0, .18);
    }

    div[data-testid="stMetric"] label {
        color: #a8b6c8 !important;
        font-size: 12px !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 850;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
        margin-bottom: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 16px;
        border: 1px solid var(--border-soft);
        border-radius: 999px;
        background: rgba(15, 23, 42, .64);
        color: #cbd5e1;
    }

    .stTabs [aria-selected="true"] {
        border-color: rgba(56, 189, 248, .65);
        background: linear-gradient(90deg, rgba(14, 165, 233, .20), rgba(20, 184, 166, .16));
        color: #f8fafc;
    }

    div[data-testid="stTabs"] button p {
        font-size: 14px;
        font-weight: 600;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(56, 189, 248, .35);
        background: linear-gradient(135deg, #0ea5e9, #0f766e);
        color: white;
        font-weight: 800;
        min-height: 42px;
        box-shadow: 0 10px 22px rgba(14, 165, 233, .18);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(125, 211, 252, .75);
        filter: brightness(1.07);
        transform: translateY(-1px);
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(6, 11, 20, .92) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
        min-height: 42px;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: rgba(56, 189, 248, .78) !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, .12) !important;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 12px 24px rgba(0, 0, 0, .16);
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(56, 189, 248, .38);
        background: rgba(15, 23, 42, .55);
        border-radius: 12px;
        padding: 10px;
    }

    .stAlert {
        border-radius: 12px;
        border: 1px solid var(--border-soft);
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }

    .event-card {
        padding: 14px 16px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(15, 23, 42, .72);
        margin: .7rem 0 1rem 0;
    }

    .event-title {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: .35rem;
    }

    .event-detail {
        color: #b8c6d8;
        font-size: 14px;
        line-height: 1.55;
    }

    .confirm-card {
        padding: 18px 20px;
        border: 1px solid rgba(251, 191, 36, .42);
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(120, 83, 20, .30), rgba(15, 23, 42, .78));
        box-shadow: 0 14px 30px rgba(0, 0, 0, .22);
        margin: 1rem 0;
    }

    .confirm-card strong {
        color: #fde68a;
    }

    hr {
        border-color: var(--border);
    }

    /* Layout rhythm */
    .main .block-container {
        padding-left: clamp(22px, 3vw, 46px);
        padding-right: clamp(22px, 3vw, 46px);
    }

    [data-testid="stVerticalBlock"] {
        gap: 1.05rem;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
        align-items: stretch;
    }

    [data-testid="column"] {
        padding: 0 .25rem;
    }

    [data-testid="column"] > div {
        height: 100%;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.35rem 1.5rem 1.35rem;
    }

    [data-testid="stSidebar"] label {
        margin-top: .95rem;
        margin-bottom: .35rem;
        color: #d7e3f2 !important;
        font-size: 13px !important;
        font-weight: 750 !important;
    }

    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] hr {
        margin: 1.45rem 0 1.25rem 0;
    }

    .element-container {
        margin-bottom: .35rem;
    }

    h2 {
        margin-top: 1.45rem;
        margin-bottom: .9rem;
        font-size: 1.45rem;
    }

    h3 {
        margin-top: 1.1rem;
        margin-bottom: .75rem;
        font-size: 1.12rem;
    }

    .stMarkdown p {
        color: #c5d1df;
        line-height: 1.62;
    }

    .hero + div[data-testid="stHorizontalBlock"] {
        margin-top: .85rem;
        margin-bottom: 2.75rem;
        row-gap: 1.35rem;
    }

    .main div[data-testid="stHorizontalBlock"]:has(.status-card) {
        margin-bottom: 0.2cm;
        row-gap: 1.4rem;
    }

    .status-card {
        margin-bottom: 0.2cm;
    }

    .stTabs {
        margin-top: 0.2cm;
        clear: both;
    }

    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        row-gap: 10px;
        padding-top: 0.2cm;
        margin-top: .35rem;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.15rem;
    }

    div[data-testid="stMetric"] {
        min-height: 108px;
        margin-bottom: .8rem;
    }

    div[data-testid="stMetricValue"] {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100%;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        margin-top: .55rem;
        margin-bottom: 1.25rem;
    }

    [data-testid="stFileUploader"] {
        margin-top: .5rem;
        margin-bottom: 1.15rem;
    }

    .stTextInput,
    .stNumberInput,
    .stSelectbox,
    .stSlider,
    .stCheckbox,
    .stMultiSelect {
        margin-bottom: .9rem;
    }

    .stButton,
    .stDownloadButton {
        margin-top: .45rem;
        margin-bottom: .8rem;
    }

    .stButton > button, .stDownloadButton > button {
        padding: .62rem 1rem;
    }

    .stAlert {
        margin-top: .75rem;
        margin-bottom: 1rem;
    }

    .js-plotly-plot {
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        overflow: hidden;
        margin: .45rem 0 1.25rem 0;
        background: rgba(15, 23, 42, .36);
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(15, 23, 42, .52);
        margin: .75rem 0 1.1rem 0;
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding-left: 16px;
            padding-right: 16px;
            padding-top: 1rem;
        }

        .hero {
            padding: 22px 20px;
            border-radius: 12px;
        }

        .hero h1 {
            font-size: 28px;
        }

        .status-card {
            min-height: 104px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 0 12px;
        }
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


def init_session_state() -> None:
    st.session_state.setdefault("ui_events", [])
    st.session_state.setdefault("pending_training", None)
    st.session_state.setdefault("pending_stop_training", False)


def push_event(level: str, title: str, detail: str = "") -> None:
    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level.upper(),
        "title": title,
        "detail": detail,
    }
    st.session_state["ui_events"] = [event, *st.session_state.get("ui_events", [])][:30]

    icon = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(level.lower(), "ℹ️")
    try:
        st.toast(f"{title}{': ' + detail if detail else ''}", icon=icon)
    except Exception:
        pass


def render_event_card(title: str, detail: str, level: str = "info") -> None:
    color = {
        "success": "#22c55e",
        "error": "#fb7185",
        "warning": "#fbbf24",
        "info": "#38bdf8",
    }.get(level, "#38bdf8")
    st.markdown(
        f"""
        <div class="event-card" style="border-left: 4px solid {color};">
            <div class="event-title">{title}</div>
            <div class="event-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_training_request(data_path: str, estimate: dict[str, Any]) -> list[str]:
    errors = []
    if st.session_state.get("active_training"):
        errors.append("Another training job is already running.")
    if not data_path.strip():
        errors.append("Training data path is empty.")
    if not estimate["files"]:
        errors.append("No CSV files found for the selected training path.")
    if estimate["total_rows"] <= 0:
        errors.append("Estimated selected rows is zero. Check dataset files or row limit.")
    return errors


def render_recent_events(limit: int = 8) -> None:
    events = st.session_state.get("ui_events", [])
    if not events:
        st.info("No dashboard events yet. Click actions such as Refresh, Start training, Stop training, or Score to see events here.")
        return
    st.dataframe(pd.DataFrame(events[:limit]), use_container_width=True, hide_index=True)


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
        st.session_state["pending_stop_training"] = True
        push_event("warning", "Stop requested", "Confirm stop to terminate the active training process.")
        st.rerun()

    if st.session_state.get("pending_stop_training"):
        st.markdown(
            """
            <div class="confirm-card">
                <strong>Confirm stop training</strong><br>
                This will terminate the running training process and save the run as stopped in Training History.
            </div>
            """,
            unsafe_allow_html=True,
        )
        confirm_col, cancel_col = st.columns([1, 1])
        if confirm_col.button("Confirm stop", type="primary"):
            stop_active_training()
            st.session_state["pending_stop_training"] = False
            push_event("warning", "Training stopped", "Stopped run saved in Training History.")
            st.rerun()
        if cancel_col.button("Cancel stop"):
            st.session_state["pending_stop_training"] = False
            push_event("info", "Stop cancelled", "Training continues.")
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


init_session_state()


with st.sidebar:
    st.title("NIDS Command Center")
    st.caption("Enterprise-style ML dashboard for CSV, PKL, training, prediction, and reporting.")

    artifacts_dir = Path(st.text_input("Artifacts folder", str(ARTIFACTS_DEFAULT)))
    dataset_dir = Path(st.text_input("Dataset folder", str(DATASET_DEFAULT)))
    model_path = Path(st.text_input("Default PKL model", str(MODEL_DEFAULT)))

    st.divider()
    sample_rows = st.slider("CSV preview rows", 100, 10000, 1000, step=100)
    if st.button("Refresh dashboard", use_container_width=True):
        push_event("info", "Dashboard refreshed", "Cache cleared and dashboard reloaded.")
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
        "Event Center",
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
        errors = validate_training_request(train_data_path, estimate)
        if errors:
            for error in errors:
                st.error(error)
            push_event("error", "Training validation failed", " | ".join(errors))
        else:
            st.session_state["pending_training"] = {
                "data_path": train_data_path,
                "max_rows": int(max_rows),
                "include_merged": include_merged,
                "test_size": float(test_size),
                "estimate": estimate,
            }
            push_event("info", "Training start requested", f"{estimate['total_rows']:,} rows selected.")
        st.rerun()

    pending_training = st.session_state.get("pending_training")
    if pending_training:
        st.markdown(
            f"""
            <div class="confirm-card">
                <strong>Confirm model training</strong><br>
                Dataset path: {pending_training['data_path']}<br>
                Estimated rows: {pending_training['estimate']['total_rows']:,}<br>
                Expected time: {format_duration(pending_training['estimate']['low_seconds'])} - {format_duration(pending_training['estimate']['high_seconds'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
        confirm_col, cancel_col = st.columns([1, 1])
        if confirm_col.button("Confirm start training", type="primary"):
            start_background_training(
                pending_training["data_path"],
                pending_training["max_rows"],
                pending_training["include_merged"],
                pending_training["test_size"],
                pending_training["estimate"],
                history_path,
            )
            st.session_state["pending_training"] = None
            push_event("success", "Training started", "Live timer and Stop button are now active.")
            st.rerun()
        if cancel_col.button("Cancel training"):
            st.session_state["pending_training"] = None
            push_event("info", "Training cancelled", "No process was started.")
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
            try:
                raw_pred = pd.read_csv(pred_upload, low_memory=False)
                if raw_pred.empty:
                    st.error("Uploaded CSV is empty.")
                    push_event("error", "Prediction failed", "Uploaded CSV is empty.")
                else:
                    with st.spinner("Scoring uploaded CSV"):
                        scored = predict_dataframe(model_package, raw_pred)
                    total = len(scored)
                    attacks = int((scored["Traffic_Type"] == "Attack").sum())
                    push_event("success", "Prediction completed", f"{total:,} rows scored, {attacks:,} attack rows flagged.")
                    render_event_card("Prediction completed", f"{total:,} rows scored. Attack rows flagged: {attacks:,}.", "success")
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
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                push_event("error", "Prediction failed", str(exc))
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
                try:
                    scored_row = predict_dataframe(model_package, edited)
                    label = scored_row.loc[0, "Predicted_Label"]
                    traffic_type = scored_row.loc[0, "Traffic_Type"]
                    if traffic_type == "Attack":
                        st.error(f"Predicted label: {label}")
                        push_event("warning", "Custom row flagged", f"Predicted label: {label}")
                        render_event_card("Custom row flagged", f"Predicted label: {label}", "warning")
                    else:
                        st.success(f"Predicted label: {label}")
                        push_event("success", "Custom row scored", f"Predicted label: {label}")
                        render_event_card("Custom row scored", f"Predicted label: {label}", "success")
                    st.dataframe(scored_row, use_container_width=True)
                except Exception as exc:
                    st.error(f"Custom scoring failed: {exc}")
                    push_event("error", "Custom scoring failed", str(exc))


with tabs[8]:
    st.subheader("Dashboard event center")
    render_event_card(
        "Click event and error monitor",
        "This panel records important dashboard actions: refresh, train start, train stop, prediction, custom scoring, validation errors, and failures.",
        "info",
    )
    render_recent_events(limit=30)
    if st.button("Clear event log"):
        st.session_state["ui_events"] = []
        push_event("info", "Event log cleared", "Recent dashboard events were removed.")
        st.rerun()
