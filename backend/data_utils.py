from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMN = "Label"
DROP_COLUMNS = ["Day"]


def list_csv_files(data_path: str | Path, include_merged: bool = False) -> list[Path]:
    path = Path(data_path)
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    if not include_merged and len(files) > 1:
        files = [file for file in files if file.name.lower() != "merged.csv"]
    return files


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    df = normalize_columns(df)
    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}")

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    df = df.replace([np.inf, -np.inf], np.nan)

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
        df = df[df[TARGET_COLUMN].notna()]
        df = df[df[TARGET_COLUMN] != ""]

    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def read_csv_sample(file_path: Path, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0:
        return pd.read_csv(file_path, low_memory=False)

    chunks: list[pd.DataFrame] = []
    remaining = max_rows
    for chunk in pd.read_csv(file_path, chunksize=100_000, low_memory=False):
        take = min(len(chunk), remaining)
        if take < len(chunk):
            chunk = chunk.sample(n=take, random_state=random_state)
        chunks.append(chunk)
        remaining -= take
        if remaining <= 0:
            break

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_dataset(
    data_path: str | Path,
    max_rows_per_file: int | None = 50000,
    random_state: int = 42,
    include_merged: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_file in list_csv_files(data_path, include_merged=include_merged):
        print(f"Loading {csv_file.name} ...")
        frame = read_csv_sample(csv_file, max_rows_per_file, random_state)
        frame = clean_dataframe(frame, require_target=True)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates()
    return data


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing")
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return x, y


def align_prediction_columns(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    df = clean_dataframe(df, require_target=False)
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    feature_columns = list(feature_columns)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]
