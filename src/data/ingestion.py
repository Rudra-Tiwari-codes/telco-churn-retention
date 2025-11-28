"""
Utilities for loading the Telco churn dataset and materializing processed snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class IngestionConfig:
    raw_path: Path
    processed_dir: Path
    snapshot_name: str = "telco_churn.parquet"
    snapshot_ts: str | None = None

    def snapshot_path(self) -> Path:
        ts = self.snapshot_ts or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return self.processed_dir / ts / self.snapshot_name


def load_raw_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("Int64")
    df = df.dropna(subset=["customerID"])
    df["customerID"] = df["customerID"].str.strip()
    return df


def persist_snapshot(df: pd.DataFrame, config: IngestionConfig) -> Path:
    snapshot_path = config.snapshot_path()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(snapshot_path, index=False)
    return snapshot_path


def run_ingestion(config: IngestionConfig) -> Path:
    raw_df = load_raw_dataset(config.raw_path)
    cleaned = clean_dataset(raw_df)
    return persist_snapshot(cleaned, config)
