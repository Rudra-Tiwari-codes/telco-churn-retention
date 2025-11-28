"""Tests for data ingestion module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import IngestionConfig, clean_dataset, load_raw_dataset


def test_ingestion_config_snapshot_path(tmp_path: Path) -> None:
    """Test that IngestionConfig generates correct snapshot paths."""
    config = IngestionConfig(
        raw_path=Path("data/raw/test.csv"),
        processed_dir=tmp_path,
        snapshot_name="test.parquet",
    )
    snapshot_path = config.snapshot_path()
    assert snapshot_path.parent == tmp_path
    assert snapshot_path.name == "test.parquet"
    assert snapshot_path.parent.name  # timestamp directory exists


def test_clean_dataset() -> None:
    """Test dataset cleaning functionality."""
    df = pd.DataFrame(
        {
            "customerID": [" 1234-ABCD ", "5678-EFGH"],
            "TotalCharges": ["100.5", "200.3"],
            "SeniorCitizen": [0, 1],
        }
    )
    cleaned = clean_dataset(df)
    assert cleaned["customerID"].iloc[0] == "1234-ABCD"
    assert cleaned["TotalCharges"].dtype in ["float64", "Float64"]
    assert cleaned["SeniorCitizen"].dtype in ["Int64", "int64"]


def test_load_raw_dataset_nonexistent() -> None:
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(Path("nonexistent_file.csv"))
