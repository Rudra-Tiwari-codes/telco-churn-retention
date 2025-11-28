"""Tests for EDA module."""

from __future__ import annotations

import pandas as pd

from src.data.eda import (
    categorical_cardinality,
    churn_distribution,
    missingness_table,
    numeric_summary,
)


def test_churn_distribution() -> None:
    """Test churn distribution calculation."""
    df = pd.DataFrame({"Churn": ["Yes", "No", "No", "Yes", "No"]})
    result = churn_distribution(df)
    assert "Yes" in result
    assert "No" in result
    assert "Count" in result


def test_missingness_table() -> None:
    """Test missingness table generation."""
    df = pd.DataFrame({"col1": [1, 2, None], "col2": [1, 2, 3]})
    result = missingness_table(df)
    assert "col1" in result
    assert "0.333" in result or "0.3" in result


def test_numeric_summary() -> None:
    """Test numeric summary generation."""
    df = pd.DataFrame({"numeric_col": [1, 2, 3, 4, 5]})
    result = numeric_summary(df)
    assert "numeric_col" in result
    assert "Mean" in result


def test_categorical_cardinality() -> None:
    """Test categorical cardinality calculation."""
    df = pd.DataFrame({"cat_col": ["A", "B", "C", "A", "B"]})
    result = categorical_cardinality(df)
    assert "cat_col" in result
    assert "3" in result

