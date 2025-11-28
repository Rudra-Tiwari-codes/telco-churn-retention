"""Tests for feature engineering module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.transformers import (
    CustomerLifetimeValueTransformer,
    RevenueSignalTransformer,
    ServiceCountTransformer,
    TenureBucketTransformer,
)


def test_tenure_bucket_transformer() -> None:
    """Test tenure bucket transformer."""
    df = pd.DataFrame({"tenure": [5, 15, 30, 50, 70]})
    transformer = TenureBucketTransformer()
    result = transformer.fit_transform(df)
    
    assert "tenure_bucket" in result.columns
    assert result["tenure_bucket"].iloc[0] == "0-12"
    assert result["tenure_bucket"].iloc[1] == "13-24"
    assert result["tenure_bucket"].iloc[2] == "25-36"


def test_service_count_transformer() -> None:
    """Test service count transformer."""
    df = pd.DataFrame({
        "OnlineSecurity": ["Yes", "No", "Yes"],
        "OnlineBackup": ["Yes", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "Yes"],
        "TechSupport": ["Yes", "No", "No"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "Yes"],
    })
    transformer = ServiceCountTransformer()
    result = transformer.fit_transform(df)
    
    assert "service_count" in result.columns
    assert result["service_count"].iloc[0] == 3  # 3 Yes values
    assert result["service_count"].iloc[1] == 3
    assert result["service_count"].iloc[2] == 2


def test_revenue_signal_transformer() -> None:
    """Test revenue signal transformer."""
    df = pd.DataFrame({
        "MonthlyCharges": [50.0, 100.0],
        "TotalCharges": [500.0, 1200.0],
        "tenure": [10, 12],
    })
    transformer = RevenueSignalTransformer()
    result = transformer.fit_transform(df)
    
    assert "charges_ratio" in result.columns
    assert "avg_monthly_charge" in result.columns
    assert "charge_increase_flag" in result.columns
    assert result["charges_ratio"].iloc[0] == pytest.approx(0.1, rel=1e-2)
    assert result["avg_monthly_charge"].iloc[0] == pytest.approx(50.0, rel=1e-2)


def test_clv_transformer() -> None:
    """Test customer lifetime value transformer."""
    df = pd.DataFrame({
        "TotalCharges": [500.0, 1200.0],
        "MonthlyCharges": [50.0, 100.0],
        "Contract": ["Month-to-month", "One year"],
    })
    transformer = CustomerLifetimeValueTransformer()
    result = transformer.fit_transform(df)
    
    assert "clv_approximation" in result.columns
    # CLV = TotalCharges + (MonthlyCharges * projection_months)
    # Month-to-month: 500 + (50 * 12) = 1100
    # One year: 1200 + (100 * 12) = 2400
    assert result["clv_approximation"].iloc[0] == pytest.approx(1100.0, rel=1e-2)
    assert result["clv_approximation"].iloc[1] == pytest.approx(2400.0, rel=1e-2)

