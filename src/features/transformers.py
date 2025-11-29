"""
Custom feature transformers for telco churn dataset.

These transformers implement reusable feature engineering logic that can be
integrated into sklearn pipelines for both batch and streaming use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TenureBucketTransformer(BaseEstimator, TransformerMixin):
    """Transform tenure into categorical buckets.

    Creates tenure buckets based on predefined ranges:
    - 0-12: New customers (high churn risk)
    - 13-24: Early customers
    - 25-36: Established customers
    - 37-48: Loyal customers
    - 49-60: Very loyal customers
    - 61-72: Long-term customers
    - 73+: Veteran customers
    """

    def __init__(
        self,
        bins: list[float] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Initialize tenure bucket transformer.

        Args:
            bins: Bin edges for tenure buckets. Default: [0, 12, 24, 36, 48, 60, 72, 100]
            labels: Labels for each bucket. Default: ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72", "73+"]
        """
        self.bins = bins or [0, 12, 24, 36, 48, 60, 72, 100]
        self.labels = labels or ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72", "73+"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TenureBucketTransformer:
        """Fit transformer (no-op for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform tenure column into buckets.

        Args:
            X: DataFrame with 'tenure' column.

        Returns:
            DataFrame with 'tenure_bucket' column added.
        """
        X = X.copy()
        X["tenure_bucket"] = pd.cut(
            X["tenure"],
            bins=self.bins,
            labels=self.labels,
            include_lowest=True,
        )
        return X


class ServiceCountTransformer(BaseEstimator, TransformerMixin):
    """Count the number of add-on services a customer has.

    Counts services from: OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies.
    """

    def __init__(self) -> None:
        """Initialize service count transformer."""
        self.service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ServiceCountTransformer:
        """Fit transformer (no-op for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Count add-on services.

        Args:
            X: DataFrame with service columns.

        Returns:
            DataFrame with 'service_count' column added.
        """
        X = X.copy()

        # Count services (excluding "No" and "No internet service")
        service_mask = X[self.service_columns].isin(["Yes"])
        X["service_count"] = service_mask.sum(axis=1)

        return X


class RevenueSignalTransformer(BaseEstimator, TransformerMixin):
    """Create revenue-related signal features.

    Creates:
    - charges_ratio: MonthlyCharges / TotalCharges (when TotalCharges > 0)
    - avg_monthly_charge: TotalCharges / tenure (when tenure > 0)
    - charge_increase_flag: Whether MonthlyCharges > avg_monthly_charge
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RevenueSignalTransformer:
        """Fit transformer (no-op for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create revenue signal features.

        Args:
            X: DataFrame with 'MonthlyCharges', 'TotalCharges', and 'tenure' columns.

        Returns:
            DataFrame with revenue signal columns added.
        """
        X = X.copy()

        # Charges ratio: MonthlyCharges / TotalCharges
        # Only compute when TotalCharges > 0 to avoid division by zero
        X["charges_ratio"] = np.where(
            X["TotalCharges"] > 0,
            X["MonthlyCharges"] / X["TotalCharges"],
            np.nan,
        )

        # Average monthly charge: TotalCharges / tenure
        # Only compute when tenure > 0
        X["avg_monthly_charge"] = np.where(
            X["tenure"] > 0,
            X["TotalCharges"] / X["tenure"],
            np.nan,
        )

        # Charge increase flag: MonthlyCharges > avg_monthly_charge
        # Handle NaN values: if avg_monthly_charge is NaN, set flag to 0
        # Fill NaN in avg_monthly_charge with 0 for comparison (or use a large value)
        avg_monthly_charge_filled = X["avg_monthly_charge"].fillna(0)
        X["charge_increase_flag"] = (X["MonthlyCharges"] > avg_monthly_charge_filled).astype(int)
        
        # If avg_monthly_charge was NaN, set flag to 0
        X.loc[X["avg_monthly_charge"].isna(), "charge_increase_flag"] = 0

        return X


class CustomerLifetimeValueTransformer(BaseEstimator, TransformerMixin):
    """Approximate customer lifetime value (CLV).

    Creates CLV proxy as: TotalCharges + (MonthlyCharges * projected_months)
    where projected_months is based on contract type or a default value.
    """

    def __init__(self, default_projection_months: int = 12) -> None:
        """Initialize CLV transformer.

        Args:
            default_projection_months: Default months to project for month-to-month contracts.
        """
        self.default_projection_months = default_projection_months
        self.contract_projections = {
            "Month-to-month": default_projection_months,
            "One year": 12,
            "Two year": 24,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CustomerLifetimeValueTransformer:
        """Fit transformer (no-op for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create CLV approximation.

        Args:
            X: DataFrame with 'TotalCharges', 'MonthlyCharges', and 'Contract' columns.

        Returns:
            DataFrame with 'clv_approximation' column added.
        """
        X = X.copy()

        # Get projection months based on contract
        projection_months = (
            X["Contract"].map(self.contract_projections).fillna(self.default_projection_months)
        )

        # CLV = TotalCharges + (MonthlyCharges * projected_months)
        X["clv_approximation"] = X["TotalCharges"].fillna(0) + (
            X["MonthlyCharges"] * projection_months
        )

        return X
