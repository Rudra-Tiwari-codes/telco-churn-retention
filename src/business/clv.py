"""
Predictive Customer Lifetime Value (CLV) calculator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CLVMetrics:
    """Customer Lifetime Value metrics."""

    customer_id: str
    current_monthly_revenue: float
    predicted_tenure_months: float
    predicted_clv: float
    clv_percentile: float  # Percentile rank (0-100)
    risk_adjusted_clv: float
    retention_probability: float
    churn_probability: float
    recommended_retention_budget: float  # Max to spend on retention

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "customer_id": self.customer_id,
            "current_monthly_revenue": self.current_monthly_revenue,
            "predicted_tenure_months": self.predicted_tenure_months,
            "predicted_clv": self.predicted_clv,
            "clv_percentile": self.clv_percentile,
            "risk_adjusted_clv": self.risk_adjusted_clv,
            "retention_probability": self.retention_probability,
            "churn_probability": self.churn_probability,
            "recommended_retention_budget": self.recommended_retention_budget,
        }


class PredictiveCLVCalculator:
    """Calculates predictive Customer Lifetime Value."""

    def __init__(
        self,
        avg_customer_lifetime_months: int = 24,
        discount_rate: float = 0.1,  # Annual discount rate
        retention_budget_ratio: float = 0.15,  # Max 15% of CLV for retention
    ):
        """Initialize CLV calculator.

        Args:
            avg_customer_lifetime_months: Average customer lifetime in months.
            discount_rate: Annual discount rate for NPV calculation.
            retention_budget_ratio: Maximum retention budget as ratio of CLV.
        """
        self.avg_customer_lifetime_months = avg_customer_lifetime_months
        self.discount_rate = discount_rate
        self.retention_budget_ratio = retention_budget_ratio

    def calculate_clv(
        self,
        monthly_revenue: float,
        tenure: int,
        churn_probability: float,
        monthly_growth_rate: float = 0.0,
    ) -> float:
        """Calculate predicted CLV.

        Args:
            monthly_revenue: Current monthly revenue.
            tenure: Current tenure in months.
            churn_probability: Predicted churn probability.
            monthly_growth_rate: Expected monthly revenue growth rate.

        Returns:
            Predicted CLV.
        """
        # Estimate remaining lifetime
        retention_probability = 1 - churn_probability

        # Simplified: expected tenure = current tenure + (retention_prob * avg_remaining_lifetime)
        avg_remaining_lifetime = self.avg_customer_lifetime_months - tenure
        expected_remaining_months = max(0, retention_probability * avg_remaining_lifetime)

        # Calculate CLV with growth and discounting
        monthly_discount = (1 + self.discount_rate) ** (1 / 12) - 1

        clv = 0.0
        for month in range(int(expected_remaining_months)):
            future_revenue = monthly_revenue * (1 + monthly_growth_rate) ** month
            discounted_revenue = future_revenue / ((1 + monthly_discount) ** month)
            clv += discounted_revenue

        return clv

    def calculate_clv_for_customers(
        self,
        customers: pd.DataFrame,
        predictions: pd.DataFrame,
        monthly_growth_rate: float = 0.0,
    ) -> list[CLVMetrics]:
        """Calculate CLV for multiple customers.

        Args:
            customers: DataFrame with customer data (must include customerID, MonthlyCharges, tenure).
            predictions: DataFrame with predictions (must include customerID, churn_probability).

        Returns:
            List of CLV metrics.
        """
        # Merge data
        df = customers.merge(predictions, on="customerID", how="inner")

        # Calculate CLV for each customer
        clv_metrics = []
        clv_values = []

        for _, row in df.iterrows():
            customer_id = row["customerID"]
            monthly_revenue = row.get("MonthlyCharges", 0)
            tenure = row.get("tenure", 0)
            churn_prob = row.get("churn_probability", 0.5)

            # Calculate CLV
            predicted_clv = self.calculate_clv(
                monthly_revenue=monthly_revenue,
                tenure=tenure,
                churn_probability=churn_prob,
                monthly_growth_rate=monthly_growth_rate,
            )

            # Risk-adjusted CLV (accounting for churn probability)
            risk_adjusted_clv = predicted_clv * (1 - churn_prob)

            # Retention budget (max to spend)
            retention_budget = predicted_clv * self.retention_budget_ratio

            metrics = CLVMetrics(
                customer_id=customer_id,
                current_monthly_revenue=monthly_revenue,
                predicted_tenure_months=tenure
                + (1 - churn_prob) * self.avg_customer_lifetime_months,
                predicted_clv=predicted_clv,
                clv_percentile=0.0,  # Will calculate after all CLVs
                risk_adjusted_clv=risk_adjusted_clv,
                retention_probability=1 - churn_prob,
                churn_probability=churn_prob,
                recommended_retention_budget=retention_budget,
            )

            clv_metrics.append(metrics)
            clv_values.append(predicted_clv)

        # Calculate percentiles
        if clv_values:
            clv_array = np.array(clv_values)
            for i, metrics in enumerate(clv_metrics):
                metrics.clv_percentile = (clv_array < clv_values[i]).sum() / len(clv_array) * 100

        return clv_metrics

    def segment_by_clv(self, clv_metrics: list[CLVMetrics]) -> dict[str, list[CLVMetrics]]:
        """Segment customers by CLV.

        Args:
            clv_metrics: List of CLV metrics.

        Returns:
            Dictionary with segments: "high_value", "medium_value", "low_value".
        """
        if not clv_metrics:
            return {"high_value": [], "medium_value": [], "low_value": []}

        clv_values = [m.predicted_clv for m in clv_metrics]
        p75 = np.percentile(clv_values, 75)
        p25 = np.percentile(clv_values, 25)

        segments = {
            "high_value": [m for m in clv_metrics if m.predicted_clv >= p75],
            "medium_value": [m for m in clv_metrics if p25 <= m.predicted_clv < p75],
            "low_value": [m for m in clv_metrics if m.predicted_clv < p25],
        }

        return segments

    def calculate_portfolio_clv(self, clv_metrics: list[CLVMetrics]) -> dict[str, float]:
        """Calculate portfolio-level CLV metrics.

        Args:
            clv_metrics: List of CLV metrics.

        Returns:
            Portfolio CLV summary.
        """
        if not clv_metrics:
            return {
                "total_clv": 0.0,
                "avg_clv": 0.0,
                "median_clv": 0.0,
                "total_risk_adjusted_clv": 0.0,
                "total_retention_budget": 0.0,
            }

        total_clv = sum(m.predicted_clv for m in clv_metrics)
        total_risk_adjusted = sum(m.risk_adjusted_clv for m in clv_metrics)
        total_budget = sum(m.recommended_retention_budget for m in clv_metrics)

        clv_values = [m.predicted_clv for m in clv_metrics]

        return {
            "total_clv": total_clv,
            "avg_clv": total_clv / len(clv_metrics),
            "median_clv": float(np.median(clv_values)),
            "total_risk_adjusted_clv": total_risk_adjusted,
            "total_retention_budget": total_budget,
            "customer_count": len(clv_metrics),
        }
