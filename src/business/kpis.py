"""
KPI tracking and business metrics for churn retention platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass
class KPIMetrics:
    """Business KPI metrics."""

    timestamp: str
    total_customers: int
    customers_at_risk: int
    at_risk_percentage: float
    total_monthly_revenue: float
    revenue_at_risk: float
    revenue_at_risk_percentage: float
    avg_churn_probability: float
    critical_risk_customers: int
    high_risk_customers: int
    medium_risk_customers: int
    low_risk_customers: int
    estimated_monthly_churn: int
    estimated_annual_churn: int
    estimated_revenue_loss: float
    model_performance_roc_auc: float | None = None
    model_performance_f1: float | None = None
    intervention_readiness_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_customers": self.total_customers,
            "customers_at_risk": self.customers_at_risk,
            "at_risk_percentage": self.at_risk_percentage,
            "total_monthly_revenue": self.total_monthly_revenue,
            "revenue_at_risk": self.revenue_at_risk,
            "revenue_at_risk_percentage": self.revenue_at_risk_percentage,
            "avg_churn_probability": self.avg_churn_probability,
            "critical_risk_customers": self.critical_risk_customers,
            "high_risk_customers": self.high_risk_customers,
            "medium_risk_customers": self.medium_risk_customers,
            "low_risk_customers": self.low_risk_customers,
            "estimated_monthly_churn": self.estimated_monthly_churn,
            "estimated_annual_churn": self.estimated_annual_churn,
            "estimated_revenue_loss": self.estimated_revenue_loss,
            "model_performance_roc_auc": self.model_performance_roc_auc,
            "model_performance_f1": self.model_performance_f1,
            "intervention_readiness_score": self.intervention_readiness_score,
        }


class KPITracker:
    """Tracks business KPIs for churn retention.

    For detailed documentation on assumptions and parameter customization,
    see docs/business_assumptions.md.
    """

    def __init__(
        self,
        critical_threshold: float = 0.75,
        high_threshold: float = 0.5,
        medium_threshold: float = 0.3,
        churn_estimation_factor: float = 0.8,  # Factor to convert probability to actual churn
    ):
        """Initialize KPI tracker.

        Args:
            critical_threshold: Churn probability threshold for critical risk.
            high_threshold: Churn probability threshold for high risk.
            medium_threshold: Churn probability threshold for medium risk.
            churn_estimation_factor: Factor to estimate actual churn from probabilities.

        Note:
            Default values are based on standard risk categorization practices.
            For production use, customize these parameters based on your business priorities.
            See docs/business_assumptions.md for detailed guidance.
        """
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.churn_estimation_factor = churn_estimation_factor

    def calculate_kpis(
        self,
        customers: pd.DataFrame,
        predictions: pd.DataFrame,
        model_metrics: dict[str, float] | None = None,
        avg_customer_lifetime_months: int = 24,
    ) -> KPIMetrics:
        """Calculate business KPIs.

        Args:
            customers: DataFrame with customer data (must include customerID, MonthlyCharges).
            predictions: DataFrame with predictions (must include customerID, churn_probability).
            model_metrics: Optional model performance metrics (roc_auc, f1, etc.).
            avg_customer_lifetime_months: Average customer lifetime for revenue loss estimation.

        Returns:
            KPI metrics.
        """
        # Merge data
        df = customers.merge(predictions, on="customerID", how="inner")

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        # Basic counts
        total_customers = len(df)
        total_monthly_revenue = df["MonthlyCharges"].sum()

        # Risk segmentation
        critical_risk = len(df[df["churn_probability"] >= self.critical_threshold])
        high_risk = len(
            df[
                (df["churn_probability"] >= self.high_threshold)
                & (df["churn_probability"] < self.critical_threshold)
            ]
        )
        medium_risk = len(
            df[
                (df["churn_probability"] >= self.medium_threshold)
                & (df["churn_probability"] < self.high_threshold)
            ]
        )
        low_risk = len(df[df["churn_probability"] < self.medium_threshold])

        customers_at_risk = critical_risk + high_risk + medium_risk
        at_risk_percentage = (
            (customers_at_risk / total_customers * 100) if total_customers > 0 else 0.0
        )

        # Revenue at risk
        at_risk_df = df[df["churn_probability"] >= self.medium_threshold]
        revenue_at_risk = at_risk_df["MonthlyCharges"].sum() if len(at_risk_df) > 0 else 0.0
        revenue_at_risk_percentage = (
            (revenue_at_risk / total_monthly_revenue * 100) if total_monthly_revenue > 0 else 0.0
        )

        # Average churn probability
        avg_churn_probability = df["churn_probability"].mean()

        # Estimated churn
        estimated_monthly_churn = int(
            total_customers * avg_churn_probability * self.churn_estimation_factor
        )
        estimated_annual_churn = estimated_monthly_churn * 12

        # Estimated revenue loss
        avg_monthly_revenue = df["MonthlyCharges"].mean()
        estimated_revenue_loss = (
            estimated_monthly_churn * avg_monthly_revenue * avg_customer_lifetime_months
        )

        # Intervention readiness score (0-100)
        # Higher score = more ready for intervention (more high-value at-risk customers)
        if customers_at_risk > 0:
            high_value_at_risk = len(
                at_risk_df[at_risk_df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(0.75)]
            )
            intervention_readiness_score = (
                high_value_at_risk / customers_at_risk * 100
            )  # % of at-risk that are high-value
        else:
            intervention_readiness_score = 0.0

        return KPIMetrics(
            timestamp=timestamp,
            total_customers=total_customers,
            customers_at_risk=customers_at_risk,
            at_risk_percentage=at_risk_percentage,
            total_monthly_revenue=total_monthly_revenue,
            revenue_at_risk=revenue_at_risk,
            revenue_at_risk_percentage=revenue_at_risk_percentage,
            avg_churn_probability=avg_churn_probability,
            critical_risk_customers=critical_risk,
            high_risk_customers=high_risk,
            medium_risk_customers=medium_risk,
            low_risk_customers=low_risk,
            estimated_monthly_churn=estimated_monthly_churn,
            estimated_annual_churn=estimated_annual_churn,
            estimated_revenue_loss=estimated_revenue_loss,
            model_performance_roc_auc=model_metrics.get("roc_auc") if model_metrics else None,
            model_performance_f1=model_metrics.get("f1") if model_metrics else None,
            intervention_readiness_score=intervention_readiness_score,
        )

    def track_kpi_history(self, kpi_history: list[KPIMetrics]) -> dict[str, Any]:
        """Analyze KPI trends over time.

        Args:
            kpi_history: List of historical KPI metrics.

        Returns:
            Trend analysis.
        """
        if len(kpi_history) < 2:
            return {"status": "insufficient_data", "message": "Need at least 2 data points"}

        # Convert to DataFrame for analysis
        pd.DataFrame([kpi.to_dict() for kpi in kpi_history])

        # Calculate trends
        latest = kpi_history[-1]
        previous = kpi_history[-2]

        trends = {
            "customers_at_risk_change": latest.customers_at_risk - previous.customers_at_risk,
            "customers_at_risk_change_pct": (
                (latest.customers_at_risk - previous.customers_at_risk)
                / previous.customers_at_risk
                * 100
                if previous.customers_at_risk > 0
                else 0.0
            ),
            "revenue_at_risk_change": latest.revenue_at_risk - previous.revenue_at_risk,
            "revenue_at_risk_change_pct": (
                (latest.revenue_at_risk - previous.revenue_at_risk) / previous.revenue_at_risk * 100
                if previous.revenue_at_risk > 0
                else 0.0
            ),
            "avg_churn_probability_change": (
                latest.avg_churn_probability - previous.avg_churn_probability
            ),
            "trend_direction": (
                "improving"
                if latest.avg_churn_probability < previous.avg_churn_probability
                else "worsening"
            ),
        }

        return {
            "status": "success",
            "latest_kpis": latest.to_dict(),
            "trends": trends,
            "data_points": len(kpi_history),
        }
