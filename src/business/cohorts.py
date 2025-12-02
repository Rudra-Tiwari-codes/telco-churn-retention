"""
Customer segmentation and cohorting system for churn risk management.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class ChurnRiskLevel(str, Enum):
    """Churn risk levels."""

    CRITICAL = "critical"  # > 0.75 probability
    HIGH = "high"  # 0.5 - 0.75
    MEDIUM = "medium"  # 0.3 - 0.5
    LOW = "low"  # < 0.3


class CustomerSegment(str, Enum):
    """Customer value segments."""

    VIP = "vip"  # High revenue, long tenure
    LOYAL = "loyal"  # Long tenure, moderate revenue
    GROWING = "growing"  # Short tenure, high revenue potential
    AT_RISK = "at_risk"  # High churn risk, moderate value
    NEW = "new"  # New customers
    DISENGAGED = "disengaged"  # Low engagement, low value


@dataclass
class CustomerCohort:
    """Represents a customer cohort with aggregated metrics."""

    cohort_id: str
    name: str
    description: str
    risk_level: ChurnRiskLevel
    segment: CustomerSegment
    customer_count: int
    avg_churn_probability: float
    avg_monthly_revenue: float
    total_annual_revenue: float
    avg_tenure: float
    key_characteristics: dict[str, Any]
    recommended_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert cohort to dictionary."""
        return {
            "cohort_id": self.cohort_id,
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "segment": self.segment.value,
            "customer_count": self.customer_count,
            "avg_churn_probability": self.avg_churn_probability,
            "avg_monthly_revenue": self.avg_monthly_revenue,
            "total_annual_revenue": self.total_annual_revenue,
            "avg_tenure": self.avg_tenure,
            "key_characteristics": self.key_characteristics,
            "recommended_actions": self.recommended_actions,
        }


class CohortAnalyzer:
    """Analyzes customers and creates actionable cohorts."""

    def __init__(
        self,
        critical_threshold: float = 0.75,
        high_threshold: float = 0.5,
        medium_threshold: float = 0.3,
        vip_revenue_threshold: float = 100.0,
        vip_tenure_threshold: int = 36,
    ):
        """Initialize cohort analyzer.

        Args:
            critical_threshold: Churn probability threshold for critical risk.
            high_threshold: Churn probability threshold for high risk.
            medium_threshold: Churn probability threshold for medium risk.
            vip_revenue_threshold: Monthly revenue threshold for VIP segment.
            vip_tenure_threshold: Tenure threshold (months) for VIP segment.
        """
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.vip_revenue_threshold = vip_revenue_threshold
        self.vip_tenure_threshold = vip_tenure_threshold

    def assign_risk_level(self, churn_probability: float) -> ChurnRiskLevel:
        """Assign risk level based on churn probability."""
        if churn_probability >= self.critical_threshold:
            return ChurnRiskLevel.CRITICAL
        elif churn_probability >= self.high_threshold:
            return ChurnRiskLevel.HIGH
        elif churn_probability >= self.medium_threshold:
            return ChurnRiskLevel.MEDIUM
        else:
            return ChurnRiskLevel.LOW

    def assign_segment(
        self,
        monthly_revenue: float,
        tenure: int,
        churn_probability: float,
        service_count: int | None = None,
    ) -> CustomerSegment:
        """Assign customer segment based on value and engagement."""
        service_count = service_count or 0

        # VIP: High revenue, long tenure
        if monthly_revenue >= self.vip_revenue_threshold and tenure >= self.vip_tenure_threshold:
            return CustomerSegment.VIP

        # Loyal: Long tenure, moderate revenue
        if tenure >= 24 and monthly_revenue >= 50:
            return CustomerSegment.LOYAL

        # Growing: Short tenure but high revenue potential
        if tenure < 12 and monthly_revenue >= 80:
            return CustomerSegment.GROWING

        # At Risk: High churn probability, moderate value
        if churn_probability >= 0.5 and monthly_revenue >= 50:
            return CustomerSegment.AT_RISK

        # New: Short tenure
        if tenure < 6:
            return CustomerSegment.NEW

        # Disengaged: Low engagement indicators
        if service_count <= 2 and monthly_revenue < 50:
            return CustomerSegment.DISENGAGED

        # Default to at_risk if high churn probability
        if churn_probability >= 0.5:
            return CustomerSegment.AT_RISK

        return CustomerSegment.LOYAL

    def create_cohorts(
        self,
        customers: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> list[CustomerCohort]:
        """Create customer cohorts from predictions.

        Args:
            customers: DataFrame with customer data (must include customerID, MonthlyCharges, tenure).
            predictions: DataFrame with predictions (must include customerID, churn_probability).

        Returns:
            List of customer cohorts.
        """
        # Merge customer data with predictions
        df = customers.merge(predictions, on="customerID", how="inner")

        # Calculate service count
        service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        if all(col in df.columns for col in service_cols):
            df["service_count"] = df[service_cols].apply(
                lambda row: sum(1 for v in row if v == "Yes"), axis=1
            )
        else:
            df["service_count"] = 0

        # Assign risk levels and segments
        df["risk_level"] = df["churn_probability"].apply(self.assign_risk_level)
        df["segment"] = df.apply(
            lambda row: self.assign_segment(
                row.get("MonthlyCharges", 0),
                row.get("tenure", 0),
                row.get("churn_probability", 0),
                row.get("service_count", 0),
            ),
            axis=1,
        )

        # Create cohorts by risk level and segment combination
        cohorts = []
        for risk_level in ChurnRiskLevel:
            for segment in CustomerSegment:
                cohort_df = df[(df["risk_level"] == risk_level) & (df["segment"] == segment)]

                if len(cohort_df) == 0:
                    continue

                # Calculate cohort metrics
                cohort_id = f"{risk_level.value}_{segment.value}"
                name = (
                    f"{risk_level.value.title()} Risk - {segment.value.replace('_', ' ').title()}"
                )

                # Key characteristics
                characteristics = {
                    "avg_tenure": cohort_df["tenure"].mean(),
                    "avg_monthly_charges": cohort_df["MonthlyCharges"].mean(),
                    "contract_distribution": (
                        cohort_df["Contract"].value_counts().to_dict()
                        if "Contract" in cohort_df.columns
                        else {}
                    ),
                    "payment_method_distribution": (
                        cohort_df["PaymentMethod"].value_counts().to_dict()
                        if "PaymentMethod" in cohort_df.columns
                        else {}
                    ),
                    "internet_service_distribution": (
                        cohort_df["InternetService"].value_counts().to_dict()
                        if "InternetService" in cohort_df.columns
                        else {}
                    ),
                }

                # Recommended actions based on cohort
                actions = self._get_recommended_actions(risk_level, segment, characteristics)

                cohort = CustomerCohort(
                    cohort_id=cohort_id,
                    name=name,
                    description=self._get_cohort_description(risk_level, segment),
                    risk_level=risk_level,
                    segment=segment,
                    customer_count=len(cohort_df),
                    avg_churn_probability=cohort_df["churn_probability"].mean(),
                    avg_monthly_revenue=cohort_df["MonthlyCharges"].mean(),
                    total_annual_revenue=cohort_df["MonthlyCharges"].sum() * 12,
                    avg_tenure=cohort_df["tenure"].mean(),
                    key_characteristics=characteristics,
                    recommended_actions=actions,
                )

                cohorts.append(cohort)

        return sorted(
            cohorts, key=lambda c: (c.avg_churn_probability, -c.total_annual_revenue), reverse=True
        )

    def _get_cohort_description(self, risk_level: ChurnRiskLevel, segment: CustomerSegment) -> str:
        """Generate description for a cohort."""
        risk_desc = {
            ChurnRiskLevel.CRITICAL: "Immediate intervention required",
            ChurnRiskLevel.HIGH: "High priority retention efforts needed",
            ChurnRiskLevel.MEDIUM: "Proactive engagement recommended",
            ChurnRiskLevel.LOW: "Maintain current service levels",
        }

        segment_desc = {
            CustomerSegment.VIP: "High-value, long-term customers",
            CustomerSegment.LOYAL: "Established, reliable customers",
            CustomerSegment.GROWING: "New customers with high potential",
            CustomerSegment.AT_RISK: "Customers showing churn signals",
            CustomerSegment.NEW: "Recently acquired customers",
            CustomerSegment.DISENGAGED: "Low-engagement customers",
        }

        return f"{risk_desc[risk_level]}. {segment_desc[segment]}."

    def _get_recommended_actions(
        self,
        risk_level: ChurnRiskLevel,
        segment: CustomerSegment,
        characteristics: dict[str, Any],
    ) -> list[str]:
        """Generate recommended actions for a cohort."""
        actions = []

        # Risk-based actions
        if risk_level == ChurnRiskLevel.CRITICAL:
            actions.append("Immediate retention call from retention specialist")
            actions.append("Offer personalized discount (15-25% off for 6 months)")
            actions.append("Contract upgrade incentive (month-to-month → annual)")
            actions.append("Priority customer service support")

        elif risk_level == ChurnRiskLevel.HIGH:
            actions.append("Proactive outreach from account manager")
            actions.append("Targeted retention offer (10-15% discount)")
            actions.append("Service enhancement recommendations")
            actions.append("Payment method optimization (switch to auto-pay)")

        elif risk_level == ChurnRiskLevel.MEDIUM:
            actions.append("Engagement campaign via email/SMS")
            actions.append("Loyalty program enrollment")
            actions.append("Service usage optimization tips")

        # Segment-based actions
        if segment == CustomerSegment.VIP:
            actions.append("Dedicated account manager assignment")
            actions.append("VIP customer appreciation program")
            actions.append("Early access to new services")

        elif segment == CustomerSegment.GROWING:
            actions.append("Onboarding completion program")
            actions.append("Service bundle recommendations")
            actions.append("Referral program invitation")

        elif segment == CustomerSegment.NEW:
            actions.append("Welcome series completion")
            actions.append("First-month satisfaction check")
            actions.append("Service setup assistance")

        elif segment == CustomerSegment.DISENGAGED:
            actions.append("Re-engagement campaign")
            actions.append("Service discovery recommendations")
            actions.append("Usage analytics and tips")

        # Contract-based actions
        contract_dist = characteristics.get("contract_distribution", {})
        if "Month-to-month" in contract_dist and contract_dist["Month-to-month"] > 0.5:
            actions.append("Contract upgrade incentive (stability discount)")

        # Payment method actions
        payment_dist = characteristics.get("payment_method_distribution", {})
        if "Electronic check" in payment_dist and payment_dist["Electronic check"] > 0.3:
            actions.append("Auto-pay enrollment incentive ($5/month discount)")

        return list(set(actions))  # Remove duplicates
