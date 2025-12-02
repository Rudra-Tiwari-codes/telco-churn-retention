"""
Customer journey mapping and lifecycle analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class JourneyStage(str, Enum):
    """Customer journey stages."""

    ACQUISITION = "acquisition"  # 0-3 months
    ONBOARDING = "onboarding"  # 3-6 months
    GROWTH = "growth"  # 6-12 months
    MATURITY = "maturity"  # 12-24 months
    RETENTION = "retention"  # 24+ months
    AT_RISK = "at_risk"  # High churn probability
    CHURNED = "churned"  # Lost customer


@dataclass
class JourneyStageMetrics:
    """Metrics for a customer journey stage."""

    stage: JourneyStage
    customer_count: int
    avg_churn_probability: float
    avg_monthly_revenue: float
    avg_tenure: float
    total_annual_revenue: float
    key_characteristics: dict[str, Any]
    transition_probabilities: dict[str, float]  # Probability of moving to next stage
    recommended_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "customer_count": self.customer_count,
            "avg_churn_probability": self.avg_churn_probability,
            "avg_monthly_revenue": self.avg_monthly_revenue,
            "avg_tenure": self.avg_tenure,
            "total_annual_revenue": self.total_annual_revenue,
            "key_characteristics": self.key_characteristics,
            "transition_probabilities": self.transition_probabilities,
            "recommended_actions": self.recommended_actions,
        }


class CustomerJourneyMapper:
    """Maps customers to journey stages and analyzes lifecycle."""

    def __init__(self):
        """Initialize journey mapper."""
        self.stage_thresholds = {
            JourneyStage.ACQUISITION: (0, 3),
            JourneyStage.ONBOARDING: (3, 6),
            JourneyStage.GROWTH: (6, 12),
            JourneyStage.MATURITY: (12, 24),
            JourneyStage.RETENTION: (24, 999),
        }

    def assign_journey_stage(self, tenure: int, churn_probability: float) -> JourneyStage:
        """Assign customer to journey stage.

        Args:
            tenure: Customer tenure in months.
            churn_probability: Predicted churn probability.

        Returns:
            Journey stage.
        """
        # High churn probability overrides stage
        if churn_probability >= 0.75:
            return JourneyStage.AT_RISK

        # Assign based on tenure
        for stage, (min_tenure, max_tenure) in self.stage_thresholds.items():
            if min_tenure <= tenure < max_tenure:
                return stage

        # Default to retention for very long tenure
        return JourneyStage.RETENTION

    def analyze_journey_stages(
        self,
        customers: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> list[JourneyStageMetrics]:
        """Analyze customer journey stages.

        Args:
            customers: DataFrame with customer data (must include customerID, tenure, MonthlyCharges).
            predictions: DataFrame with predictions (must include customerID, churn_probability).

        Returns:
            List of journey stage metrics.
        """
        # Merge data
        df = customers.merge(predictions, on="customerID", how="inner")

        # Assign journey stages
        df["journey_stage"] = df.apply(
            lambda row: self.assign_journey_stage(
                row.get("tenure", 0), row.get("churn_probability", 0)
            ),
            axis=1,
        )

        # Analyze each stage
        stage_metrics = []
        for stage in JourneyStage:
            if stage == JourneyStage.CHURNED:
                continue  # Skip churned stage (would need historical data)

            stage_df = df[df["journey_stage"] == stage]

            if len(stage_df) == 0:
                continue

            # Calculate metrics
            metrics = JourneyStageMetrics(
                stage=stage,
                customer_count=len(stage_df),
                avg_churn_probability=stage_df["churn_probability"].mean(),
                avg_monthly_revenue=stage_df["MonthlyCharges"].mean(),
                avg_tenure=stage_df["tenure"].mean(),
                total_annual_revenue=stage_df["MonthlyCharges"].sum() * 12,
                key_characteristics=self._extract_characteristics(stage_df),
                transition_probabilities=self._calculate_transitions(stage_df, df),
                recommended_actions=self._get_stage_actions(stage),
            )

            stage_metrics.append(metrics)

        return sorted(stage_metrics, key=lambda m: m.customer_count, reverse=True)

    def _extract_characteristics(self, stage_df: pd.DataFrame) -> dict[str, Any]:
        """Extract key characteristics for a stage."""
        characteristics = {
            "avg_tenure": stage_df["tenure"].mean(),
            "avg_monthly_charges": stage_df["MonthlyCharges"].mean(),
        }

        # Contract distribution
        if "Contract" in stage_df.columns:
            characteristics["contract_distribution"] = (
                stage_df["Contract"].value_counts(normalize=True).to_dict()
            )

        # Payment method distribution
        if "PaymentMethod" in stage_df.columns:
            characteristics["payment_method_distribution"] = (
                stage_df["PaymentMethod"].value_counts(normalize=True).to_dict()
            )

        # Service count (if available)
        service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        if all(col in stage_df.columns for col in service_cols):
            stage_df["service_count"] = stage_df[service_cols].apply(
                lambda row: sum(1 for v in row if v == "Yes"), axis=1
            )
            characteristics["avg_service_count"] = stage_df["service_count"].mean()

        return characteristics

    def _calculate_transitions(
        self, stage_df: pd.DataFrame, full_df: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate transition probabilities to next stages."""
        transitions = {}

        # Simplified: estimate based on churn probability and tenure
        avg_churn_prob = stage_df["churn_probability"].mean()
        stage_df["tenure"].mean()

        # Probability of staying in current stage (if low churn)
        if avg_churn_prob < 0.3:
            transitions["stay"] = 0.7
            transitions["advance"] = 0.25
            transitions["at_risk"] = 0.05
        elif avg_churn_prob < 0.5:
            transitions["stay"] = 0.5
            transitions["advance"] = 0.2
            transitions["at_risk"] = 0.3
        else:
            transitions["stay"] = 0.3
            transitions["advance"] = 0.1
            transitions["at_risk"] = 0.6

        return transitions

    def _get_stage_actions(self, stage: JourneyStage) -> list[str]:
        """Get recommended actions for a journey stage."""
        actions_map = {
            JourneyStage.ACQUISITION: [
                "Welcome series completion",
                "First-month satisfaction check",
                "Service setup assistance",
                "Onboarding completion program",
            ],
            JourneyStage.ONBOARDING: [
                "Feature discovery campaigns",
                "Usage optimization tips",
                "Service bundle recommendations",
                "Referral program invitation",
            ],
            JourneyStage.GROWTH: [
                "Upsell opportunities",
                "Service enhancement recommendations",
                "Loyalty program enrollment",
                "Contract upgrade incentives",
            ],
            JourneyStage.MATURITY: [
                "VIP program enrollment",
                "Account manager assignment",
                "Early access to new services",
                "Customer appreciation programs",
            ],
            JourneyStage.RETENTION: [
                "Dedicated account manager",
                "VIP customer benefits",
                "Long-term contract incentives",
                "Exclusive offers",
            ],
            JourneyStage.AT_RISK: [
                "Immediate retention call",
                "Personalized discount offers",
                "Contract upgrade incentives",
                "Priority customer service",
            ],
        }

        return actions_map.get(stage, [])

    def generate_journey_insights(self, stage_metrics: list[JourneyStageMetrics]) -> dict[str, Any]:
        """Generate insights from journey analysis.

        Args:
            stage_metrics: List of journey stage metrics.

        Returns:
            Journey insights dictionary.
        """
        total_customers = sum(m.customer_count for m in stage_metrics)
        total_revenue = sum(m.total_annual_revenue for m in stage_metrics)

        # Find critical stages
        at_risk_stage = next((m for m in stage_metrics if m.stage == JourneyStage.AT_RISK), None)

        insights: dict[str, Any] = {
            "total_customers": total_customers,
            "total_annual_revenue": total_revenue,
            "stage_distribution": {
                m.stage.value: {
                    "count": m.customer_count,
                    "percentage": (
                        m.customer_count / total_customers * 100 if total_customers > 0 else 0
                    ),
                    "revenue": m.total_annual_revenue,
                }
                for m in stage_metrics
            },
            "critical_findings": [],
            "recommendations": [],
        }

        # Critical findings
        if at_risk_stage and at_risk_stage.customer_count > 0:
            insights["critical_findings"].append(
                f"{at_risk_stage.customer_count} customers ({at_risk_stage.customer_count/total_customers*100:.1f}%) "
                f"are at risk with {at_risk_stage.avg_churn_probability:.1%} average churn probability"
            )

        # Find stage with highest churn probability
        highest_risk_stage = max(
            [m for m in stage_metrics if m.stage != JourneyStage.AT_RISK],
            key=lambda m: m.avg_churn_probability,
            default=None,
        )

        if highest_risk_stage:
            insights["critical_findings"].append(
                f"{highest_risk_stage.stage.value.title()} stage has highest churn risk "
                f"({highest_risk_stage.avg_churn_probability:.1%})"
            )

        # Recommendations
        if at_risk_stage and at_risk_stage.customer_count > total_customers * 0.1:
            insights["recommendations"].append(
                "URGENT: Deploy immediate retention interventions for at-risk customers"
            )

        acquisition_stage = next(
            (m for m in stage_metrics if m.stage == JourneyStage.ACQUISITION), None
        )
        if acquisition_stage and acquisition_stage.avg_churn_probability > 0.4:
            insights["recommendations"].append(
                "Improve onboarding process: High churn in acquisition stage indicates onboarding issues"
            )

        return insights
