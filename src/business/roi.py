"""
ROI calculator for churn prevention and retention initiatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ROIMetrics:
    """ROI metrics for retention initiatives."""

    total_customers_at_risk: int
    total_monthly_revenue_at_risk: float
    total_annual_revenue_at_risk: float
    estimated_churn_without_intervention: int
    estimated_revenue_lost_without_intervention: float
    intervention_cost: float
    estimated_churn_prevented: int
    estimated_revenue_saved: float
    net_benefit: float
    roi_percentage: float
    payback_period_months: float
    cost_per_customer_retained: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_customers_at_risk": self.total_customers_at_risk,
            "total_monthly_revenue_at_risk": self.total_monthly_revenue_at_risk,
            "total_annual_revenue_at_risk": self.total_annual_revenue_at_risk,
            "estimated_churn_without_intervention": self.estimated_churn_without_intervention,
            "estimated_revenue_lost_without_intervention": self.estimated_revenue_lost_without_intervention,
            "intervention_cost": self.intervention_cost,
            "estimated_churn_prevented": self.estimated_churn_prevented,
            "estimated_revenue_saved": self.estimated_revenue_saved,
            "net_benefit": self.net_benefit,
            "roi_percentage": self.roi_percentage,
            "payback_period_months": self.payback_period_months,
            "cost_per_customer_retained": self.cost_per_customer_retained,
        }


class ROICalculator:
    """Calculates ROI for retention initiatives.

    For detailed documentation on assumptions and parameter customization,
    see docs/business_assumptions.md.
    """

    def __init__(
        self,
        base_churn_rate: float = 0.265,  # Historical churn rate
        intervention_effectiveness: float = 0.30,  # 30% of interventions prevent churn
        customer_lifetime_months: int = 24,  # Average customer lifetime
    ):
        """Initialize ROI calculator.

        Args:
            base_churn_rate: Historical baseline churn rate.
            intervention_effectiveness: Effectiveness of retention interventions (0-1).
            customer_lifetime_months: Average customer lifetime in months.

        Note:
            Default values are based on industry benchmarks and the Telco dataset.
            For production use, customize these parameters based on your business metrics.
            See docs/business_assumptions.md for detailed guidance.
        """
        self.base_churn_rate = base_churn_rate
        self.intervention_effectiveness = intervention_effectiveness
        self.customer_lifetime_months = customer_lifetime_months

    def calculate_roi(
        self,
        customers: pd.DataFrame,
        predictions: pd.DataFrame,
        intervention_cost_per_customer: float = 50.0,
        churn_threshold: float = 0.5,
    ) -> ROIMetrics:
        """Calculate ROI for retention intervention.

        Args:
            customers: DataFrame with customer data (must include customerID, MonthlyCharges).
            predictions: DataFrame with predictions (must include customerID, churn_probability).
            intervention_cost_per_customer: Average cost per customer intervention.
            churn_threshold: Churn probability threshold for "at risk" classification.

        Returns:
            ROI metrics.
        """
        # Merge data
        df = customers.merge(predictions, on="customerID", how="inner")

        # Identify at-risk customers
        at_risk = df[df["churn_probability"] >= churn_threshold].copy()

        if len(at_risk) == 0:
            # Return zero metrics if no at-risk customers
            return ROIMetrics(
                total_customers_at_risk=0,
                total_monthly_revenue_at_risk=0.0,
                total_annual_revenue_at_risk=0.0,
                estimated_churn_without_intervention=0,
                estimated_revenue_lost_without_intervention=0.0,
                intervention_cost=0.0,
                estimated_churn_prevented=0,
                estimated_revenue_saved=0.0,
                net_benefit=0.0,
                roi_percentage=0.0,
                payback_period_months=0.0,
                cost_per_customer_retained=0.0,
            )

        # Calculate revenue at risk
        total_monthly_revenue_at_risk = at_risk["MonthlyCharges"].sum()
        total_annual_revenue_at_risk = total_monthly_revenue_at_risk * 12

        # Estimate churn without intervention
        # Use weighted average of churn probabilities
        estimated_churn_rate = at_risk["churn_probability"].mean()
        estimated_churn_without_intervention = int(len(at_risk) * estimated_churn_rate)

        # Estimate revenue lost
        avg_monthly_revenue = at_risk["MonthlyCharges"].mean()
        estimated_revenue_lost_without_intervention = (
            estimated_churn_without_intervention
            * avg_monthly_revenue
            * self.customer_lifetime_months
        )

        # Calculate intervention costs
        intervention_cost = len(at_risk) * intervention_cost_per_customer

        # Estimate churn prevented
        estimated_churn_prevented = int(
            estimated_churn_without_intervention * self.intervention_effectiveness
        )

        # Estimate revenue saved
        estimated_revenue_saved = (
            estimated_churn_prevented * avg_monthly_revenue * self.customer_lifetime_months
        )

        # Calculate net benefit and ROI
        net_benefit = estimated_revenue_saved - intervention_cost
        roi_percentage = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0.0

        # Payback period (months)
        monthly_savings = estimated_revenue_saved / 12
        payback_period_months = intervention_cost / monthly_savings if monthly_savings > 0 else 0.0

        # Cost per customer retained
        cost_per_customer_retained = (
            intervention_cost / estimated_churn_prevented if estimated_churn_prevented > 0 else 0.0
        )

        return ROIMetrics(
            total_customers_at_risk=len(at_risk),
            total_monthly_revenue_at_risk=total_monthly_revenue_at_risk,
            total_annual_revenue_at_risk=total_annual_revenue_at_risk,
            estimated_churn_without_intervention=estimated_churn_without_intervention,
            estimated_revenue_lost_without_intervention=estimated_revenue_lost_without_intervention,
            intervention_cost=intervention_cost,
            estimated_churn_prevented=estimated_churn_prevented,
            estimated_revenue_saved=estimated_revenue_saved,
            net_benefit=net_benefit,
            roi_percentage=roi_percentage,
            payback_period_months=payback_period_months,
            cost_per_customer_retained=cost_per_customer_retained,
        )

    def calculate_cohort_roi(
        self,
        cohort: dict[str, Any],
        intervention_cost_per_customer: float = 50.0,
    ) -> ROIMetrics:
        """Calculate ROI for a specific cohort.

        Args:
            cohort: Cohort dictionary with metrics.
            intervention_cost_per_customer: Average cost per customer intervention.

        Returns:
            ROI metrics for the cohort.
        """
        customer_count = cohort.get("customer_count", 0)
        avg_churn_prob = cohort.get("avg_churn_probability", 0.5)
        avg_monthly_revenue = cohort.get("avg_monthly_revenue", 0)

        if customer_count == 0:
            return ROIMetrics(
                total_customers_at_risk=0,
                total_monthly_revenue_at_risk=0.0,
                total_annual_revenue_at_risk=0.0,
                estimated_churn_without_intervention=0,
                estimated_revenue_lost_without_intervention=0.0,
                intervention_cost=0.0,
                estimated_churn_prevented=0,
                estimated_revenue_saved=0.0,
                net_benefit=0.0,
                roi_percentage=0.0,
                payback_period_months=0.0,
                cost_per_customer_retained=0.0,
            )

        total_monthly_revenue_at_risk = avg_monthly_revenue * customer_count
        total_annual_revenue_at_risk = total_monthly_revenue_at_risk * 12

        estimated_churn_without_intervention = int(customer_count * avg_churn_prob)
        estimated_revenue_lost_without_intervention = (
            estimated_churn_without_intervention
            * avg_monthly_revenue
            * self.customer_lifetime_months
        )

        intervention_cost = customer_count * intervention_cost_per_customer
        estimated_churn_prevented = int(
            estimated_churn_without_intervention * self.intervention_effectiveness
        )
        estimated_revenue_saved = (
            estimated_churn_prevented * avg_monthly_revenue * self.customer_lifetime_months
        )

        net_benefit = estimated_revenue_saved - intervention_cost
        roi_percentage = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0.0

        monthly_savings = estimated_revenue_saved / 12
        payback_period_months = intervention_cost / monthly_savings if monthly_savings > 0 else 0.0

        cost_per_customer_retained = (
            intervention_cost / estimated_churn_prevented if estimated_churn_prevented > 0 else 0.0
        )

        return ROIMetrics(
            total_customers_at_risk=customer_count,
            total_monthly_revenue_at_risk=total_monthly_revenue_at_risk,
            total_annual_revenue_at_risk=total_annual_revenue_at_risk,
            estimated_churn_without_intervention=estimated_churn_without_intervention,
            estimated_revenue_lost_without_intervention=estimated_revenue_lost_without_intervention,
            intervention_cost=intervention_cost,
            estimated_churn_prevented=estimated_churn_prevented,
            estimated_revenue_saved=estimated_revenue_saved,
            net_benefit=net_benefit,
            roi_percentage=roi_percentage,
            payback_period_months=payback_period_months,
            cost_per_customer_retained=cost_per_customer_retained,
        )

    def calculate_scenario_roi(
        self,
        scenario_name: str,
        customers_at_risk: int,
        avg_monthly_revenue: float,
        intervention_cost_per_customer: float,
        intervention_effectiveness: float | None = None,
    ) -> dict[str, Any]:
        """Calculate ROI for a specific scenario.

        Args:
            scenario_name: Name of the scenario.
            customers_at_risk: Number of customers at risk.
            avg_monthly_revenue: Average monthly revenue per customer.
            intervention_cost_per_customer: Cost per customer intervention.
            intervention_effectiveness: Override default effectiveness (optional).

        Returns:
            Scenario ROI analysis.
        """
        effectiveness = (
            intervention_effectiveness
            if intervention_effectiveness is not None
            else self.intervention_effectiveness
        )

        total_revenue_at_risk = customers_at_risk * avg_monthly_revenue * 12
        estimated_churn = int(customers_at_risk * 0.5)  # Assume 50% churn rate
        estimated_churn_prevented = int(estimated_churn * effectiveness)
        revenue_saved = (
            estimated_churn_prevented * avg_monthly_revenue * self.customer_lifetime_months
        )
        intervention_cost = customers_at_risk * intervention_cost_per_customer
        net_benefit = revenue_saved - intervention_cost
        roi = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0.0

        return {
            "scenario_name": scenario_name,
            "customers_at_risk": customers_at_risk,
            "total_revenue_at_risk": total_revenue_at_risk,
            "estimated_churn": estimated_churn,
            "estimated_churn_prevented": estimated_churn_prevented,
            "revenue_saved": revenue_saved,
            "intervention_cost": intervention_cost,
            "net_benefit": net_benefit,
            "roi_percentage": roi,
        }
