"""
Retention playbook generator with actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RetentionAction:
    """A single retention action recommendation."""

    action_id: str
    title: str
    description: str
    priority: str  # "critical", "high", "medium", "low"
    estimated_cost: float
    estimated_effectiveness: float  # 0-1, probability of preventing churn
    expected_roi: float
    implementation_time: str  # e.g., "immediate", "1-2 days", "1 week"
    target_customers: list[str]  # customer IDs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "estimated_cost": self.estimated_cost,
            "estimated_effectiveness": self.estimated_effectiveness,
            "expected_roi": self.expected_roi,
            "implementation_time": self.implementation_time,
            "target_customers": self.target_customers,
            "target_count": len(self.target_customers),
        }


@dataclass
class RetentionPlaybook:
    """Complete retention playbook for a customer or cohort."""

    playbook_id: str
    customer_id: str | None
    cohort_id: str | None
    churn_probability: float
    monthly_revenue: float
    lifetime_value_estimate: float
    actions: list[RetentionAction]
    total_estimated_cost: float
    total_expected_savings: float
    net_roi: float
    recommended_strategy: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "playbook_id": self.playbook_id,
            "customer_id": self.customer_id,
            "cohort_id": self.cohort_id,
            "churn_probability": self.churn_probability,
            "monthly_revenue": self.monthly_revenue,
            "lifetime_value_estimate": self.lifetime_value_estimate,
            "actions": [action.to_dict() for action in self.actions],
            "total_estimated_cost": self.total_estimated_cost,
            "total_expected_savings": self.total_expected_savings,
            "net_roi": self.net_roi,
            "recommended_strategy": self.recommended_strategy,
        }


class RetentionPlaybookGenerator:
    """Generates personalized retention playbooks."""

    def __init__(
        self,
        discount_cost_percentage: float = 0.15,  # Cost of discount as % of revenue
        retention_specialist_cost: float = 50.0,  # Cost per retention call
        account_manager_cost: float = 25.0,  # Cost per proactive outreach
    ):
        """Initialize playbook generator.

        Args:
            discount_cost_percentage: Cost of discount as percentage of monthly revenue.
            retention_specialist_cost: Cost per retention specialist call.
            account_manager_cost: Cost per account manager outreach.
        """
        self.discount_cost_percentage = discount_cost_percentage
        self.retention_specialist_cost = retention_specialist_cost
        self.account_manager_cost = account_manager_cost

    def generate_playbook(
        self,
        customer_data: dict[str, Any],
        churn_probability: float,
        monthly_revenue: float,
        tenure: int,
        explanations: list[dict[str, Any]] | None = None,
    ) -> RetentionPlaybook:
        """Generate retention playbook for a customer.

        Args:
            customer_data: Customer data dictionary.
            churn_probability: Predicted churn probability.
            monthly_revenue: Monthly revenue from customer.
            tenure: Customer tenure in months.
            explanations: SHAP explanations for top features.

        Returns:
            Retention playbook.
        """
        customer_id = customer_data.get("customerID", "unknown")
        playbook_id = f"playbook_{customer_id}"

        # Calculate lifetime value estimate (simplified: monthly revenue * expected tenure)
        expected_tenure_months = max(12, tenure + (1 - churn_probability) * 24)
        lifetime_value_estimate = monthly_revenue * expected_tenure_months

        # Generate actions based on churn probability and customer characteristics
        actions = self._generate_actions(
            customer_data,
            churn_probability,
            monthly_revenue,
            tenure,
            explanations or [],
        )

        # Calculate totals
        total_cost = sum(action.estimated_cost for action in actions)
        total_savings = sum(
            action.estimated_cost * action.estimated_effectiveness * monthly_revenue * 12
            for action in actions
        )
        net_roi = (total_savings - total_cost) / total_cost if total_cost > 0 else 0

        # Determine strategy
        strategy = self._determine_strategy(churn_probability, monthly_revenue, tenure)

        return RetentionPlaybook(
            playbook_id=playbook_id,
            customer_id=customer_id,
            cohort_id=None,
            churn_probability=churn_probability,
            monthly_revenue=monthly_revenue,
            lifetime_value_estimate=lifetime_value_estimate,
            actions=actions,
            total_estimated_cost=total_cost,
            total_expected_savings=total_savings,
            net_roi=net_roi,
            recommended_strategy=strategy,
        )

    def generate_cohort_playbook(
        self,
        cohort: dict[str, Any],
        customers: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> RetentionPlaybook:
        """Generate playbook for an entire cohort.

        Args:
            cohort: Cohort information dictionary.
            customers: DataFrame with customer data.
            predictions: DataFrame with predictions.

        Returns:
            Cohort retention playbook.
        """
        cohort_id = cohort.get("cohort_id", "unknown")
        playbook_id = f"cohort_playbook_{cohort_id}"

        # Get customers in this cohort
        customers.merge(predictions, on="customerID", how="inner")
        # Filter by cohort characteristics (simplified - in practice would use cohort assignment)

        avg_churn_prob = cohort.get("avg_churn_probability", 0.5)
        avg_revenue = cohort.get("avg_monthly_revenue", 0)
        cohort.get("avg_tenure", 0)
        customer_count = cohort.get("customer_count", 0)

        # Generate cohort-level actions
        actions = self._generate_cohort_actions(cohort, customer_count)

        # Calculate totals
        total_cost = sum(action.estimated_cost for action in actions)
        total_savings = sum(
            action.estimated_cost
            * action.estimated_effectiveness
            * avg_revenue
            * 12
            * customer_count
            for action in actions
        )
        net_roi = (total_savings - total_cost) / total_cost if total_cost > 0 else 0

        strategy = f"Cohort-level intervention for {customer_count} customers"

        return RetentionPlaybook(
            playbook_id=playbook_id,
            customer_id=None,
            cohort_id=cohort_id,
            churn_probability=avg_churn_prob,
            monthly_revenue=avg_revenue * customer_count,
            lifetime_value_estimate=avg_revenue * customer_count * 24,
            actions=actions,
            total_estimated_cost=total_cost,
            total_expected_savings=total_savings,
            net_roi=net_roi,
            recommended_strategy=strategy,
        )

    def _generate_actions(
        self,
        customer_data: dict[str, Any],
        churn_probability: float,
        monthly_revenue: float,
        tenure: int,
        explanations: list[dict[str, Any]],
    ) -> list[RetentionAction]:
        """Generate retention actions for a customer."""
        actions = []
        customer_id = customer_data.get("customerID", "unknown")

        # Critical actions for high churn probability
        if churn_probability >= 0.75:
            # Immediate retention call
            actions.append(
                RetentionAction(
                    action_id="retention_call_critical",
                    title="Immediate Retention Specialist Call",
                    description="Priority call from retention specialist to understand concerns and offer personalized solution",
                    priority="critical",
                    estimated_cost=self.retention_specialist_cost,
                    estimated_effectiveness=0.35,  # 35% chance of preventing churn
                    expected_roi=(monthly_revenue * 12 * 0.35 - self.retention_specialist_cost)
                    / self.retention_specialist_cost,
                    implementation_time="immediate",
                    target_customers=[customer_id],
                )
            )

            # Personalized discount
            discount_pct = 0.20  # 20% discount
            discount_cost = monthly_revenue * discount_pct * 6  # 6 months
            actions.append(
                RetentionAction(
                    action_id="personalized_discount",
                    title=f"Personalized {int(discount_pct * 100)}% Discount (6 months)",
                    description=f"Offer {int(discount_pct * 100)}% discount on monthly charges for 6 months to retain customer",
                    priority="critical",
                    estimated_cost=discount_cost,
                    estimated_effectiveness=0.45,
                    expected_roi=(monthly_revenue * 12 * 0.45 - discount_cost) / discount_cost,
                    implementation_time="1-2 days",
                    target_customers=[customer_id],
                )
            )

        elif churn_probability >= 0.5:
            # High risk actions
            actions.append(
                RetentionAction(
                    action_id="proactive_outreach",
                    title="Proactive Account Manager Outreach",
                    description="Proactive call/email from account manager to check satisfaction and offer improvements",
                    priority="high",
                    estimated_cost=self.account_manager_cost,
                    estimated_effectiveness=0.25,
                    expected_roi=(monthly_revenue * 12 * 0.25 - self.account_manager_cost)
                    / self.account_manager_cost,
                    implementation_time="1-2 days",
                    target_customers=[customer_id],
                )
            )

            # Targeted discount
            discount_pct = 0.15
            discount_cost = monthly_revenue * discount_pct * 3  # 3 months
            actions.append(
                RetentionAction(
                    action_id="targeted_discount",
                    title=f"Targeted {int(discount_pct * 100)}% Discount (3 months)",
                    description=f"Offer {int(discount_pct * 100)}% discount for 3 months",
                    priority="high",
                    estimated_cost=discount_cost,
                    estimated_effectiveness=0.30,
                    expected_roi=(monthly_revenue * 12 * 0.30 - discount_cost) / discount_cost,
                    implementation_time="1-2 days",
                    target_customers=[customer_id],
                )
            )

        # Contract-based actions
        contract = customer_data.get("Contract", "")
        if contract == "Month-to-month" and churn_probability >= 0.4:
            # Contract upgrade incentive
            monthly_revenue * 0.10 * 12  # 10% annual discount
            actions.append(
                RetentionAction(
                    action_id="contract_upgrade",
                    title="Contract Upgrade Incentive",
                    description="Offer 10% annual discount to switch from month-to-month to annual contract",
                    priority="medium" if churn_probability < 0.5 else "high",
                    estimated_cost=monthly_revenue * 0.10 * 12,  # Annual discount cost
                    estimated_effectiveness=0.40,
                    expected_roi=(monthly_revenue * 12 * 0.40 - monthly_revenue * 0.10 * 12)
                    / (monthly_revenue * 0.10 * 12),
                    implementation_time="1 week",
                    target_customers=[customer_id],
                )
            )

        # Payment method optimization
        payment_method = customer_data.get("PaymentMethod", "")
        if payment_method == "Electronic check" and churn_probability >= 0.3:
            actions.append(
                RetentionAction(
                    action_id="auto_pay_incentive",
                    title="Auto-Pay Enrollment Incentive",
                    description="Offer $5/month discount for switching to automatic payment",
                    priority="medium",
                    estimated_cost=5.0 * 12,  # $5/month for 12 months
                    estimated_effectiveness=0.20,
                    expected_roi=(monthly_revenue * 12 * 0.20 - 60) / 60,
                    implementation_time="1-2 days",
                    target_customers=[customer_id],
                )
            )

        # Service enhancement based on SHAP explanations
        for explanation in explanations[:3]:  # Top 3 features
            feature = explanation.get("feature", "")
            shap_value = explanation.get("shap_value", 0)

            if shap_value > 0.1:  # Feature increases churn risk
                if "Contract" in feature and "Month-to-month" in str(feature):
                    # Already handled above
                    continue
                elif "OnlineSecurity" in feature or "TechSupport" in feature:
                    actions.append(
                        RetentionAction(
                            action_id=f"service_enhancement_{feature}",
                            title=f"Service Enhancement: {feature}",
                            description=f"Offer complimentary {feature} service to improve satisfaction",
                            priority="medium",
                            estimated_cost=10.0 * 6,  # $10/month for 6 months
                            estimated_effectiveness=0.15,
                            expected_roi=(monthly_revenue * 12 * 0.15 - 60) / 60,
                            implementation_time="1 week",
                            target_customers=[customer_id],
                        )
                    )

        # Sort by priority and ROI
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        actions.sort(key=lambda a: (priority_order.get(a.priority, 99), -a.expected_roi))

        return actions[:5]  # Return top 5 actions

    def _generate_cohort_actions(
        self, cohort: dict[str, Any], customer_count: int
    ) -> list[RetentionAction]:
        """Generate actions for a cohort."""
        actions = []
        risk_level = cohort.get("risk_level", "medium")
        recommended_actions = cohort.get("recommended_actions", [])

        # Convert recommended actions to RetentionAction objects
        for i, action_desc in enumerate(recommended_actions[:5]):
            # Estimate costs based on action type
            if "discount" in action_desc.lower():
                cost = cohort.get("avg_monthly_revenue", 50) * 0.15 * customer_count * 3
            elif "call" in action_desc.lower() or "outreach" in action_desc.lower():
                cost = self.retention_specialist_cost * customer_count * 0.5  # 50% reach
            else:
                cost = 10.0 * customer_count

            actions.append(
                RetentionAction(
                    action_id=f"cohort_action_{i}",
                    title=action_desc,
                    description=action_desc,
                    priority=risk_level if isinstance(risk_level, str) else risk_level.value,
                    estimated_cost=cost,
                    estimated_effectiveness=0.25,  # Average effectiveness
                    expected_roi=2.0,  # Estimated 2x ROI
                    implementation_time="1-2 days",
                    target_customers=[],  # Would be populated with actual customer IDs
                )
            )

        return actions

    def _determine_strategy(
        self, churn_probability: float, monthly_revenue: float, tenure: int
    ) -> str:
        """Determine overall retention strategy."""
        if churn_probability >= 0.75:
            return "Immediate intervention required - deploy all available retention tools"
        elif churn_probability >= 0.5:
            return "Proactive retention - targeted offers and engagement"
        elif churn_probability >= 0.3:
            return "Preventive engagement - maintain satisfaction and address concerns early"
        else:
            return "Maintain current service levels - focus on value delivery"
