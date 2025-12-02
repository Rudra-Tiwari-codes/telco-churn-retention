"""
Experiment design and A/B testing framework for retention interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class ExperimentStatus(str, Enum):
    """Experiment status."""

    DESIGNED = "designed"
    RUNNING = "running"
    COMPLETED = "completed"
    ANALYZED = "analyzed"


@dataclass
class ExperimentDesign:
    """Design for a retention intervention experiment."""

    experiment_id: str
    name: str
    description: str
    hypothesis: str
    intervention_type: str  # e.g., "discount", "outreach", "service_upgrade"
    target_cohort: str
    sample_size_control: int
    sample_size_treatment: int
    duration_weeks: int
    success_metric: str  # e.g., "churn_rate", "retention_rate", "revenue"
    expected_effect_size: float
    statistical_power: float = 0.80
    significance_level: float = 0.05
    status: ExperimentStatus = ExperimentStatus.DESIGNED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "intervention_type": self.intervention_type,
            "target_cohort": self.target_cohort,
            "sample_size_control": self.sample_size_control,
            "sample_size_treatment": self.sample_size_treatment,
            "duration_weeks": self.duration_weeks,
            "success_metric": self.success_metric,
            "expected_effect_size": self.expected_effect_size,
            "statistical_power": self.statistical_power,
            "significance_level": self.significance_level,
            "status": self.status.value,
        }


@dataclass
class ExperimentResults:
    """Results from an A/B test experiment."""

    experiment_id: str
    control_group_size: int
    treatment_group_size: int
    control_metric_value: float
    treatment_metric_value: float
    absolute_lift: float
    relative_lift_percentage: float
    p_value: float
    is_statistically_significant: bool
    confidence_interval_lower: float
    confidence_interval_upper: float
    effect_size: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "control_group_size": self.control_group_size,
            "treatment_group_size": self.treatment_group_size,
            "control_metric_value": self.control_metric_value,
            "treatment_metric_value": self.treatment_metric_value,
            "absolute_lift": self.absolute_lift,
            "relative_lift_percentage": self.relative_lift_percentage,
            "p_value": self.p_value,
            "is_statistically_significant": self.is_statistically_significant,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "effect_size": self.effect_size,
            "interpretation": self.interpretation,
        }


class ExperimentDesigner:
    """Designs experiments for retention interventions."""

    def calculate_sample_size(
        self,
        baseline_rate: float,
        expected_lift: float,
        power: float = 0.80,
        alpha: float = 0.05,
        two_tailed: bool = True,
    ) -> int:
        """Calculate required sample size for A/B test.

        Args:
            baseline_rate: Baseline metric rate (e.g., churn rate).
            expected_lift: Expected lift (e.g., 0.10 for 10% reduction).
            power: Statistical power (1 - beta).
            alpha: Significance level.
            two_tailed: Whether test is two-tailed.

        Returns:
            Required sample size per group.
        """
        # Effect size
        expected_lift / baseline_rate if baseline_rate > 0 else 0

        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2) if two_tailed else stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        # Sample size calculation (proportions)
        p1 = baseline_rate
        p2 = baseline_rate * (1 - expected_lift)

        pooled_p = (p1 + p2) / 2
        pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / 1 + 1 / 1))  # Equal groups

        numerator = (z_alpha * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)) + z_beta * pooled_se) ** 2
        denominator = (p1 - p2) ** 2

        sample_size = int(np.ceil(numerator / denominator)) if denominator > 0 else 1000

        return max(sample_size, 30)  # Minimum sample size

    def design_experiment(
        self,
        name: str,
        description: str,
        hypothesis: str,
        intervention_type: str,
        target_cohort: str,
        baseline_churn_rate: float,
        expected_churn_reduction: float,
        duration_weeks: int = 4,
        power: float = 0.80,
        alpha: float = 0.05,
    ) -> ExperimentDesign:
        """Design a retention intervention experiment.

        Args:
            name: Experiment name.
            description: Experiment description.
            hypothesis: Test hypothesis.
            intervention_type: Type of intervention.
            target_cohort: Target customer cohort.
            baseline_churn_rate: Baseline churn rate.
            expected_churn_reduction: Expected churn reduction (e.g., 0.10 for 10%).
            duration_weeks: Experiment duration in weeks.
            power: Statistical power.
            alpha: Significance level.

        Returns:
            Experiment design.
        """
        sample_size = self.calculate_sample_size(
            baseline_rate=baseline_churn_rate,
            expected_lift=expected_churn_reduction,
            power=power,
            alpha=alpha,
        )

        experiment_id = f"exp_{name.lower().replace(' ', '_')}"

        return ExperimentDesign(
            experiment_id=experiment_id,
            name=name,
            description=description,
            hypothesis=hypothesis,
            intervention_type=intervention_type,
            target_cohort=target_cohort,
            sample_size_control=sample_size,
            sample_size_treatment=sample_size,
            duration_weeks=duration_weeks,
            success_metric="churn_rate",
            expected_effect_size=expected_churn_reduction,
            statistical_power=power,
            significance_level=alpha,
        )


class ExperimentAnalyzer:
    """Analyzes A/B test experiment results."""

    def analyze_experiment(
        self,
        control_group: pd.DataFrame,
        treatment_group: pd.DataFrame,
        metric_column: str = "churned",
        experiment_id: str = "experiment",
    ) -> ExperimentResults:
        """Analyze A/B test results.

        Args:
            control_group: Control group data.
            treatment_group: Treatment group data.
            metric_column: Column name for the metric.
            experiment_id: Experiment identifier.

        Returns:
            Experiment results.
        """
        # Calculate metric values
        control_metric = control_group[metric_column].mean() if len(control_group) > 0 else 0.0
        treatment_metric = (
            treatment_group[metric_column].mean() if len(treatment_group) > 0 else 0.0
        )

        # Calculate lift
        absolute_lift = treatment_metric - control_metric
        relative_lift = (
            (treatment_metric - control_metric) / control_metric * 100
            if control_metric > 0
            else 0.0
        )

        # Statistical test (two-proportion z-test)
        n_control = len(control_group)
        n_treatment = len(treatment_group)

        if n_control > 0 and n_treatment > 0:
            # Two-proportion z-test
            p1 = control_metric
            p2 = treatment_metric

            pooled_p = (p1 * n_control + p2 * n_treatment) / (n_control + n_treatment)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n_control + 1 / n_treatment))

            if se > 0:
                z_score = (p2 - p1) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed
            else:
                p_value = 1.0
                z_score = 0.0

            # Confidence interval (95%)
            diff_se = np.sqrt(p1 * (1 - p1) / n_control + p2 * (1 - p2) / n_treatment)
            margin_error = 1.96 * diff_se
            ci_lower = absolute_lift - margin_error
            ci_upper = absolute_lift + margin_error

            # Effect size (Cohen's h)
            effect_size = (
                2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))) if p1 > 0 and p2 > 0 else 0.0
            )

            is_significant = p_value < 0.05
        else:
            p_value = 1.0
            ci_lower = 0.0
            ci_upper = 0.0
            effect_size = 0.0
            is_significant = False

        # Interpretation
        if is_significant:
            if absolute_lift < 0:
                interpretation = (
                    f"Statistically significant improvement: {abs(relative_lift):.1f}% "
                    f"reduction in {metric_column} (p={p_value:.4f})"
                )
            else:
                interpretation = (
                    f"Statistically significant increase: {relative_lift:.1f}% "
                    f"increase in {metric_column} (p={p_value:.4f})"
                )
        else:
            interpretation = (
                f"No statistically significant difference (p={p_value:.4f}). "
                f"Lift: {relative_lift:.1f}%"
            )

        return ExperimentResults(
            experiment_id=experiment_id,
            control_group_size=n_control,
            treatment_group_size=n_treatment,
            control_metric_value=control_metric,
            treatment_metric_value=treatment_metric,
            absolute_lift=absolute_lift,
            relative_lift_percentage=relative_lift,
            p_value=p_value,
            is_statistically_significant=is_significant,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            effect_size=effect_size,
            interpretation=interpretation,
        )

    def simulate_experiment_results(
        self,
        design: ExperimentDesign,
        baseline_rate: float,
        true_effect: float,
    ) -> ExperimentResults:
        """Simulate experiment results for planning.

        Args:
            design: Experiment design.
            baseline_rate: Baseline metric rate.
            true_effect: True effect size.

        Returns:
            Simulated experiment results.
        """
        # Simulate control group
        np.random.seed(42)
        control_churned = np.random.binomial(1, baseline_rate, design.sample_size_control)
        control_df = pd.DataFrame({"churned": control_churned})

        # Simulate treatment group (with effect)
        treatment_rate = baseline_rate * (1 - true_effect)
        treatment_churned = np.random.binomial(1, treatment_rate, design.sample_size_treatment)
        treatment_df = pd.DataFrame({"churned": treatment_churned})

        return self.analyze_experiment(
            control_group=control_df,
            treatment_group=treatment_df,
            metric_column="churned",
            experiment_id=design.experiment_id,
        )
