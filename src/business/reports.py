"""
Executive report generator for business insights and presentation materials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from src.business.cohorts import CustomerCohort
from src.business.kpis import KPIMetrics
from src.business.playbooks import RetentionPlaybook
from src.business.roi import ROIMetrics

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class ExecutiveReportGenerator:
    """Generates executive-ready reports and presentation materials."""

    def __init__(self, output_dir: Path | str = "reports/business"):
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_executive_summary(
        self,
        kpi_metrics: KPIMetrics,
        roi_metrics: ROIMetrics,
        top_cohorts: list[CustomerCohort],
        top_playbooks: list[RetentionPlaybook],
    ) -> dict[str, Any]:
        """Generate executive summary report.

        Args:
            kpi_metrics: Current KPI metrics.
            roi_metrics: ROI metrics for retention initiatives.
            top_cohorts: Top priority customer cohorts.
            top_playbooks: Top priority retention playbooks.

        Returns:
            Executive summary dictionary.
        """
        summary = {
            "report_type": "executive_summary",
            "timestamp": kpi_metrics.timestamp,
            "key_findings": {
                "customers_at_risk": kpi_metrics.customers_at_risk,
                "revenue_at_risk": f"${kpi_metrics.revenue_at_risk:,.2f}",
                "revenue_at_risk_percentage": f"{kpi_metrics.revenue_at_risk_percentage:.1f}%",
                "estimated_annual_churn": kpi_metrics.estimated_annual_churn,
                "estimated_revenue_loss": f"${kpi_metrics.estimated_revenue_loss:,.2f}",
            },
            "roi_analysis": {
                "intervention_cost": f"${roi_metrics.intervention_cost:,.2f}",
                "estimated_revenue_saved": f"${roi_metrics.estimated_revenue_saved:,.2f}",
                "net_benefit": f"${roi_metrics.net_benefit:,.2f}",
                "roi_percentage": f"{roi_metrics.roi_percentage:.1f}%",
                "payback_period_months": f"{roi_metrics.payback_period_months:.1f}",
            },
            "top_priorities": {
                "critical_cohorts": [
                    {
                        "name": cohort.name,
                        "customer_count": cohort.customer_count,
                        "total_annual_revenue": f"${cohort.total_annual_revenue:,.2f}",
                        "avg_churn_probability": f"{cohort.avg_churn_probability:.1%}",
                    }
                    for cohort in top_cohorts[:5]
                ],
                "high_impact_playbooks": [
                    {
                        "playbook_id": playbook.playbook_id,
                        "churn_probability": f"{playbook.churn_probability:.1%}",
                        "monthly_revenue": f"${playbook.monthly_revenue:,.2f}",
                        "net_roi": f"{playbook.net_roi:.1f}x",
                    }
                    for playbook in top_playbooks[:5]
                ],
            },
            "recommendations": self._generate_recommendations(
                kpi_metrics, roi_metrics, top_cohorts
            ),
        }

        return summary

    def generate_cohort_analysis_report(self, cohorts: list[CustomerCohort]) -> dict[str, Any]:
        """Generate detailed cohort analysis report.

        Args:
            cohorts: List of customer cohorts.

        Returns:
            Cohort analysis report.
        """
        # Calculate summary statistics
        total_customers = sum(cohort.customer_count for cohort in cohorts)
        total_revenue = sum(cohort.total_annual_revenue for cohort in cohorts)

        # Group by risk level

        by_risk: dict[str, dict[str, Any]] = {}
        for cohort in cohorts:
            risk = cohort.risk_level.value
            if risk not in by_risk:
                by_risk[risk] = {
                    "cohorts": [],
                    "total_customers": 0,
                    "total_revenue": 0.0,
                }
            by_risk[risk]["cohorts"].append(cohort.to_dict())
            by_risk[risk]["total_customers"] += cohort.customer_count
            by_risk[risk]["total_revenue"] += cohort.total_annual_revenue

        report = {
            "report_type": "cohort_analysis",
            "summary": {
                "total_cohorts": len(cohorts),
                "total_customers": total_customers,
                "total_annual_revenue": total_revenue,
            },
            "by_risk_level": by_risk,
            "all_cohorts": [cohort.to_dict() for cohort in cohorts],
        }

        return report

    def generate_retention_strategy_report(
        self,
        cohorts: list[CustomerCohort],
        playbooks: list[RetentionPlaybook],
    ) -> dict[str, Any]:
        """Generate retention strategy report.

        Args:
            cohorts: List of customer cohorts.
            playbooks: List of retention playbooks.

        Returns:
            Retention strategy report.
        """
        # Aggregate actions by type
        action_types = {}
        for playbook in playbooks:
            for action in playbook.actions:
                action_type = action.title.split(":")[0] if ":" in action.title else action.title
                if action_type not in action_types:
                    action_types[action_type] = {
                        "count": 0,
                        "total_cost": 0.0,
                        "total_effectiveness": 0.0,
                        "total_roi": 0.0,
                    }
                action_types[action_type]["count"] += 1
                action_types[action_type]["total_cost"] += action.estimated_cost
                action_types[action_type]["total_effectiveness"] += action.estimated_effectiveness
                action_types[action_type]["total_roi"] += action.expected_roi

        # Calculate averages
        for action_type in action_types:
            count = action_types[action_type]["count"]
            action_types[action_type]["avg_cost"] = action_types[action_type]["total_cost"] / count
            action_types[action_type]["avg_effectiveness"] = (
                action_types[action_type]["total_effectiveness"] / count
            )
            action_types[action_type]["avg_roi"] = action_types[action_type]["total_roi"] / count

        report = {
            "report_type": "retention_strategy",
            "strategy_overview": {
                "total_cohorts": len(cohorts),
                "total_playbooks": len(playbooks),
                "total_actions": sum(len(p.actions) for p in playbooks),
            },
            "action_effectiveness": action_types,
            "cohort_strategies": [
                {
                    "cohort_id": cohort.cohort_id,
                    "cohort_name": cohort.name,
                    "recommended_actions": cohort.recommended_actions,
                    "customer_count": cohort.customer_count,
                }
                for cohort in cohorts
            ],
        }

        return report

    def save_report(self, report: dict[str, Any], filename: str) -> Path:
        """Save report to JSON file.

        Args:
            report: Report dictionary.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return filepath

    def generate_visualizations(
        self,
        kpi_metrics: KPIMetrics,
        roi_metrics: ROIMetrics,
        cohorts: list[CustomerCohort],
    ) -> dict[str, Path]:
        """Generate visualization charts.

        Args:
            kpi_metrics: KPI metrics.
            roi_metrics: ROI metrics.
            cohorts: Customer cohorts.

        Returns:
            Dictionary mapping chart names to file paths.
        """
        charts = {}

        # 1. Risk Distribution Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        risk_counts = {
            "Critical": kpi_metrics.critical_risk_customers,
            "High": kpi_metrics.high_risk_customers,
            "Medium": kpi_metrics.medium_risk_customers,
            "Low": kpi_metrics.low_risk_customers,
        }
        ax.bar(
            list(risk_counts.keys()),
            list(risk_counts.values()),
            color=["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"],
        )
        ax.set_title("Customer Risk Distribution", fontsize=16, fontweight="bold")
        ax.set_xlabel("Risk Level", fontsize=12)
        ax.set_ylabel("Number of Customers", fontsize=12)
        plt.tight_layout()
        chart_path = self.output_dir / "risk_distribution.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts["risk_distribution"] = chart_path

        # 2. Revenue at Risk by Cohort
        if cohorts:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_cohorts = sorted(cohorts, key=lambda c: c.total_annual_revenue, reverse=True)[:10]
            cohort_names = [c.name[:30] for c in top_cohorts]  # Truncate long names
            revenues = [c.total_annual_revenue for c in top_cohorts]
            colors = [
                (
                    "#e74c3c"
                    if c.risk_level.value == "critical"
                    else "#f39c12"
                    if c.risk_level.value == "high"
                    else "#f1c40f"
                )
                for c in top_cohorts
            ]
            ax.barh(cohort_names, revenues, color=colors)
            ax.set_title("Revenue at Risk by Cohort (Top 10)", fontsize=16, fontweight="bold")
            ax.set_xlabel("Annual Revenue ($)", fontsize=12)
            plt.tight_layout()
            chart_path = self.output_dir / "revenue_by_cohort.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()
            charts["revenue_by_cohort"] = chart_path

        # 3. ROI Analysis Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ROI Breakdown
        categories = ["Revenue Saved", "Intervention Cost", "Net Benefit"]
        values = [
            roi_metrics.estimated_revenue_saved,
            roi_metrics.intervention_cost,
            roi_metrics.net_benefit,
        ]
        colors_roi = ["#2ecc71", "#e74c3c", "#3498db"]
        ax1.bar(categories, values, color=colors_roi)
        ax1.set_title("ROI Breakdown", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Amount ($)", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)

        # ROI Percentage
        ax2.barh(["ROI"], [roi_metrics.roi_percentage], color="#27ae60")
        ax2.set_title(
            f"Return on Investment: {roi_metrics.roi_percentage:.1f}%",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("ROI (%)", fontsize=12)

        plt.tight_layout()
        chart_path = self.output_dir / "roi_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts["roi_analysis"] = chart_path

        # 4. KPI Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # At Risk Percentage
        axes[0, 0].pie(
            [kpi_metrics.customers_at_risk, kpi_metrics.low_risk_customers],
            labels=["At Risk", "Low Risk"],
            autopct="%1.1f%%",
            colors=["#e74c3c", "#2ecc71"],
            startangle=90,
        )
        axes[0, 0].set_title("Customer Risk Overview", fontweight="bold")

        # Revenue at Risk
        revenue_data = {
            "At Risk": kpi_metrics.revenue_at_risk,
            "Safe": kpi_metrics.total_monthly_revenue - kpi_metrics.revenue_at_risk,
        }
        axes[0, 1].bar(revenue_data.keys(), revenue_data.values(), color=["#e74c3c", "#2ecc71"])
        axes[0, 1].set_title("Monthly Revenue at Risk", fontweight="bold")
        axes[0, 1].set_ylabel("Revenue ($)")

        # Estimated Churn
        axes[1, 0].bar(
            ["Monthly", "Annual"],
            [kpi_metrics.estimated_monthly_churn, kpi_metrics.estimated_annual_churn],
            color="#f39c12",
        )
        axes[1, 0].set_title("Estimated Churn", fontweight="bold")
        axes[1, 0].set_ylabel("Number of Customers")

        # Intervention Readiness
        axes[1, 1].barh(
            ["Readiness Score"],
            [kpi_metrics.intervention_readiness_score or 0],
            color="#3498db",
        )
        axes[1, 1].set_title("Intervention Readiness Score", fontweight="bold")
        axes[1, 1].set_xlabel("Score (0-100)")
        axes[1, 1].set_xlim(0, 100)

        plt.suptitle("Business KPI Dashboard", fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout()
        chart_path = self.output_dir / "kpi_dashboard.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts["kpi_dashboard"] = chart_path

        return charts

    def _generate_recommendations(
        self,
        kpi_metrics: KPIMetrics,
        roi_metrics: ROIMetrics,
        top_cohorts: list[CustomerCohort],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # High-level recommendations
        if kpi_metrics.customers_at_risk > kpi_metrics.total_customers * 0.3:
            recommendations.append(
                f"URGENT: {kpi_metrics.customers_at_risk} customers ({kpi_metrics.at_risk_percentage:.1f}%) "
                f"are at risk. Immediate intervention required."
            )

        if kpi_metrics.revenue_at_risk_percentage > 30:
            recommendations.append(
                f"CRITICAL: ${kpi_metrics.revenue_at_risk:,.0f} in monthly revenue "
                f"({kpi_metrics.revenue_at_risk_percentage:.1f}%) is at risk."
            )

        if roi_metrics.roi_percentage > 200:
            recommendations.append(
                f"STRONG ROI: Retention initiatives show {roi_metrics.roi_percentage:.1f}% ROI. "
                f"Scale intervention programs."
            )

        # Cohort-specific recommendations
        critical_cohorts = [c for c in top_cohorts if c.risk_level.value == "critical"]
        if critical_cohorts:
            recommendations.append(
                f"Focus on {len(critical_cohorts)} critical-risk cohorts with "
                f"${sum(c.total_annual_revenue for c in critical_cohorts):,.0f} in annual revenue."
            )

        # Model performance recommendations
        if kpi_metrics.model_performance_roc_auc and kpi_metrics.model_performance_roc_auc < 0.80:
            recommendations.append(
                "Consider model retraining: Current model performance may be degrading."
            )

        return recommendations
