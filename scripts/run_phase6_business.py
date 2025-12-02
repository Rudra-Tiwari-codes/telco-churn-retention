"""
Phase 6: Business Intelligence & Executive Delivery

This script generates comprehensive business intelligence reports including:
- Customer cohort analysis
- Retention playbooks
- ROI calculations
- KPI tracking
- Experiment designs
- Customer journey mapping
- Predictive CLV analysis
- Executive reports and visualizations
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.api.service import ModelService
from src.business.clv import PredictiveCLVCalculator
from src.business.cohorts import CohortAnalyzer
from src.business.experiments import ExperimentDesigner
from src.business.journey import CustomerJourneyMapper
from src.business.kpis import KPITracker
from src.business.playbooks import RetentionPlaybookGenerator
from src.business.reports import ExecutiveReportGenerator
from src.business.roi import ROICalculator
from src.data.ingestion import clean_dataset, load_raw_dataset

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 6: Business Intelligence & Executive Delivery"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/telco_data_28_11_2025.csv"),
        help="Path to customer data CSV",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/business"),
        help="Directory to save business reports",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for analysis (None = all customers)",
    )
    parser.add_argument(
        "--churn-threshold",
        type=float,
        default=0.5,
        help="Churn probability threshold for at-risk classification",
    )
    parser.add_argument(
        "--intervention-cost",
        type=float,
        default=50.0,
        help="Average cost per customer intervention",
    )
    return parser.parse_args()


def load_data_and_predictions(
    data_path: Path,
    model_dir: Path,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load customer data and generate predictions.

    Args:
        data_path: Path to customer data CSV.
        model_dir: Directory containing trained models.
        sample_size: Optional sample size limit.

    Returns:
        Tuple of (customers_df, predictions_df).
    """
    console.print("[bold cyan]Loading customer data...[/bold cyan]")

    # Load and clean data
    df = load_raw_dataset(data_path)
    df = clean_dataset(df)

    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        console.print(f"[yellow]Sampled {sample_size} customers for analysis[/yellow]")

    console.print(f"[green]Loaded {len(df)} customers[/green]")

    # Load model service
    console.print("[bold cyan]Loading model service...[/bold cyan]")
    model_service = ModelService(model_dir=model_dir, threshold=0.5)

    if not model_service.is_ready():
        raise RuntimeError("Model service not ready. Please ensure models are trained (Phase 3).")

    console.print("[green]Model service ready[/green]")

    # Generate predictions
    console.print("[bold cyan]Generating predictions...[/bold cyan]")
    predictions = []

    for idx, row in df.iterrows():
        customer_dict = row.to_dict()
        result = model_service.predict(customer_dict, include_explanation=False)
        predictions.append(
            {
                "customerID": customer_dict["customerID"],
                "churn_probability": result["churn_probability"],
            }
        )

        if (idx + 1) % 100 == 0:
            console.print(f"  Processed {idx + 1}/{len(df)} customers...")

    predictions_df = pd.DataFrame(predictions)
    console.print(f"[green]Generated predictions for {len(predictions_df)} customers[/green]")

    return df, predictions_df


def main() -> None:
    """Run Phase 6 business intelligence workflow."""
    args = parse_args()

    console.print("\n" + "=" * 80)
    console.print("[bold green]Phase 6: Business Intelligence & Executive Delivery[/bold green]")
    console.print("=" * 80 + "\n")

    # Create output directory
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Output directory:[/bold] {output_dir}\n")

    try:
        # Load data and generate predictions
        customers_df, predictions_df = load_data_and_predictions(
            args.data_path, args.model_dir, args.sample_size
        )

        # 1. Cohort Analysis
        console.print("\n[bold cyan]1. Customer Cohort Analysis[/bold cyan]")
        cohort_analyzer = CohortAnalyzer()
        cohorts = cohort_analyzer.create_cohorts(customers_df, predictions_df)
        console.print(f"[green]Created {len(cohorts)} customer cohorts[/green]")

        # Save cohort analysis
        cohorts_data = [cohort.to_dict() for cohort in cohorts]
        with open(output_dir / "cohorts.json", "w") as f:
            json.dump(cohorts_data, f, indent=2, default=str)

        # Display top cohorts
        top_cohorts = sorted(cohorts, key=lambda c: c.total_annual_revenue, reverse=True)[:5]
        table = Table(title="Top 5 Cohorts by Revenue")
        table.add_column("Cohort", style="cyan")
        table.add_column("Customers", justify="right")
        table.add_column("Avg Churn Prob", justify="right")
        table.add_column("Annual Revenue", justify="right")
        for cohort in top_cohorts:
            table.add_row(
                cohort.name,
                str(cohort.customer_count),
                f"{cohort.avg_churn_probability:.1%}",
                f"${cohort.total_annual_revenue:,.0f}",
            )
        console.print(table)

        # 2. KPI Tracking
        console.print("\n[bold cyan]2. Business KPI Calculation[/bold cyan]")
        kpi_tracker = KPITracker()
        kpi_metrics = kpi_tracker.calculate_kpis(
            customers=customers_df,
            predictions=predictions_df,
            model_metrics={"roc_auc": 0.85, "f1": 0.63},  # Example metrics
        )
        console.print("[green]KPI metrics calculated[/green]")

        # Display KPIs
        kpi_table = Table(title="Key Performance Indicators")
        kpi_table.add_column("Metric", style="cyan")
        kpi_table.add_column("Value", justify="right")
        kpi_table.add_row("Total Customers", str(kpi_metrics.total_customers))
        kpi_table.add_row(
            "Customers at Risk",
            f"{kpi_metrics.customers_at_risk} ({kpi_metrics.at_risk_percentage:.1f}%)",
        )
        kpi_table.add_row(
            "Revenue at Risk",
            f"${kpi_metrics.revenue_at_risk:,.0f} ({kpi_metrics.revenue_at_risk_percentage:.1f}%)",
        )
        kpi_table.add_row("Estimated Annual Churn", str(kpi_metrics.estimated_annual_churn))
        kpi_table.add_row("Estimated Revenue Loss", f"${kpi_metrics.estimated_revenue_loss:,.0f}")
        console.print(kpi_table)

        # Save KPIs
        with open(output_dir / "kpis.json", "w") as f:
            json.dump(kpi_metrics.to_dict(), f, indent=2, default=str)

        # 3. ROI Analysis
        console.print("\n[bold cyan]3. ROI Analysis[/bold cyan]")
        roi_calculator = ROICalculator()
        roi_metrics = roi_calculator.calculate_roi(
            customers=customers_df,
            predictions=predictions_df,
            intervention_cost_per_customer=args.intervention_cost,
            churn_threshold=args.churn_threshold,
        )
        console.print("[green]ROI metrics calculated[/green]")

        # Display ROI
        roi_table = Table(title="ROI Analysis")
        roi_table.add_column("Metric", style="cyan")
        roi_table.add_column("Value", justify="right")
        roi_table.add_row("Customers at Risk", str(roi_metrics.total_customers_at_risk))
        roi_table.add_row("Intervention Cost", f"${roi_metrics.intervention_cost:,.0f}")
        roi_table.add_row("Estimated Revenue Saved", f"${roi_metrics.estimated_revenue_saved:,.0f}")
        roi_table.add_row("Net Benefit", f"${roi_metrics.net_benefit:,.0f}")
        roi_table.add_row("ROI", f"{roi_metrics.roi_percentage:.1f}%")
        roi_table.add_row("Payback Period", f"{roi_metrics.payback_period_months:.1f} months")
        console.print(roi_table)

        # Save ROI
        with open(output_dir / "roi.json", "w") as f:
            json.dump(roi_metrics.to_dict(), f, indent=2, default=str)

        # 4. Customer Journey Mapping
        console.print("\n[bold cyan]4. Customer Journey Analysis[/bold cyan]")
        journey_mapper = CustomerJourneyMapper()
        journey_stages = journey_mapper.analyze_journey_stages(customers_df, predictions_df)
        console.print(f"[green]Analyzed {len(journey_stages)} journey stages[/green]")

        journey_insights = journey_mapper.generate_journey_insights(journey_stages)
        with open(output_dir / "journey_analysis.json", "w") as f:
            json.dump(
                {
                    "stages": [stage.to_dict() for stage in journey_stages],
                    "insights": journey_insights,
                },
                f,
                indent=2,
                default=str,
            )

        # 5. Predictive CLV
        console.print("\n[bold cyan]5. Predictive Customer Lifetime Value[/bold cyan]")
        clv_calculator = PredictiveCLVCalculator()
        clv_metrics = clv_calculator.calculate_clv_for_customers(customers_df, predictions_df)
        console.print(f"[green]Calculated CLV for {len(clv_metrics)} customers[/green]")

        portfolio_clv = clv_calculator.calculate_portfolio_clv(clv_metrics)
        console.print(f"[green]Portfolio CLV: ${portfolio_clv['total_clv']:,.0f}[/green]")
        console.print(f"[green]Average CLV: ${portfolio_clv['avg_clv']:,.0f}[/green]")

        # Save CLV
        with open(output_dir / "clv_analysis.json", "w") as f:
            json.dump(
                {
                    "individual_clv": [m.to_dict() for m in clv_metrics[:100]],  # Sample
                    "portfolio_summary": portfolio_clv,
                },
                f,
                indent=2,
                default=str,
            )

        # 6. Retention Playbooks
        console.print("\n[bold cyan]6. Retention Playbook Generation[/bold cyan]")
        playbook_generator = RetentionPlaybookGenerator()

        # Generate playbooks for top at-risk customers
        at_risk_df = customers_df.merge(predictions_df, on="customerID").query(
            f"churn_probability >= {args.churn_threshold}"
        )
        top_at_risk = at_risk_df.nlargest(10, "churn_probability")

        playbooks = []
        for _, row in top_at_risk.iterrows():
            customer_dict = row.to_dict()
            playbook = playbook_generator.generate_playbook(
                customer_data=customer_dict,
                churn_probability=row["churn_probability"],
                monthly_revenue=row.get("MonthlyCharges", 0),
                tenure=row.get("tenure", 0),
                explanations=[],  # Would include SHAP explanations in full implementation
            )
            playbooks.append(playbook)

        console.print(f"[green]Generated {len(playbooks)} retention playbooks[/green]")

        # Save playbooks
        with open(output_dir / "playbooks.json", "w") as f:
            json.dump([p.to_dict() for p in playbooks], f, indent=2, default=str)

        # 7. Experiment Design
        console.print("\n[bold cyan]7. Experiment Design[/bold cyan]")
        experiment_designer = ExperimentDesigner()

        # Design experiment for critical-risk cohort
        critical_cohort = next((c for c in cohorts if c.risk_level.value == "critical"), None)
        if critical_cohort:
            experiment = experiment_designer.design_experiment(
                name="Critical Risk Retention Intervention",
                description="A/B test for personalized discount intervention",
                hypothesis="Personalized 20% discount will reduce churn by 15%",
                intervention_type="discount",
                target_cohort=critical_cohort.cohort_id,
                baseline_churn_rate=critical_cohort.avg_churn_probability,
                expected_churn_reduction=0.15,
            )
            console.print(f"[green]Designed experiment: {experiment.name}[/green]")
            console.print(f"  Sample size: {experiment.sample_size_control} per group")
            console.print(f"  Duration: {experiment.duration_weeks} weeks")

            with open(output_dir / "experiment_design.json", "w") as f:
                json.dump(experiment.to_dict(), f, indent=2, default=str)

        # 8. Executive Report Generation
        console.print("\n[bold cyan]8. Executive Report Generation[/bold cyan]")
        report_generator = ExecutiveReportGenerator(output_dir=output_dir)

        # Generate executive summary
        executive_summary = report_generator.generate_executive_summary(
            kpi_metrics=kpi_metrics,
            roi_metrics=roi_metrics,
            top_cohorts=cohorts[:10],
            top_playbooks=playbooks[:10],
        )

        report_generator.save_report(executive_summary, "executive_summary.json")
        console.print("[green]Executive summary generated[/green]")

        # Generate cohort analysis report
        cohort_report = report_generator.generate_cohort_analysis_report(cohorts)
        report_generator.save_report(cohort_report, "cohort_analysis.json")
        console.print("[green]Cohort analysis report generated[/green]")

        # Generate retention strategy report
        strategy_report = report_generator.generate_retention_strategy_report(cohorts, playbooks)
        report_generator.save_report(strategy_report, "retention_strategy.json")
        console.print("[green]Retention strategy report generated[/green]")

        # Generate visualizations
        console.print("\n[bold cyan]9. Generating Visualizations[/bold cyan]")
        charts = report_generator.generate_visualizations(
            kpi_metrics=kpi_metrics,
            roi_metrics=roi_metrics,
            cohorts=cohorts,
        )
        console.print(f"[green]Generated {len(charts)} visualization charts[/green]")
        for chart_name, chart_path in charts.items():
            console.print(f"  - {chart_name}: {chart_path}")

        # Summary
        console.print("\n" + "=" * 80)
        console.print("[bold green]Phase 6 Complete![/bold green]")
        console.print("=" * 80)
        console.print(f"\n[bold]Reports saved to:[/bold] {output_dir}")
        console.print("\n[bold]Generated Reports:[/bold]")
        console.print("  - cohorts.json: Customer cohort analysis")
        console.print("  - kpis.json: Business KPI metrics")
        console.print("  - roi.json: ROI analysis")
        console.print("  - journey_analysis.json: Customer journey mapping")
        console.print("  - clv_analysis.json: Predictive CLV analysis")
        console.print("  - playbooks.json: Retention playbooks")
        console.print("  - experiment_design.json: A/B test experiment design")
        console.print("  - executive_summary.json: Executive summary report")
        console.print("  - cohort_analysis.json: Detailed cohort analysis")
        console.print("  - retention_strategy.json: Retention strategy report")
        console.print("\n[bold]Visualizations:[/bold]")
        for chart_name in charts.keys():
            console.print(f"  - {chart_name}.png")

    except Exception as e:
        console.print(f"[red][FAIL] Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
