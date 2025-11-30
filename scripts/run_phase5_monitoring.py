"""Phase 5 monitoring workflow script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from datetime import UTC, datetime

import pandas as pd
from rich.console import Console

from src.monitoring.alerts import AlertManager
from src.monitoring.dashboard import MonitoringDashboard
from src.monitoring.drift import DriftDetector
from src.monitoring.performance import PerformanceMonitor

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 5 monitoring workflow")
    parser.add_argument(
        "--reference-data",
        type=Path,
        required=True,
        help="Path to reference/training data (parquet)",
    )
    parser.add_argument(
        "--current-data",
        type=Path,
        required=True,
        help="Path to current/production data (parquet)",
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        help="Path to reference predictions (optional, CSV with 'predictions' column)",
    )
    parser.add_argument(
        "--current-predictions",
        type=Path,
        help="Path to current predictions (optional, CSV with 'predictions' column)",
    )
    parser.add_argument(
        "--reference-labels",
        type=Path,
        help="Path to reference labels (optional, CSV with 'labels' column)",
    )
    parser.add_argument(
        "--current-labels",
        type=Path,
        help="Path to current labels (optional, CSV with 'labels' column)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/monitoring"),
        help="Directory for monitoring reports",
    )
    parser.add_argument(
        "--slack-webhook",
        type=str,
        help="Slack webhook URL for alerts",
    )
    parser.add_argument(
        "--alert-file",
        type=Path,
        help="File path to log alerts (for testing)",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        help="Path to baseline performance metrics JSON",
    )
    parser.add_argument(
        "--current-metrics",
        type=Path,
        help="Path to current performance metrics JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Run monitoring workflow."""
    args = parse_args()

    console.print("[bold blue]Phase 5: Monitoring & Drift Detection[/bold blue]\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    console.print("[cyan]Loading data...[/cyan]")
    reference_data = pd.read_parquet(args.reference_data)
    current_data = pd.read_parquet(args.current_data)

    console.print(
        f"Reference data: {len(reference_data)} samples, {len(reference_data.columns)} features"
    )
    console.print(
        f"Current data: {len(current_data)} samples, {len(current_data.columns)} features\n"
    )

    # Initialize drift detector
    drift_detector = DriftDetector(psi_threshold=0.2, ks_threshold=0.05)

    # Load optional data
    reference_predictions = None
    current_predictions = None
    reference_labels = None
    current_labels = None

    if args.reference_predictions:
        ref_pred_df = pd.read_csv(args.reference_predictions)
        reference_predictions = ref_pred_df["predictions"].values

    if args.current_predictions:
        curr_pred_df = pd.read_csv(args.current_predictions)
        current_predictions = curr_pred_df["predictions"].values

    if args.reference_labels:
        ref_label_df = pd.read_csv(args.reference_labels)
        reference_labels = ref_label_df["labels"].values

    if args.current_labels:
        curr_label_df = pd.read_csv(args.current_labels)
        current_labels = curr_label_df["labels"].values

    # Generate drift report
    console.print("[cyan]Detecting drift...[/cyan]")
    drift_report = drift_detector.generate_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        reference_labels=reference_labels,
        current_labels=current_labels,
    )

    # Save drift report
    drift_report_path = args.output_dir / "drift_report.json"
    drift_report.to_json(drift_report_path)
    console.print(f"[green][OK] Drift report saved to {drift_report_path}[/green]")

    # Print summary
    console.print("\n[bold]Drift Detection Summary:[/bold]")
    console.print(f"  Overall drift detected: {drift_report.overall_drift_detected}")
    if drift_report.drift_summary:
        console.print(
            f"  Features with drift: {drift_report.drift_summary.get('features_with_drift', 0)}/"
            f"{drift_report.drift_summary.get('total_features_checked', 0)}"
        )

    # Performance monitoring
    if args.baseline_metrics and args.current_metrics:
        console.print("\n[cyan]Comparing performance metrics...[/cyan]")
        performance_monitor = PerformanceMonitor()

        # Load metrics (simplified - in practice, these would come from model evaluation)
        import json

        with open(args.baseline_metrics) as f:
            baseline_data = json.load(f)
        with open(args.current_metrics) as f:
            current_data_metrics = json.load(f)

        from src.monitoring.performance import PerformanceMetrics

        # Extract only the fields that PerformanceMetrics expects
        def extract_metrics(data: dict) -> dict:
            return {
                "timestamp": data.get("timestamp", datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")),
                "roc_auc": data.get("roc_auc"),
                "accuracy": data.get("accuracy"),
                "precision": data.get("precision"),
                "recall": data.get("recall"),
                "f1": data.get("f1"),
                "sample_size": data.get("sample_size", 0),
            }

        baseline_metrics = PerformanceMetrics(**extract_metrics(baseline_data))
        current_metrics = PerformanceMetrics(**extract_metrics(current_data_metrics))

        performance_report = performance_monitor.compare_with_baseline(
            current_metrics, baseline_metrics
        )

        # Save performance report
        perf_report_path = args.output_dir / "performance_report.json"
        performance_report.to_json(perf_report_path)
        console.print(f"[green][OK] Performance report saved to {perf_report_path}[/green]")

        console.print("\n[bold]Performance Summary:[/bold]")
        console.print(f"  Degradation detected: {performance_report.performance_degradation}")
        console.print(f"  Severity: {performance_report.degradation_severity}")

    # Generate dashboards
    console.print("\n[cyan]Generating dashboards...[/cyan]")
    dashboard = MonitoringDashboard(args.output_dir)

    dashboard.plot_drift_metrics(drift_report)
    console.print("[green][OK] Drift metrics dashboard generated[/green]")

    if args.baseline_metrics and args.current_metrics:
        dashboard.generate_summary_dashboard(
            drift_report=drift_report,
            performance_report=performance_report if "performance_report" in locals() else None,
        )
        console.print("[green][OK] Summary dashboard generated[/green]")

    # Send alerts if configured
    if args.slack_webhook or args.alert_file:
        console.print("\n[cyan]Sending alerts...[/cyan]")
        alert_manager = AlertManager(
            slack_webhook_url=args.slack_webhook,
            alert_file=args.alert_file,
        )

        if drift_report.overall_drift_detected:
            alert_manager.alert_on_drift(drift_report, threshold_severity="low")

        if args.baseline_metrics and args.current_metrics and "performance_report" in locals():
            if performance_report.performance_degradation:
                alert_manager.alert_on_performance_degradation(
                    performance_report, threshold_severity="low"
                )

        console.print("[green][OK] Alerts processed[/green]")

    console.print("\n[bold green]Monitoring workflow completed![/bold green]")
    console.print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
