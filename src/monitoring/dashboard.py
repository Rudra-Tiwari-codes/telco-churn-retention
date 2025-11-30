"""
Dashboard generator for monitoring metrics visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Try to import seaborn, but make it optional
try:
    import seaborn as sns

    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

plt.rcParams["figure.figsize"] = (12, 6)


class MonitoringDashboard:
    """Generator for monitoring dashboards."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize dashboard generator.

        Args:
            output_dir: Directory to save dashboard images.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_drift_metrics(
        self,
        drift_report: Any,  # DriftReport from drift.py
        save_path: Path | None = None,
    ) -> None:
        """Plot drift metrics visualization.

        Args:
            drift_report: DriftReport object.
            save_path: Optional path to save figure. Defaults to output_dir/drift_metrics.png.
        """
        if save_path is None:
            save_path = self.output_dir / "drift_metrics.png"

        # Prepare data
        drift_data = []
        for metric in drift_report.data_drift:
            if metric.psi is not None:
                drift_data.append(
                    {
                        "feature": metric.feature_name,
                        "psi": metric.psi,
                        "severity": metric.drift_severity,
                    }
                )

        if not drift_data:
            return

        df = pd.DataFrame(drift_data)
        df = df.sort_values("psi", ascending=False).head(20)  # Top 20 features

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color map for severity
        color_map = {"none": "green", "low": "yellow", "medium": "orange", "high": "red"}
        colors = [color_map.get(sev, "gray") for sev in df["severity"]]

        # Plot
        ax.barh(df["feature"], df["psi"], color=colors)
        ax.axvline(x=0.1, color="yellow", linestyle="--", label="Low Drift Threshold")
        ax.axvline(x=0.25, color="red", linestyle="--", label="High Drift Threshold")
        ax.set_xlabel("PSI (Population Stability Index)")
        ax.set_ylabel("Feature")
        ax.set_title("Data Drift Metrics by Feature")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_performance_trends(
        self,
        metrics_history: list[Any],  # List of PerformanceMetrics
        save_path: Path | None = None,
    ) -> None:
        """Plot performance metrics over time.

        Args:
            metrics_history: List of PerformanceMetrics objects.
            save_path: Optional path to save figure.
        """
        if save_path is None:
            save_path = self.output_dir / "performance_trends.png"

        if not metrics_history:
            return

        # Prepare data
        data = []
        for metrics in metrics_history:
            data.append(
                {
                    "timestamp": metrics.timestamp,
                    "roc_auc": metrics.roc_auc,
                    "accuracy": metrics.accuracy,
                    "f1": metrics.f1,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                }
            )

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%SZ", errors="coerce")
        df = df.sort_values("timestamp")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # ROC-AUC
        if df["roc_auc"].notna().any():
            axes[0, 0].plot(df["timestamp"], df["roc_auc"], marker="o", label="ROC-AUC")
            axes[0, 0].set_title("ROC-AUC Over Time")
            axes[0, 0].set_ylabel("ROC-AUC")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

        # Accuracy
        if df["accuracy"].notna().any():
            axes[0, 1].plot(
                df["timestamp"], df["accuracy"], marker="o", label="Accuracy", color="green"
            )
            axes[0, 1].set_title("Accuracy Over Time")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # F1 Score
        if df["f1"].notna().any():
            axes[1, 0].plot(df["timestamp"], df["f1"], marker="o", label="F1 Score", color="orange")
            axes[1, 0].set_title("F1 Score Over Time")
            axes[1, 0].set_ylabel("F1 Score")
            axes[1, 0].set_xlabel("Timestamp")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Precision/Recall
        if df["precision"].notna().any() or df["recall"].notna().any():
            if df["precision"].notna().any():
                axes[1, 1].plot(
                    df["timestamp"], df["precision"], marker="o", label="Precision", color="blue"
                )
            if df["recall"].notna().any():
                axes[1, 1].plot(
                    df["timestamp"], df["recall"], marker="s", label="Recall", color="red"
                )
            axes[1, 1].set_title("Precision & Recall Over Time")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].set_xlabel("Timestamp")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_prediction_distribution(
        self,
        reference_predictions: list[float] | pd.Series,
        current_predictions: list[float] | pd.Series,
        save_path: Path | None = None,
    ) -> None:
        """Plot distribution comparison of predictions.

        Args:
            reference_predictions: Reference/training predictions.
            current_predictions: Current/production predictions.
            save_path: Optional path to save figure.
        """
        if save_path is None:
            save_path = self.output_dir / "prediction_distribution.png"

        fig, ax = plt.subplots(figsize=(10, 6))

        ref_series = pd.Series(reference_predictions)
        curr_series = pd.Series(current_predictions)

        ax.hist(ref_series, bins=30, alpha=0.5, label="Reference", color="blue", density=True)
        ax.hist(curr_series, bins=30, alpha=0.5, label="Current", color="red", density=True)
        ax.set_xlabel("Prediction Probability")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Distribution Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_summary_dashboard(
        self,
        drift_report: Any | None = None,
        performance_report: Any | None = None,
        metrics_history: list[Any] | None = None,
        save_path: Path | None = None,
    ) -> None:
        """Generate comprehensive summary dashboard.

        Args:
            drift_report: Optional DriftReport object.
            performance_report: Optional PerformanceReport object.
            metrics_history: Optional list of PerformanceMetrics.
            save_path: Optional path to save figure.
        """
        if save_path is None:
            save_path = self.output_dir / "monitoring_dashboard.png"

        # Determine number of subplots needed
        n_plots = 0
        if drift_report:
            n_plots += 1
        if performance_report:
            n_plots += 1
        if metrics_history:
            n_plots += 1

        if n_plots == 0:
            return

        fig = plt.figure(figsize=(16, 5 * n_plots))
        plot_idx = 1

        # Drift metrics
        if drift_report:
            ax = fig.add_subplot(n_plots, 1, plot_idx)
            drift_data = []
            for metric in drift_report.data_drift:
                if metric.psi is not None:
                    drift_data.append(
                        {
                            "feature": metric.feature_name,
                            "psi": metric.psi,
                            "severity": metric.drift_severity,
                        }
                    )

            if drift_data:
                df = pd.DataFrame(drift_data)
                df = df.sort_values("psi", ascending=False).head(15)

                color_map = {"none": "green", "low": "yellow", "medium": "orange", "high": "red"}
                colors = [color_map.get(sev, "gray") for sev in df["severity"]]

                ax.barh(df["feature"], df["psi"], color=colors)
                ax.axvline(x=0.1, color="yellow", linestyle="--", alpha=0.5)
                ax.axvline(x=0.25, color="red", linestyle="--", alpha=0.5)
                ax.set_xlabel("PSI")
                ax.set_title("Data Drift Metrics (Top 15 Features)")
                ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Performance comparison
        if performance_report:
            ax = fig.add_subplot(n_plots, 1, plot_idx)

            metrics = ["roc_auc", "accuracy", "f1", "precision", "recall"]
            current_vals = []
            baseline_vals = []
            metric_labels = []

            for metric in metrics:
                curr_val = getattr(performance_report.current_metrics, metric, None)
                base_val = (
                    getattr(performance_report.baseline_metrics, metric, None)
                    if performance_report.baseline_metrics
                    else None
                )

                if curr_val is not None and base_val is not None:
                    current_vals.append(curr_val)
                    baseline_vals.append(base_val)
                    metric_labels.append(metric.replace("_", " ").title())

            if current_vals and baseline_vals:
                x = range(len(metric_labels))
                width = 0.35
                ax.bar(
                    [i - width / 2 for i in x], baseline_vals, width, label="Baseline", color="blue"
                )
                ax.bar(
                    [i + width / 2 for i in x], current_vals, width, label="Current", color="red"
                )
                ax.set_ylabel("Score")
                ax.set_title("Performance Comparison: Baseline vs Current")
                ax.set_xticks(x)
                ax.set_xticklabels(metric_labels, rotation=45, ha="right")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Performance trends
        if metrics_history:
            ax = fig.add_subplot(n_plots, 1, plot_idx)

            data = []
            for metrics in metrics_history:
                data.append(
                    {
                        "timestamp": metrics.timestamp,
                        "roc_auc": metrics.roc_auc,
                        "accuracy": metrics.accuracy,
                        "f1": metrics.f1,
                    }
                )

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], format="%Y%m%dT%H%M%SZ", errors="coerce"
            )
            df = df.sort_values("timestamp")

            if df["roc_auc"].notna().any():
                ax.plot(df["timestamp"], df["roc_auc"], marker="o", label="ROC-AUC")
            if df["accuracy"].notna().any():
                ax.plot(df["timestamp"], df["accuracy"], marker="s", label="Accuracy")
            if df["f1"].notna().any():
                ax.plot(df["timestamp"], df["f1"], marker="^", label="F1")

            ax.set_ylabel("Score")
            ax.set_title("Performance Trends Over Time")
            ax.set_xlabel("Timestamp")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
