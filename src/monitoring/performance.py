"""
Performance monitoring module for tracking model metrics over time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics at a point in time."""

    timestamp: str
    roc_auc: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    sample_size: int = 0
    prediction_threshold: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceReport:
    """Container for performance monitoring report."""

    current_metrics: PerformanceMetrics
    baseline_metrics: PerformanceMetrics | None = None
    performance_degradation: bool = False
    degradation_severity: str = "none"  # "none", "low", "medium", "high"
    metric_changes: dict[str, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "current_metrics": self.current_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "performance_degradation": self.performance_degradation,
            "degradation_severity": self.degradation_severity,
            "metric_changes": self.metric_changes,
        }

    def to_json(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class PerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(
        self,
        roc_auc_threshold: float = 0.05,
        accuracy_threshold: float = 0.05,
        f1_threshold: float = 0.05,
    ) -> None:
        """Initialize performance monitor.

        Args:
            roc_auc_threshold: Threshold for ROC-AUC degradation (absolute difference).
            accuracy_threshold: Threshold for accuracy degradation (absolute difference).
            f1_threshold: Threshold for F1 score degradation (absolute difference).
        """
        self.roc_auc_threshold = roc_auc_threshold
        self.accuracy_threshold = accuracy_threshold
        self.f1_threshold = f1_threshold

    def calculate_metrics(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred_proba: np.ndarray | pd.Series,
        threshold: float = 0.5,
        timestamp: str | None = None,
    ) -> PerformanceMetrics:
        """Calculate performance metrics.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            threshold: Classification threshold.
            timestamp: Timestamp for metrics. Auto-generated if None.

        Returns:
            PerformanceMetrics object.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        y_true = pd.Series(y_true).dropna()
        y_pred_proba = pd.Series(y_pred_proba).dropna()

        if len(y_true) == 0 or len(y_pred_proba) == 0:
            return PerformanceMetrics(
                timestamp=timestamp,
                sample_size=0,
                prediction_threshold=threshold,
            )

        # Align indices
        common_idx = y_true.index.intersection(y_pred_proba.index)
        y_true = y_true.loc[common_idx]
        y_pred_proba = y_pred_proba.loc[common_idx]

        if len(y_true) == 0:
            return PerformanceMetrics(
                timestamp=timestamp,
                sample_size=0,
                prediction_threshold=threshold,
            )

        # Binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            roc_auc = None

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return PerformanceMetrics(
            timestamp=timestamp,
            roc_auc=roc_auc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            sample_size=len(y_true),
            prediction_threshold=threshold,
        )

    def compare_with_baseline(
        self,
        current_metrics: PerformanceMetrics,
        baseline_metrics: PerformanceMetrics,
    ) -> PerformanceReport:
        """Compare current metrics with baseline.

        Args:
            current_metrics: Current performance metrics.
            baseline_metrics: Baseline/reference performance metrics.

        Returns:
            PerformanceReport object.
        """
        metric_changes = {}
        degradation_detected = False
        max_degradation = 0.0

        # Compare each metric
        if current_metrics.roc_auc is not None and baseline_metrics.roc_auc is not None:
            change = baseline_metrics.roc_auc - current_metrics.roc_auc
            metric_changes["roc_auc"] = change
            if change > self.roc_auc_threshold:
                degradation_detected = True
                max_degradation = max(max_degradation, change)

        if current_metrics.accuracy is not None and baseline_metrics.accuracy is not None:
            change = baseline_metrics.accuracy - current_metrics.accuracy
            metric_changes["accuracy"] = change
            if change > self.accuracy_threshold:
                degradation_detected = True
                max_degradation = max(max_degradation, change)

        if current_metrics.f1 is not None and baseline_metrics.f1 is not None:
            change = baseline_metrics.f1 - current_metrics.f1
            metric_changes["f1"] = change
            if change > self.f1_threshold:
                degradation_detected = True
                max_degradation = max(max_degradation, change)

        # Determine severity
        severity = "none"
        if degradation_detected:
            if max_degradation < 0.05:
                severity = "low"
            elif max_degradation < 0.15:
                severity = "medium"
            else:
                severity = "high"

        return PerformanceReport(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            performance_degradation=degradation_detected,
            degradation_severity=severity,
            metric_changes=metric_changes,
        )

    def load_metrics_history(self, metrics_file: Path) -> list[PerformanceMetrics]:
        """Load historical performance metrics.

        Args:
            metrics_file: Path to JSON file containing metrics history.

        Returns:
            List of PerformanceMetrics objects.
        """
        if not metrics_file.exists():
            return []

        with open(metrics_file) as f:
            data = json.load(f)

        metrics_list = []
        if isinstance(data, list):
            for item in data:
                metrics_list.append(PerformanceMetrics(**item))
        elif isinstance(data, dict):
            # Single metrics object
            metrics_list.append(PerformanceMetrics(**data))

        return metrics_list

    def save_metrics(self, metrics: PerformanceMetrics, metrics_file: Path) -> None:
        """Save metrics to history file.

        Args:
            metrics: PerformanceMetrics to save.
            metrics_file: Path to JSON file for metrics history.
        """
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing history
        history = self.load_metrics_history(metrics_file)

        # Append new metrics
        history.append(metrics)

        # Keep only last 100 entries
        history = history[-100:]

        # Save
        with open(metrics_file, "w") as f:
            json.dump([m.to_dict() for m in history], f, indent=2)
