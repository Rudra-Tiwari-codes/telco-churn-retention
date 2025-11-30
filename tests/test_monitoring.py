"""Tests for monitoring modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import DriftDetector, DriftMetrics, DriftReport
from src.monitoring.performance import PerformanceMetrics, PerformanceMonitor


def test_drift_detector_psi_numeric() -> None:
    """Test PSI calculation for numeric features."""
    detector = DriftDetector()

    # Create reference and current distributions
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(0, 1, 1000)  # Same distribution

    psi = detector.calculate_psi(reference, current, feature_type="numeric")
    assert psi is not None
    assert psi >= 0.0
    assert psi < 0.1  # Should be low for similar distributions


def test_drift_detector_psi_categorical() -> None:
    """Test PSI calculation for categorical features."""
    detector = DriftDetector()

    reference = pd.Series(["A", "B", "C"] * 100)
    current = pd.Series(["A", "B", "C"] * 100)  # Same distribution

    psi = detector.calculate_psi(reference, current, feature_type="categorical")
    assert psi is not None
    assert psi >= 0.0
    assert psi < 0.1  # Should be low for same distribution


def test_drift_detector_ks_test() -> None:
    """Test KS test calculation."""
    detector = DriftDetector()

    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(0, 1, 1000)

    ks_stat, p_value = detector.calculate_ks_test(reference, current)
    assert ks_stat is not None
    assert p_value is not None
    assert 0.0 <= ks_stat <= 1.0
    assert 0.0 <= p_value <= 1.0


def test_detect_feature_drift() -> None:
    """Test feature drift detection."""
    detector = DriftDetector()

    reference_data = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
    current_data = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})

    metrics = detector.detect_feature_drift(reference_data, current_data, "feature1")

    assert isinstance(metrics, DriftMetrics)
    assert metrics.feature_name == "feature1"
    assert metrics.psi is not None


def test_generate_drift_report() -> None:
    """Test drift report generation."""
    detector = DriftDetector()

    reference_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.choice(["A", "B", "C"], 1000),
        }
    )
    current_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.choice(["A", "B", "C"], 1000),
        }
    )

    report = detector.generate_drift_report(reference_data, current_data)

    assert isinstance(report, DriftReport)
    assert len(report.data_drift) > 0
    assert report.timestamp is not None


def test_performance_monitor_calculate_metrics() -> None:
    """Test performance metrics calculation."""
    monitor = PerformanceMonitor()

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.9, 0.75])

    metrics = monitor.calculate_metrics(y_true, y_pred_proba)

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.roc_auc is not None
    assert metrics.accuracy is not None
    assert metrics.f1 is not None
    assert metrics.sample_size == len(y_true)


def test_performance_monitor_compare_baseline() -> None:
    """Test performance comparison with baseline."""
    monitor = PerformanceMonitor()

    current_metrics = PerformanceMetrics(
        timestamp="20250101T000000Z",
        roc_auc=0.85,
        accuracy=0.80,
        f1=0.75,
        sample_size=1000,
    )

    baseline_metrics = PerformanceMetrics(
        timestamp="20250101T000000Z",
        roc_auc=0.90,
        accuracy=0.85,
        f1=0.80,
        sample_size=1000,
    )

    report = monitor.compare_with_baseline(current_metrics, baseline_metrics)

    assert report.performance_degradation is True
    assert report.degradation_severity in ["none", "low", "medium", "high"]
    assert report.metric_changes is not None
