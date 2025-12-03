"""
Drift detection module implementing PSI and KS tests for data, prediction, and label drift.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DriftMetrics:
    """Container for drift detection metrics."""

    feature_name: str
    psi: float | None = None
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    drift_detected: bool = False
    drift_severity: str = "none"  # "none", "low", "medium", "high"
    feature_type: str = "unknown"  # "numeric", "categorical"
    # Feature metadata from feature store
    description: str | None = None
    owner: str | None = None
    tags: list[str] | None = None
    drift_importance_score: float | None = None  # Combined PSI/KS score for ranking

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DriftReport:
    """Container for complete drift report."""

    timestamp: str
    data_drift: list[DriftMetrics]
    prediction_drift: DriftMetrics | None = None
    label_drift: DriftMetrics | None = None
    overall_drift_detected: bool = False
    drift_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "data_drift": [m.to_dict() for m in self.data_drift],
            "prediction_drift": self.prediction_drift.to_dict() if self.prediction_drift else None,
            "label_drift": self.label_drift.to_dict() if self.label_drift else None,
            "overall_drift_detected": self.overall_drift_detected,
            "drift_summary": self.drift_summary,
        }

    def to_json(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class DriftDetector:
    """Detector for data, prediction, and label drift using PSI and KS tests."""

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        psi_bins: int = 10,
        feature_store_metadata_dir: Path | None = None,
    ) -> None:
        """Initialize drift detector.

        Args:
            psi_threshold: PSI threshold for drift detection.
                           < 0.1: no drift, 0.1-0.25: low, > 0.25: high
            ks_threshold: KS test p-value threshold for statistical significance.
            psi_bins: Number of bins for PSI calculation.
            feature_store_metadata_dir: Optional path to feature store metadata directory.
                                        If provided, enables feature metadata enrichment.
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.psi_bins = psi_bins
        self.feature_store_metadata_dir = feature_store_metadata_dir

    def calculate_psi(
        self,
        expected: pd.Series | np.ndarray,
        actual: pd.Series | np.ndarray,
        feature_type: str = "numeric",
    ) -> float:
        """Calculate Population Stability Index (PSI).

        Args:
            expected: Expected/reference distribution.
            actual: Actual/current distribution.
            feature_type: Type of feature ("numeric" or "categorical").

        Returns:
            PSI value.
        """
        if feature_type == "categorical":
            return self._calculate_psi_categorical(expected, actual)
        else:
            return self._calculate_psi_numeric(expected, actual)

    def _calculate_psi_numeric(
        self, expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray
    ) -> float:
        """Calculate PSI for numeric features.

        Uses standard PSI formula with proper normalization:
        PSI = sum((actual_prop - expected_prop) * log(actual_prop / expected_prop))
        where proportions are properly normalized (sum to 1.0).
        """
        expected = pd.Series(expected).dropna()
        actual = pd.Series(actual).dropna()

        if len(expected) == 0 or len(actual) == 0:
            return np.nan

        # Create bins based on expected distribution
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())

        if min_val == max_val:
            # Constant feature, no drift
            return 0.0

        bins = np.linspace(min_val, max_val, self.psi_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calculate distributions
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        # Normalize to proper proportions (sum to 1.0)
        expected_total = len(expected)
        actual_total = len(actual)

        if expected_total == 0 or actual_total == 0:
            return np.nan

        expected_props = expected_counts.astype(float) / expected_total
        actual_props = actual_counts.astype(float) / actual_total

        # Add small epsilon only to avoid log(0) in the calculation
        # This ensures distributions remain properly normalized
        epsilon = 1e-6
        expected_props = np.where(expected_props == 0, epsilon, expected_props)
        actual_props = np.where(actual_props == 0, epsilon, actual_props)

        # Calculate PSI
        psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))

        return float(psi)

    def _calculate_psi_categorical(
        self, expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray
    ) -> float:
        """Calculate PSI for categorical features.

        Uses standard PSI formula with proper normalization:
        PSI = sum((actual_prop - expected_prop) * log(actual_prop / expected_prop))
        where proportions are properly normalized (sum to 1.0).
        """
        expected = pd.Series(expected).dropna()
        actual = pd.Series(actual).dropna()

        if len(expected) == 0 or len(actual) == 0:
            return np.nan

        # Get all unique categories
        all_categories = set(expected.unique()) | set(actual.unique())

        # Calculate proportions
        expected_counts = expected.value_counts()
        actual_counts = actual.value_counts()

        # Normalize to proper proportions (sum to 1.0)
        expected_total = len(expected)
        actual_total = len(actual)

        if expected_total == 0 or actual_total == 0:
            return np.nan

        # Add small epsilon only to avoid log(0) in the calculation
        # This ensures distributions remain properly normalized
        epsilon = 1e-6

        psi = 0.0
        for category in all_categories:
            # Calculate properly normalized proportions
            expected_prop = expected_counts.get(category, 0) / expected_total
            actual_prop = actual_counts.get(category, 0) / actual_total

            # Add epsilon only if needed to avoid log(0)
            if expected_prop == 0:
                expected_prop = epsilon
            if actual_prop == 0:
                actual_prop = epsilon

            psi += (actual_prop - expected_prop) * np.log(actual_prop / expected_prop)

        return float(psi)

    def calculate_ks_test(
        self, expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray
    ) -> tuple[float, float]:
        """Calculate Kolmogorov-Smirnov test statistic and p-value.

        Args:
            expected: Expected/reference distribution.
            actual: Actual/current distribution.

        Returns:
            Tuple of (KS statistic, p-value).
        """
        expected = pd.Series(expected).dropna()
        actual = pd.Series(actual).dropna()

        if len(expected) == 0 or len(actual) == 0:
            return (np.nan, np.nan)

        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(expected, actual)

        return (float(ks_statistic), float(p_value))

    def detect_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_name: str,
        feature_type: str | None = None,
    ) -> DriftMetrics:
        """Detect drift for a single feature.

        Args:
            reference_data: Reference/training data.
            current_data: Current/production data.
            feature_name: Name of the feature to check.
            feature_type: Type of feature ("numeric" or "categorical"). Auto-detected if None.

        Returns:
            DriftMetrics object.
        """
        if feature_name not in reference_data.columns or feature_name not in current_data.columns:
            return DriftMetrics(
                feature_name=feature_name,
                drift_detected=False,
                drift_severity="none",
                feature_type="unknown",
            )

        ref_series = reference_data[feature_name]
        curr_series = current_data[feature_name]

        # Auto-detect feature type if not provided
        if feature_type is None:
            if pd.api.types.is_numeric_dtype(ref_series):
                feature_type = "numeric"
            else:
                feature_type = "categorical"

        # Calculate PSI
        psi = self.calculate_psi(ref_series, curr_series, feature_type=feature_type)

        # Calculate KS test (only for numeric features)
        ks_statistic = None
        ks_pvalue = None
        if feature_type == "numeric":
            ks_statistic, ks_pvalue = self.calculate_ks_test(ref_series, curr_series)

        # Determine drift severity
        drift_detected = False
        drift_severity = "none"

        if not np.isnan(psi):
            if psi < 0.1:
                drift_severity = "none"
            elif psi < 0.25:
                drift_severity = "low"
                drift_detected = True
            else:
                drift_severity = "high"
                drift_detected = True

        # Also check KS test for numeric features
        if feature_type == "numeric" and ks_pvalue is not None:
            if ks_pvalue < self.ks_threshold:
                drift_detected = True
                if drift_severity == "none":
                    drift_severity = "low"

        # Calculate drift importance score (for ranking)
        drift_importance_score = self._calculate_drift_importance_score(
            psi, ks_statistic, ks_pvalue, feature_type
        )

        return DriftMetrics(
            feature_name=feature_name,
            psi=psi,
            ks_statistic=ks_statistic,
            ks_pvalue=ks_pvalue,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            feature_type=feature_type,
            drift_importance_score=drift_importance_score,
        )

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
    ) -> list[DriftMetrics]:
        """Detect data drift across multiple features.

        Args:
            reference_data: Reference/training data.
            current_data: Current/production data.
            features: List of features to check. If None, checks all common features.

        Returns:
            List of DriftMetrics objects.
        """
        if features is None:
            # Check all common features
            features = list(set(reference_data.columns) & set(current_data.columns))

        drift_metrics = []
        for feature in features:
            metrics = self.detect_feature_drift(reference_data, current_data, feature)
            drift_metrics.append(metrics)

        return drift_metrics

    def _calculate_drift_importance_score(
        self,
        psi: float | None,
        ks_statistic: float | None,
        ks_pvalue: float | None,
        feature_type: str,
    ) -> float | None:
        """Calculate a combined drift importance score for ranking features.

        Combines PSI and KS statistics into a single score that can be used
        to rank features by drift severity. Higher scores indicate more significant drift.

        Args:
            psi: PSI value.
            ks_statistic: KS statistic value.
            ks_pvalue: KS p-value.
            feature_type: Type of feature.

        Returns:
            Combined importance score, or None if insufficient data.
        """
        if psi is None or np.isnan(psi):
            return None

        # Base score from PSI (normalized to 0-1 scale, higher is worse)
        # PSI typically ranges 0-1 for low drift, 1+ for high drift
        psi_score = min(psi / 1.0, 1.0)  # Cap at 1.0 for normalization

        # Add KS component for numeric features
        ks_score = 0.0
        if feature_type == "numeric" and ks_statistic is not None and not np.isnan(ks_statistic):
            # KS statistic ranges 0-1, higher is worse
            # Also consider p-value significance
            ks_component = ks_statistic
            if ks_pvalue is not None and not np.isnan(ks_pvalue):
                # Weight by significance: lower p-value = more significant
                significance_weight = 1.0 - min(ks_pvalue / self.ks_threshold, 1.0)
                ks_component *= 1.0 + significance_weight
            ks_score = min(ks_component, 1.0)

        # Combined score: weighted average (PSI is primary, KS adds for numeric)
        if feature_type == "numeric" and ks_score > 0:
            importance_score = 0.7 * psi_score + 0.3 * ks_score
        else:
            importance_score = psi_score

        return float(importance_score)

    def enrich_with_feature_metadata(
        self,
        drift_metrics: list[DriftMetrics],
        feature_set_name: str = "telco_churn_features",
        feature_set_version: str | None = None,
    ) -> list[DriftMetrics]:
        """Enrich drift metrics with feature store metadata.

        Args:
            drift_metrics: List of DriftMetrics to enrich.
            feature_set_name: Name of the feature set in the feature store.
            feature_set_version: Optional version of the feature set. If None, uses latest.

        Returns:
            List of enriched DriftMetrics objects.
        """
        if self.feature_store_metadata_dir is None:
            return drift_metrics

        try:
            from src.features.store import FeatureStore

            feature_store = FeatureStore(self.feature_store_metadata_dir)
            metadata = feature_store.get_feature_set_metadata(feature_set_name, feature_set_version)

            if metadata is None:
                return drift_metrics

            # Create lookup dictionary for feature metadata
            feature_metadata_map = {feat.name: feat for feat in metadata.features}

            # Enrich each drift metric
            enriched_metrics = []
            for metric in drift_metrics:
                if metric.feature_name in feature_metadata_map:
                    feat_meta = feature_metadata_map[metric.feature_name]
                    metric.description = feat_meta.description
                    metric.owner = feat_meta.owner
                    metric.tags = feat_meta.tags
                enriched_metrics.append(metric)

            return enriched_metrics
        except Exception:
            # If feature store is unavailable, return original metrics
            return drift_metrics

    def rank_features_by_drift_importance(
        self,
        drift_metrics: list[DriftMetrics],
        top_n: int | None = None,
    ) -> list[DriftMetrics]:
        """Rank features by drift importance score.

        Features are sorted by their drift_importance_score (highest first),
        which combines PSI and KS statistics. This provides a business-relevant
        ranking beyond just sorting by PSI or KS alone.

        Args:
            drift_metrics: List of DriftMetrics to rank.
            top_n: Optional number of top features to return. If None, returns all.

        Returns:
            List of DriftMetrics sorted by drift importance (highest first).
        """
        # Filter out metrics without importance scores
        scorable_metrics = [
            m
            for m in drift_metrics
            if m.drift_importance_score is not None and not np.isnan(m.drift_importance_score)
        ]

        # Sort by importance score (descending)
        ranked = sorted(
            scorable_metrics, key=lambda x: x.drift_importance_score or 0.0, reverse=True
        )

        # Add metrics without scores at the end
        unscorable_metrics = [
            m
            for m in drift_metrics
            if m.drift_importance_score is None or np.isnan(m.drift_importance_score)
        ]
        ranked.extend(unscorable_metrics)

        if top_n is not None:
            ranked = ranked[:top_n]

        return ranked

    def detect_prediction_drift(
        self,
        reference_predictions: pd.Series | np.ndarray,
        current_predictions: pd.Series | np.ndarray,
    ) -> DriftMetrics:
        """Detect drift in model predictions.

        Args:
            reference_predictions: Reference/training predictions.
            current_predictions: Current/production predictions.

        Returns:
            DriftMetrics object.
        """
        ref_series = pd.Series(reference_predictions).dropna()
        curr_series = pd.Series(current_predictions).dropna()

        # Calculate PSI (predictions are numeric)
        psi = self.calculate_psi(ref_series, curr_series, feature_type="numeric")

        # Calculate KS test
        ks_statistic, ks_pvalue = self.calculate_ks_test(ref_series, curr_series)

        # Determine drift
        drift_detected = False
        drift_severity = "none"

        if not np.isnan(psi):
            if psi < 0.1:
                drift_severity = "none"
            elif psi < 0.25:
                drift_severity = "low"
                drift_detected = True
            else:
                drift_severity = "high"
                drift_detected = True

        if ks_pvalue is not None and ks_pvalue < self.ks_threshold:
            drift_detected = True
            if drift_severity == "none":
                drift_severity = "low"

        # Calculate drift importance score
        drift_importance_score = self._calculate_drift_importance_score(
            psi, ks_statistic, ks_pvalue, "numeric"
        )

        return DriftMetrics(
            feature_name="predictions",
            psi=psi,
            ks_statistic=ks_statistic,
            ks_pvalue=ks_pvalue,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            feature_type="numeric",
            drift_importance_score=drift_importance_score,
        )

    def detect_label_drift(
        self,
        reference_labels: pd.Series | np.ndarray,
        current_labels: pd.Series | np.ndarray,
    ) -> DriftMetrics:
        """Detect drift in target labels.

        Args:
            reference_labels: Reference/training labels.
            current_labels: Current/production labels.

        Returns:
            DriftMetrics object.
        """
        ref_series = pd.Series(reference_labels).dropna()
        curr_series = pd.Series(current_labels).dropna()

        # For binary classification, use PSI on categorical distribution
        psi = self.calculate_psi(ref_series, curr_series, feature_type="categorical")

        # Determine drift
        drift_detected = False
        drift_severity = "none"

        if not np.isnan(psi):
            if psi < 0.1:
                drift_severity = "none"
            elif psi < 0.25:
                drift_severity = "low"
                drift_detected = True
            else:
                drift_severity = "high"
                drift_detected = True

        # Calculate drift importance score
        drift_importance_score = self._calculate_drift_importance_score(
            psi, None, None, "categorical"
        )

        return DriftMetrics(
            feature_name="labels",
            psi=psi,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            feature_type="categorical",
            drift_importance_score=drift_importance_score,
        )

    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        reference_predictions: pd.Series | np.ndarray | None = None,
        current_predictions: pd.Series | np.ndarray | None = None,
        reference_labels: pd.Series | np.ndarray | None = None,
        current_labels: pd.Series | np.ndarray | None = None,
        timestamp: str | None = None,
    ) -> DriftReport:
        """Generate comprehensive drift report.

        Args:
            reference_data: Reference/training data.
            current_data: Current/production data.
            reference_predictions: Reference predictions (optional).
            current_predictions: Current predictions (optional).
            reference_labels: Reference labels (optional).
            current_labels: Current labels (optional).
            timestamp: Timestamp for the report. Auto-generated if None.

        Returns:
            DriftReport object.
        """
        from datetime import UTC, datetime

        if timestamp is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        # Detect data drift
        data_drift = self.detect_data_drift(reference_data, current_data)

        # Detect prediction drift
        prediction_drift = None
        if reference_predictions is not None and current_predictions is not None:
            prediction_drift = self.detect_prediction_drift(
                reference_predictions, current_predictions
            )

        # Detect label drift
        label_drift = None
        if reference_labels is not None and current_labels is not None:
            label_drift = self.detect_label_drift(reference_labels, current_labels)

        # Determine overall drift
        overall_drift = False
        drift_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}

        for metric in data_drift:
            if metric.drift_detected:
                overall_drift = True
            drift_counts[metric.drift_severity] = drift_counts.get(metric.drift_severity, 0) + 1

        if prediction_drift and prediction_drift.drift_detected:
            overall_drift = True
            drift_counts[prediction_drift.drift_severity] = (
                drift_counts.get(prediction_drift.drift_severity, 0) + 1
            )

        if label_drift and label_drift.drift_detected:
            overall_drift = True
            drift_counts[label_drift.drift_severity] = (
                drift_counts.get(label_drift.drift_severity, 0) + 1
            )

        # Enrich with feature metadata if available
        if self.feature_store_metadata_dir is not None:
            data_drift = self.enrich_with_feature_metadata(data_drift)

        # Rank features by drift importance
        ranked_features = self.rank_features_by_drift_importance(data_drift)

        drift_summary = {
            "total_features_checked": len(data_drift),
            "features_with_drift": sum(1 for m in data_drift if m.drift_detected),
            "drift_severity_counts": drift_counts,
            "prediction_drift_detected": (
                prediction_drift.drift_detected if prediction_drift else False
            ),
            "label_drift_detected": label_drift.drift_detected if label_drift else False,
            "top_drifted_features": [
                {
                    "feature_name": m.feature_name,
                    "drift_importance_score": m.drift_importance_score,
                    "psi": m.psi,
                    "drift_severity": m.drift_severity,
                    "description": m.description,
                    "owner": m.owner,
                }
                for m in ranked_features[:10]
                if m.drift_detected
            ],
        }

        return DriftReport(
            timestamp=timestamp,
            data_drift=data_drift,
            prediction_drift=prediction_drift,
            label_drift=label_drift,
            overall_drift_detected=overall_drift,
            drift_summary=drift_summary,
        )
