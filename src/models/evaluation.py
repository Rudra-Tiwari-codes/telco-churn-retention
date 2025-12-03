"""Model evaluation utilities with comprehensive metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    precision_at_top_k: dict[int, float] | None = None
    recall_at_top_k: dict[int, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Save metrics to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ModelEvaluator:
    """Comprehensive model evaluator with multiple metrics and visualizations."""

    def __init__(self, model: Any, feature_names: list[str] | None = None) -> None:
        """Initialize evaluator.

        Args:
            model: Trained model with predict_proba method.
            feature_names: Optional list of feature names for interpretability.
        """
        self.model = model
        self.feature_names = feature_names

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
        top_k_percentiles: list[int] | None = None,
    ) -> EvaluationMetrics:
        """Evaluate model performance.

        Args:
            X: Feature matrix.
            y: True labels.
            threshold: Classification threshold.
            top_k_percentiles: Percentiles for precision/recall at top-k.

        Returns:
            EvaluationMetrics object.
        """
        # Handle mutable default argument
        if top_k_percentiles is None:
            top_k_percentiles = [10, 20, 30]

        # Get predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Basic metrics
        roc_auc = roc_auc_score(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)

        # Precision/Recall at top-k
        precision_at_top_k = {}
        recall_at_top_k = {}

        for k_percentile in top_k_percentiles:
            k = int(len(y) * k_percentile / 100)
            top_k_indices = np.argsort(y_pred_proba)[-k:]
            top_k_true = y[top_k_indices]
            precision_at_top_k[k_percentile] = top_k_true.mean() if len(top_k_true) > 0 else 0.0
            recall_at_top_k[k_percentile] = (
                top_k_true.sum() / y.sum() if y.sum() > 0 else 0.0
            )

        return EvaluationMetrics(
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            precision_at_top_k=precision_at_top_k,
            recall_at_top_k=recall_at_top_k,
        )

    def plot_roc_curve(
        self, X: np.ndarray, y: np.ndarray, save_path: Path | None = None
    ) -> None:
        """Plot ROC curve.

        Args:
            X: Feature matrix.
            y: True labels.
            save_path: Optional path to save figure.
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_pr_curve(
        self, X: np.ndarray, y: np.ndarray, save_path: Path | None = None
    ) -> None:
        """Plot Precision-Recall curve.

        Args:
            X: Feature matrix.
            y: True labels.
            save_path: Optional path to save figure.
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_calibration_curve(
        self, X: np.ndarray, y: np.ndarray, save_path: Path | None = None
    ) -> None:
        """Plot calibration curve.

        Args:
            X: Feature matrix.
            y: True labels.
            save_path: Optional path to save figure.
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_pred_proba, n_bins=10
        )

        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(
        self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5, save_path: Path | None = None
    ) -> None:
        """Plot confusion matrix.

        Args:
            X: Feature matrix.
            y: True labels.
            threshold: Classification threshold.
            save_path: Optional path to save figure.
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        # Ensure threshold is a float
        threshold_val = float(threshold) if not isinstance(threshold, (int, float)) else threshold
        y_pred = (y_pred_proba >= threshold_val).astype(int)
        cm_matrix = confusion_matrix(y, y_pred)

        plt.figure(figsize=(8, 6))
        from matplotlib import cm

        plt.imshow(cm_matrix, interpolation="nearest", cmap=cm.get_cmap("Blues"))
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No Churn", "Churn"])
        plt.yticks(tick_marks, ["No Churn", "Churn"])

        thresh = cm_matrix.max() / 2.0
        for i, j in np.ndindex(cm_matrix.shape):
            plt.text(
                j,
                i,
                format(cm_matrix[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm_matrix[i, j] > thresh else "black",
            )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_report(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Path,
        model_name: str = "model",
    ) -> EvaluationMetrics:
        """Generate comprehensive evaluation report.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.
            output_dir: Directory to save reports and plots.
            model_name: Name of the model for file naming.

        Returns:
            Test set evaluation metrics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate on train and test
        train_metrics = self.evaluate(X_train, y_train)
        test_metrics = self.evaluate(X_test, y_test)

        # Save metrics
        train_metrics.to_json(output_dir / f"{model_name}_train_metrics.json")
        test_metrics.to_json(output_dir / f"{model_name}_test_metrics.json")

        # Generate plots
        self.plot_roc_curve(X_test, y_test, output_dir / f"{model_name}_roc_curve.png")
        self.plot_pr_curve(X_test, y_test, output_dir / f"{model_name}_pr_curve.png")
        self.plot_calibration_curve(
            X_test, y_test, output_dir / f"{model_name}_calibration_curve.png"
        )
        self.plot_confusion_matrix(
            X_test, y_test, threshold=0.5, save_path=output_dir / f"{model_name}_confusion_matrix.png"
        )

        # Generate classification report
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)

        with open(output_dir / f"{model_name}_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        return test_metrics

