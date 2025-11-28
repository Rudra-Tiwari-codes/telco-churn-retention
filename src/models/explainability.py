"""SHAP-based model explainability utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap


class ModelExplainer:
    """SHAP-based model explainer for churn prediction."""

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize explainer.

        Args:
            model: Trained model with predict_proba method.
            X_background: Background dataset for SHAP (typically training set sample).
            feature_names: Optional list of feature names.
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names

        # Initialize SHAP explainer based on model type
        model_type = type(model).__name__.lower()

        if "xgboost" in model_type or "lgbm" in model_type or "lightgbm" in model_type:
            # Tree-based models - use TreeExplainer
            self.explainer = shap.TreeExplainer(model)
        else:
            # Other models - use KernelExplainer (slower but more general)
            # Use a sample of background data for efficiency
            if len(X_background) > 100:
                background_sample = X_background[np.random.choice(len(X_background), 100, replace=False)]
            else:
                background_sample = X_background
            self.explainer = shap.KernelExplainer(model.predict_proba, background_sample)

    def explain_instance(
        self, X_instance: np.ndarray, max_evals: int = 100
    ) -> tuple[np.ndarray, dict]:
        """Explain a single prediction instance.

        Args:
            X_instance: Single instance to explain (shape: (n_features,)).
            max_evals: Maximum evaluations for KernelExplainer.

        Returns:
            Tuple of (SHAP values, explanation dict).
        """
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_values = self.explainer.shap_values(X_instance.reshape(1, -1))
            # TreeExplainer returns list for binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = self.explainer.shap_values(
                X_instance.reshape(1, -1), nsamples=max_evals
            )
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        # Create explanation dict
        explanation = {
            "feature_names": self.feature_names or [f"feature_{i}" for i in range(len(X_instance))],
            "shap_values": shap_values.flatten().tolist(),
            "base_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value),
            "prediction": float(self.model.predict_proba(X_instance.reshape(1, -1))[0, 1]),
        }

        return shap_values, explanation

    def explain_batch(
        self, X: np.ndarray, max_evals: int | None = None
    ) -> tuple[np.ndarray, dict]:
        """Explain a batch of predictions.

        Args:
            X: Feature matrix to explain.
            max_evals: Maximum evaluations for KernelExplainer (only used if needed).

        Returns:
            Tuple of (SHAP values array, summary dict).
        """
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
        else:
            # For KernelExplainer, use a reasonable number of samples
            if max_evals is None:
                max_evals = min(100, len(X) * 2)
            shap_values = self.explainer.shap_values(X, nsamples=max_evals)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.abs(shap_values).mean(axis=0)

        summary = {
            "feature_names": self.feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            "feature_importance": feature_importance.tolist(),
            "mean_shap_values": shap_values.mean(axis=0).tolist(),
        }

        return shap_values, summary

    def plot_summary(
        self, X: np.ndarray, save_path: Path | None = None, max_display: int = 20
    ) -> None:
        """Plot SHAP summary plot.

        Args:
            X: Feature matrix to explain.
            save_path: Optional path to save figure.
            max_display: Maximum number of features to display.
        """
        shap_values, _ = self.explain_batch(X)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_waterfall(
        self, X_instance: np.ndarray, save_path: Path | None = None
    ) -> None:
        """Plot SHAP waterfall plot for a single instance.

        Args:
            X_instance: Single instance to explain.
            save_path: Optional path to save figure.
        """
        shap_values, _ = self.explain_instance(X_instance)

        # Create SHAP Explanation object for waterfall plot
        if isinstance(self.explainer, shap.TreeExplainer):
            expected_value = (
                self.explainer.expected_value[1]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )
        else:
            expected_value = (
                self.explainer.expected_value[1]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )

        explanation = shap.Explanation(
            values=shap_values.flatten(),
            base_values=expected_value,
            data=X_instance,
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_feature_importance(
        self, X: np.ndarray, save_path: Path | None = None, top_n: int = 20
    ) -> None:
        """Plot feature importance based on mean absolute SHAP values.

        Args:
            X: Feature matrix to explain.
            save_path: Optional path to save figure.
            top_n: Number of top features to display.
        """
        _, summary = self.explain_batch(X)

        # Get top N features
        feature_names = summary["feature_names"]
        importance = np.array(summary["feature_importance"])
        top_indices = np.argsort(importance)[-top_n:][::-1]

        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel("Mean |SHAP Value|")
        plt.title(f"Top {top_n} Feature Importance (SHAP)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_explanation_report(
        self,
        X: np.ndarray,
        output_dir: Path,
        model_name: str = "model",
        sample_size: int | None = None,
    ) -> dict:
        """Generate comprehensive explanation report.

        Args:
            X: Feature matrix to explain.
            output_dir: Directory to save explanation artifacts.
            model_name: Name of the model for file naming.
            sample_size: Optional sample size for explanation (for efficiency).

        Returns:
            Summary dictionary with feature importance.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sample data if needed
        if sample_size and len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Generate explanations
        shap_values, summary = self.explain_batch(X_sample)

        # Save SHAP values
        np.save(output_dir / f"{model_name}_shap_values.npy", shap_values)

        # Save summary
        import json

        with open(output_dir / f"{model_name}_shap_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Generate plots
        self.plot_summary(X_sample, output_dir / f"{model_name}_shap_summary.png")
        self.plot_feature_importance(X_sample, output_dir / f"{model_name}_feature_importance.png")

        # Example waterfall plot for a high-probability instance
        y_pred_proba = self.model.predict_proba(X_sample)[:, 1]
        high_prob_idx = np.argmax(y_pred_proba)
        self.plot_waterfall(
            X_sample[high_prob_idx], output_dir / f"{model_name}_waterfall_example.png"
        )

        return summary

