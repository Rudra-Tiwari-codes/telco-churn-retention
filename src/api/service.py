"""
Service layer for model loading, feature transformation, and predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.pipeline import apply_feature_pipeline, create_feature_pipeline
from src.models.baseline import BaselineModel
from src.models.explainability import ModelExplainer


class ModelService:
    """Service for loading models and making predictions."""

    def __init__(
        self,
        model_path: Path | None = None,
        pipeline_path: Path | None = None,
        model_dir: Path | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Initialize model service.

        Args:
            model_path: Path to saved model file.
            pipeline_path: Path to saved feature pipeline file.
            model_dir: Directory containing model artifacts (will find latest).
            threshold: Prediction threshold for binary classification.
        """
        self.model: Any = None
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.model_type: str = ""
        self.model_version: str = ""
        self.performance_metrics: dict[str, float] = {}
        self.threshold = threshold
        self.explainer: ModelExplainer | None = None
        self.X_background: np.ndarray | None = None

        # Load model and pipeline
        if model_dir:
            self._load_from_directory(model_dir)
        elif model_path and pipeline_path:
            self._load_model(model_path)
            self._load_pipeline(pipeline_path)
        else:
            raise ValueError("Must provide either model_dir or both model_path and pipeline_path")

    def _load_from_directory(self, model_dir: Path) -> None:
        """Load model and pipeline from directory.

        Looks for latest timestamped directory or specific model type.
        """
        model_dir = Path(model_dir)

        # Find model files
        if model_dir.is_file():
            # Direct path to model file
            model_path = model_dir
            pipeline_path = model_dir.parent / "feature_pipeline.pkl"
        elif (model_dir / "model_summary.json").exists():
            # Phase 3 output directory structure
            summary_path = model_dir / "model_summary.json"
            with open(summary_path) as f:
                summary = json.load(f)

            # Get best model or first available
            best_model_type = summary.get("best_model", {}).get("model", "xgboost")
            model_type_dir = model_dir / best_model_type

            if not model_type_dir.exists():
                # Try to find any model type
                model_type_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                if model_type_dirs:
                    model_type_dir = model_type_dirs[0]
                    best_model_type = model_type_dir.name
                else:
                    raise FileNotFoundError(f"No model directories found in {model_dir}")

            model_path = model_type_dir / f"{best_model_type}_model.pkl"
            pipeline_path = model_type_dir / "feature_pipeline.pkl"

            # If pipeline not in model dir, look in parent or processed data
            if not pipeline_path.exists():
                # Try parent directory
                pipeline_path = model_dir / "feature_pipeline.pkl"
                if not pipeline_path.exists():
                    # Try to find in processed data directory
                    processed_dir = Path("data/processed")
                    if processed_dir.exists():
                        timestamp_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
                        if timestamp_dirs:
                            latest_processed = max(timestamp_dirs, key=lambda p: p.name)
                            pipeline_path = latest_processed / "feature_pipeline.pkl"

            self.model_type = best_model_type
            self.model_version = summary.get("timestamp", "unknown")
            self.performance_metrics = summary.get("best_model", {})
        else:
            # Look for timestamped subdirectories
            timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if not timestamp_dirs:
                raise FileNotFoundError(f"No model directories found in {model_dir}")

            # Get latest timestamp
            latest_dir = max(timestamp_dirs, key=lambda p: p.name)
            self.model_version = latest_dir.name

            # Find model in latest directory
            summary_path = latest_dir / "model_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                best_model_type = summary.get("best_model", {}).get("model", "xgboost")
                model_type_dir = latest_dir / best_model_type
            else:
                # Find any model type directory
                model_type_dirs = [d for d in latest_dir.iterdir() if d.is_dir()]
                if model_type_dirs:
                    model_type_dir = model_type_dirs[0]
                    best_model_type = model_type_dir.name
                else:
                    raise FileNotFoundError(f"No model type directories found in {latest_dir}")

            model_path = model_type_dir / f"{best_model_type}_model.pkl"
            pipeline_path = model_type_dir / "feature_pipeline.pkl"

            if not pipeline_path.exists():
                pipeline_path = latest_dir / "feature_pipeline.pkl"

            self.model_type = best_model_type
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                self.performance_metrics = summary.get("best_model", {})

        self._load_model(model_path)
        self._load_pipeline(pipeline_path)

    def _load_model(self, model_path: Path) -> None:
        """Load model from file."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)

        # Handle BaselineModel wrapper
        if isinstance(self.model, BaselineModel):
            self.model = self.model.get_model()

    def _load_pipeline(self, pipeline_path: Path) -> None:
        """Load feature pipeline from file."""
        if pipeline_path.exists():
            self.pipeline = joblib.load(pipeline_path)
            
            # Extract feature names from pipeline
            try:
                # Try to get feature names from the preprocessor
                if hasattr(self.pipeline, "named_steps"):
                    preprocessor = self.pipeline.named_steps.get("preprocessor")
                    if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
                        self.feature_names = list(preprocessor.get_feature_names_out())
                elif hasattr(self.pipeline, "steps"):
                    # Try to access steps directly
                    steps_dict = dict(self.pipeline.steps)
                    preprocessor = steps_dict.get("preprocessor")
                    if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
                        self.feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                # If we can't get feature names, they'll be empty
                # This is okay - they'll be populated when we make predictions
                pass
        else:
            # Pipeline not saved, create and fit with dummy data
            # This is a fallback - ideally pipeline should be saved in Phase 3
            raise FileNotFoundError(
                f"Feature pipeline not found at {pipeline_path}. "
                "Please ensure feature pipeline is saved during Phase 3 training."
            )

    def _prepare_background_data(self, processed_data_dir: Path) -> None:
        """Prepare background data for SHAP explainer.

        Args:
            processed_data_dir: Directory containing processed training data.
        """
        try:
            # Try to load from latest processed data
            timestamp_dirs = [d for d in processed_data_dir.iterdir() if d.is_dir()]
            if timestamp_dirs:
                latest_dir = max(timestamp_dirs, key=lambda p: p.name)
                train_path = latest_dir / "train.parquet"
                if train_path.exists():
                    X_train = pd.read_parquet(train_path)
                    # Sample for efficiency
                    sample_size = min(100, len(X_train))
                    X_sample = X_train.sample(n=sample_size, random_state=42)
                    self.X_background = X_sample.values
                    self.feature_names = list(X_sample.columns)
                    return

            # Fallback: create dummy background
            # This won't work well but prevents errors
            if self.feature_names:
                n_features = len(self.feature_names)
                self.X_background = np.zeros((10, n_features))
        except Exception:
            # If all else fails, use empty background
            pass

    def predict(self, customer_data: dict[str, Any], include_explanation: bool = True) -> dict[str, Any]:
        """Make prediction for a single customer.

        Args:
            customer_data: Customer data dictionary.
            include_explanation: Whether to include SHAP explanations.

        Returns:
            Prediction result dictionary.
        """
        if self.model is None or self.pipeline is None:
            raise ValueError("Model and pipeline must be loaded before prediction")

        # Convert to DataFrame
        df = pd.DataFrame([customer_data])

        # Apply feature pipeline
        X_transformed, _ = apply_feature_pipeline(df, self.pipeline, target_column=None)

        # Get feature names if not set
        if not self.feature_names:
            self.feature_names = list(X_transformed.columns)
        elif len(self.feature_names) != len(X_transformed.columns):
            # Update feature names if they don't match
            self.feature_names = list(X_transformed.columns)

        # Convert to numpy array
        X_array = X_transformed.values

        # Make prediction
        y_pred_proba = self.model.predict_proba(X_array)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(bool)

        result = {
            "customerID": customer_data["customerID"],
            "churn_probability": float(y_pred_proba[0]),
            "churn_prediction": bool(y_pred[0]),
            "threshold": self.threshold,
            "explanation": [],
        }

        # Add SHAP explanation if requested
        if include_explanation:
            try:
                explanation = self._get_explanation(X_array[0])
                result["explanation"] = explanation
            except Exception:
                # If explanation fails, continue without it
                pass

        return result

    def _get_explanation(self, X_instance: np.ndarray) -> list[dict[str, float]]:
        """Get SHAP explanation for a single instance.

        Args:
            X_instance: Feature array for single instance.

        Returns:
            List of feature explanations.
        """
        if self.explainer is None:
            # Initialize explainer if not already done
            if self.X_background is None:
                # Try to get background from processed data
                processed_dir = Path("data/processed")
                if processed_dir.exists():
                    self._prepare_background_data(processed_dir)

            if self.X_background is None or len(self.X_background) == 0:
                # Can't create explainer without background
                return []

            self.explainer = ModelExplainer(
                model=self.model,
                X_background=self.X_background,
                feature_names=self.feature_names,
            )

        try:
            shap_values, explanation_dict = self.explainer.explain_instance(X_instance)
            shap_values_flat = shap_values.flatten()

            # Get top features by absolute SHAP value
            feature_importance = [
                (self.feature_names[i], float(shap_values_flat[i]), float(X_instance[i]))
                for i in range(len(self.feature_names))
            ]
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            # Return top 10 features
            top_features = feature_importance[:10]
            return [
                {"feature": feat, "shap_value": shap_val, "value": val}
                for feat, shap_val, val in top_features
            ]
        except Exception:
            # If explanation fails, return empty list
            return []

    def get_metadata(self) -> dict[str, Any]:
        """Get model metadata.

        Returns:
            Metadata dictionary.
        """
        # Filter performance_metrics to only include numeric values
        perf_metrics = {}
        for key, value in self.performance_metrics.items():
            if isinstance(value, (int, float)):
                perf_metrics[key] = float(value)
        
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "performance_metrics": perf_metrics,
            "threshold": self.threshold,
        }

    def is_ready(self) -> bool:
        """Check if service is ready for predictions.

        Returns:
            True if model and pipeline are loaded.
        """
        return self.model is not None and self.pipeline is not None

