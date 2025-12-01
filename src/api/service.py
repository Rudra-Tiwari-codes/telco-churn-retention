"""
Service layer for model loading, feature transformation, and predictions.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.pipeline import apply_feature_pipeline
from src.models.baseline import BaselineModel
from src.models.explainability import ModelExplainer

logger = logging.getLogger(__name__)


def _resolve_processed_data_dir(model_dir: Path | None = None) -> Path | None:
    """Resolve processed data directory path.
    
    Tries multiple strategies:
    1. Environment variable PROCESSED_DATA_DIR
    2. Relative to project root (assuming standard structure)
    3. Relative to model_dir parent
    
    Args:
        model_dir: Optional model directory to use as reference.
        
    Returns:
        Resolved Path if found, None otherwise.
    """
    # Try environment variable first
    env_path = os.getenv("PROCESSED_DATA_DIR")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path.resolve()
    
    # Try relative to project root (go up from src/api/service.py)
    try:
        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"
        if processed_dir.exists():
            return processed_dir.resolve()
    except (IndexError, AttributeError):
        pass
    
    # Try relative to model_dir parent if provided
    if model_dir:
        try:
            # Assume model_dir is in project root, so go to parent/data/processed
            project_root = model_dir.resolve().parent
            processed_dir = project_root / "data" / "processed"
            if processed_dir.exists():
                return processed_dir.resolve()
        except (AttributeError, ValueError):
            pass
    
    # Try current working directory as last resort
    cwd_processed = Path("data/processed").resolve()
    if cwd_processed.exists():
        return cwd_processed
    
    return None


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

        Raises:
            FileNotFoundError: If model directory doesn't exist or is empty.
            ValueError: If model directory structure is invalid.
        """
        model_dir = Path(model_dir)

        # Validate model directory exists
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        if not model_dir.is_dir():
            raise ValueError(f"Model path is not a directory: {model_dir}")

        # Find model files
        checked_paths: list[Path] = []
        if model_dir.is_file():
            # Direct path to model file
            model_path = model_dir
            pipeline_path = model_dir.parent / "feature_pipeline.pkl"
            checked_paths.append(pipeline_path)
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
            checked_paths = [pipeline_path]
            if not pipeline_path.exists():
                # Try parent directory
                pipeline_path = model_dir / "feature_pipeline.pkl"
                checked_paths.append(pipeline_path)
                if not pipeline_path.exists():
                    # Try to find in processed data directory
                    processed_dir = _resolve_processed_data_dir(model_dir)
                    if processed_dir:
                        timestamp_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
                        if timestamp_dirs:
                            latest_processed = max(timestamp_dirs, key=lambda p: p.name)
                            pipeline_path = latest_processed / "feature_pipeline.pkl"
                            checked_paths.append(pipeline_path)

            self.model_type = best_model_type
            self.model_version = summary.get("timestamp", "unknown")
            self.performance_metrics = summary.get("best_model", {})
        else:
            # Look for timestamped subdirectories
            timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if not timestamp_dirs:
                raise FileNotFoundError(
                    f"No model directories found in {model_dir}. "
                    f"Expected structure: {model_dir}/<timestamp>/<model_type>/<model_files>"
                )

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

            # If pipeline not in model dir, look in parent or processed data (consistent with above)
            checked_paths = [pipeline_path]
            if not pipeline_path.exists():
                # Try parent directory
                pipeline_path = latest_dir / "feature_pipeline.pkl"
                checked_paths.append(pipeline_path)
                if not pipeline_path.exists():
                    # Try to find in processed data directory
                    processed_dir = _resolve_processed_data_dir(model_dir)
                    if processed_dir:
                        timestamp_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
                        if timestamp_dirs:
                            latest_processed = max(timestamp_dirs, key=lambda p: p.name)
                            pipeline_path = latest_processed / "feature_pipeline.pkl"
                            checked_paths.append(pipeline_path)

            self.model_type = best_model_type
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                self.performance_metrics = summary.get("best_model", {})

        self._load_model(model_path)
        # Ensure checked_paths includes the final pipeline_path
        if pipeline_path not in checked_paths:
            checked_paths.append(pipeline_path)
        self._load_pipeline(pipeline_path, checked_paths)

    def _load_model(self, model_path: Path) -> None:
        """Load model from file."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)

        # Handle BaselineModel wrapper
        if isinstance(self.model, BaselineModel):
            self.model = self.model.get_model()

    def _load_pipeline(self, pipeline_path: Path, checked_paths: list[Path] | None = None) -> None:
        """Load feature pipeline from file.
        
        Args:
            pipeline_path: Path to pipeline file.
            checked_paths: Optional list of all paths that were checked (for better error messages).
        """
        if pipeline_path.exists():
            self.pipeline = joblib.load(pipeline_path)
            logger.info(f"Loaded feature pipeline from {pipeline_path}")

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
            error_msg = f"Feature pipeline not found at {pipeline_path}."
            if checked_paths:
                error_msg += f"\nChecked the following locations:\n"
                for path in checked_paths:
                    error_msg += f"  - {path.resolve()}\n"
            error_msg += "Please ensure feature pipeline is saved during Phase 3 training."
            raise FileNotFoundError(error_msg)

    def _prepare_background_data(self, processed_data_dir: Path) -> None:
        """Prepare background data for SHAP explainer.

        Args:
            processed_data_dir: Directory containing processed training data.
        """
        try:
            # Try to load from latest processed data
            if not processed_data_dir.exists():
                logger.warning(
                    f"Processed data directory does not exist: {processed_data_dir}. "
                    "SHAP explanations may be inaccurate."
                )
                return
                
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
                    logger.info(f"Loaded background data from {train_path} for SHAP explanations")
                    return
                else:
                    logger.warning(
                        f"Train data not found at {train_path}. "
                        "SHAP explanations may be inaccurate."
                    )
            else:
                logger.warning(
                    f"No timestamp directories found in {processed_data_dir}. "
                    "SHAP explanations may be inaccurate."
                )

            # Fallback: create dummy background
            # This won't work well but prevents errors
            if self.feature_names:
                n_features = len(self.feature_names)
                self.X_background = np.zeros((10, n_features))
                logger.warning(
                    f"Using dummy background data (zeros) for SHAP explanations. "
                    f"This may produce inaccurate explanations. "
                    f"Please ensure processed training data is available at {processed_data_dir}."
                )
        except Exception as e:
            # If all else fails, use empty background
            logger.warning(
                f"Failed to prepare background data from {processed_data_dir}: {e}. "
                "SHAP explanations will be unavailable.",
                exc_info=True
            )

    def predict(
        self, customer_data: dict[str, Any], include_explanation: bool = True
    ) -> dict[str, Any]:
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

        # Explicitly drop customerID and other metadata columns before feature transformation
        metadata_columns = ["customerID"]
        if metadata_columns:
            df = df.drop(columns=[col for col in metadata_columns if col in df.columns])

        # Apply feature pipeline (customerID already dropped, so pass empty list to skip redundant drop)
        X_transformed, _ = apply_feature_pipeline(
            df, self.pipeline, target_column=None, metadata_columns=[]
        )

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
            except Exception as e:
                # Log the error but continue without explanation
                logger.warning(f"Failed to generate explanation: {e}", exc_info=True)

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
                # Use model_dir if available to resolve processed data directory
                processed_dir = _resolve_processed_data_dir()
                if processed_dir:
                    self._prepare_background_data(processed_dir)
                else:
                    logger.warning(
                        "Could not resolve processed data directory. "
                        "SHAP explanations will be unavailable. "
                        "Set PROCESSED_DATA_DIR environment variable or ensure data/processed exists."
                    )

            if self.X_background is None or len(self.X_background) == 0:
                # Can't create explainer without background
                logger.warning("Cannot create SHAP explainer: no background data available")
                return []

            try:
                self.explainer = ModelExplainer(
                    model=self.model,
                    X_background=self.X_background,
                    feature_names=self.feature_names,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
                return []

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
        except Exception as e:
            # Log the error but return empty list to not break predictions
            logger.warning(f"Failed to generate SHAP explanation: {e}", exc_info=True)
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
