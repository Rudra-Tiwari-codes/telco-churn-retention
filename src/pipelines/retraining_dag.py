"""
Retraining DAG pipeline orchestrating validation, feature build, training, and promotion.

This module provides a workflow that can be used standalone or integrated with
Airflow/Dagster for automated retraining.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from rich.console import Console
from sklearn.model_selection import train_test_split

from src.data.ingestion import IngestionConfig, run_ingestion
from src.data.validation import run_validation
from src.features.pipeline import apply_feature_pipeline, create_feature_pipeline
from src.models.evaluation import ModelEvaluator
from src.models.trainer import ModelTrainer, TrainingConfig

logger = logging.getLogger(__name__)
console = Console()


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_name: str
    status: TaskStatus
    message: str
    output_paths: dict[str, Path] | None = None
    error: str | None = None
    duration_seconds: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "status": self.status.value,
            "message": self.message,
            "output_paths": {k: str(v) for k, v in (self.output_paths or {}).items()},
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


class RetrainingDAG:
    """DAG for orchestrating the retraining pipeline."""

    def __init__(
        self,
        raw_data_path: Path,
        processed_dir: Path,
        models_dir: Path,
        validation_report_dir: Path,
        mlflow_experiment: str = "telco_churn",
        min_roc_auc: float = 0.85,
        enable_promotion: bool = True,
    ) -> None:
        """Initialize retraining DAG.

        Args:
            raw_data_path: Path to raw data file.
            processed_dir: Directory for processed data.
            models_dir: Directory for trained models.
            validation_report_dir: Directory for validation reports.
            mlflow_experiment: MLflow experiment name.
            min_roc_auc: Minimum ROC-AUC threshold for model promotion.
            enable_promotion: Whether to automatically promote models to production.
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.validation_report_dir = Path(validation_report_dir)
        self.mlflow_experiment = mlflow_experiment
        self.min_roc_auc = min_roc_auc
        self.enable_promotion = enable_promotion

        self.timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.results: list[TaskResult] = []

    def run(self) -> dict[str, Any]:
        """Execute the complete retraining pipeline.

        Returns:
            Dictionary with execution results.
        """
        console.print(f"[bold blue]Starting retraining pipeline at {self.timestamp}[/bold blue]")

        try:
            # Task 1: Data Ingestion
            result1 = self._task_data_ingestion()
            self.results.append(result1)
            if result1.status == TaskStatus.FAILED:
                return self._build_summary()

            # Task 2: Data Validation
            if result1.output_paths is None:
                return self._build_summary()
            result2 = self._task_data_validation(result1.output_paths["snapshot_path"])
            self.results.append(result2)
            if result2.status == TaskStatus.FAILED:
                return self._build_summary()

            # Task 3: Feature Engineering
            result3 = self._task_feature_engineering(result1.output_paths["snapshot_path"])
            self.results.append(result3)
            if result3.status == TaskStatus.FAILED:
                return self._build_summary()
            if result3.output_paths is None:
                return self._build_summary()

            # Task 4: Model Training
            result4 = self._task_model_training(
                result3.output_paths["train_path"],
                result3.output_paths["target_path"],
            )
            self.results.append(result4)
            if result4.status == TaskStatus.FAILED:
                return self._build_summary()
            if result4.output_paths is None:
                return self._build_summary()

            # Task 5: Model Evaluation
            result5 = self._task_model_evaluation(
                result4.output_paths["model_path"],
                result3.output_paths["train_path"],
                result3.output_paths["target_path"],
            )
            self.results.append(result5)
            if result5.status == TaskStatus.FAILED:
                return self._build_summary()
            if result5.output_paths is None:
                return self._build_summary()

            # Task 6: Model Promotion (if enabled and metrics meet threshold)
            if self.enable_promotion:
                result6 = self._task_model_promotion(
                    result4.output_paths["summary_path"], result5.output_paths["metrics_path"]
                )
                self.results.append(result6)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            console.print(f"[bold red]Pipeline failed: {e}[/bold red]")

        return self._build_summary()

    def _task_data_ingestion(self) -> TaskResult:
        """Task 1: Ingest and clean raw data."""
        console.print("[cyan]Task 1: Data Ingestion[/cyan]")
        start_time = datetime.now(UTC)

        try:
            config = IngestionConfig(
                raw_path=self.raw_data_path,
                processed_dir=self.processed_dir,
                snapshot_ts=self.timestamp,
            )

            snapshot_path = run_ingestion(config)

            duration = (datetime.now(UTC) - start_time).total_seconds()
            console.print("[green][OK] Data ingestion completed[/green]")

            return TaskResult(
                task_name="data_ingestion",
                status=TaskStatus.SUCCESS,
                message="Data ingested and cleaned successfully",
                output_paths={"snapshot_path": snapshot_path},
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now(UTC) - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Data ingestion failed: {error_msg}")
            console.print(f"[red][X] Data ingestion failed: {error_msg}[/red]")

            return TaskResult(
                task_name="data_ingestion",
                status=TaskStatus.FAILED,
                message="Data ingestion failed",
                error=error_msg,
                duration_seconds=duration,
            )

    def _task_data_validation(self, snapshot_path: Path) -> TaskResult:
        """Task 2: Validate data quality."""
        console.print("[cyan]Task 2: Data Validation[/cyan]")
        start_time = datetime.now(UTC)

        try:
            # Load snapshot
            df = pd.read_parquet(snapshot_path)

            # Run validation
            validation_report_path = (
                self.validation_report_dir / f"validation_{self.timestamp}.json"
            )
            validation_results = run_validation(df, validation_report_path)

            # Check if validation passed
            if not validation_results.get("success", False):
                raise ValueError("Data validation failed - see report for details")

            duration = (datetime.now(UTC) - start_time).total_seconds()
            console.print("[green][OK] Data validation passed[/green]")

            return TaskResult(
                task_name="data_validation",
                status=TaskStatus.SUCCESS,
                message="Data validation passed",
                output_paths={"validation_report": validation_report_path},
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now(UTC) - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Data validation failed: {error_msg}")
            console.print(f"[red][X] Data validation failed: {error_msg}[/red]")

            return TaskResult(
                task_name="data_validation",
                status=TaskStatus.FAILED,
                message="Data validation failed",
                error=error_msg,
                duration_seconds=duration,
            )

    def _task_feature_engineering(self, snapshot_path: Path) -> TaskResult:
        """Task 3: Build features.

        This task creates and fits a feature pipeline on the cleaned parquet snapshot
        from Phase 1. The pipeline is then applied to generate transformed features
        which are saved for model training.

        Note on pipeline consistency:
        - This method uses the same `create_feature_pipeline()` function as notebooks
        - Both paths fit the pipeline on cleaned data (either from Phase 1 parquet or raw CSV)
        - To maintain consistency, always use `create_feature_pipeline()` from
          src.features.pipeline rather than recreating pipeline logic elsewhere
        - The fitted pipeline is saved alongside features to ensure reproducibility
        """
        console.print("[cyan]Task 3: Feature Engineering[/cyan]")
        start_time = datetime.now(UTC)

        try:
            # Load cleaned data snapshot from Phase 1
            # This ensures consistency with the data validation step
            df = pd.read_parquet(snapshot_path)

            # Create feature pipeline using the standard factory function
            # This ensures consistency with notebook-based workflows
            pipeline = create_feature_pipeline()

            # Fit pipeline on the data (must be done before applying transformations)
            # The pipeline learns statistics (e.g., mean for imputation, scale for normalization)
            # from the training data
            pipeline.fit(df)

            # Apply pipeline to transform features
            # This drops metadata columns (customerID) and target (Churn) automatically
            X_transformed, y = apply_feature_pipeline(df, pipeline, target_column="Churn")

            # Create output directory
            output_dir = self.processed_dir / self.timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save features and target
            train_path = output_dir / "train.parquet"
            target_path = output_dir / "target.parquet"
            pipeline_path = output_dir / "feature_pipeline.pkl"

            X_transformed.to_parquet(train_path, index=False)
            # Convert target Series to DataFrame for parquet saving
            y_df = pd.DataFrame({"Churn": y})
            y_df.to_parquet(target_path, index=False)

            # Save pipeline
            import joblib

            joblib.dump(pipeline, pipeline_path)

            # Save provenance
            provenance = {
                "timestamp": self.timestamp,
                "source_snapshot": str(snapshot_path),
                "feature_count": len(X_transformed.columns),
                "sample_count": len(X_transformed),
            }
            provenance_path = output_dir / "provenance.json"
            with open(provenance_path, "w") as f:
                json.dump(provenance, f, indent=2)

            duration = (datetime.now(UTC) - start_time).total_seconds()
            console.print(
                f"[green][OK] Feature engineering completed ({len(X_transformed.columns)} features)[/green]"
            )

            return TaskResult(
                task_name="feature_engineering",
                status=TaskStatus.SUCCESS,
                message=f"Features created: {len(X_transformed.columns)} features",
                output_paths={
                    "train_path": train_path,
                    "target_path": target_path,
                    "pipeline_path": pipeline_path,
                    "provenance_path": provenance_path,
                },
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now(UTC) - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Feature engineering failed: {error_msg}")
            console.print(f"[red][X] Feature engineering failed: {error_msg}[/red]")

            return TaskResult(
                task_name="feature_engineering",
                status=TaskStatus.FAILED,
                message="Feature engineering failed",
                error=error_msg,
                duration_seconds=duration,
            )

    def _task_model_training(self, train_path: Path, target_path: Path) -> TaskResult:
        """Task 4: Train models."""
        console.print("[cyan]Task 4: Model Training[/cyan]")
        start_time = datetime.now(UTC)

        try:
            # Load data
            X = pd.read_parquet(train_path)
            y = pd.read_parquet(target_path).squeeze()

            # Convert Churn to binary
            if y.dtype == "object":
                y = (y == "Yes").astype(int)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Initialize MLflow
            mlflow.set_experiment(self.mlflow_experiment)

            # Train models
            config = TrainingConfig(
                model_type="xgboost",
                random_state=42,
                cv_folds=5,
                test_size=0.2,
                n_trials=20,  # Reduced for faster execution
            )

            trainer = ModelTrainer(config, mlflow_experiment_name=self.mlflow_experiment)
            trainer.train(X_train.values, y_train.values)

            # Evaluate on test set
            evaluator = ModelEvaluator(trainer.model)
            test_metrics = evaluator.evaluate(X_test.values, y_test.values)

            # Save model
            model_output_dir = self.models_dir / self.timestamp
            model_output_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_output_dir / "xgboost" / "xgboost_model.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            import joblib

            joblib.dump(trainer.model, model_path)

            # Save pipeline
            pipeline_path = train_path.parent / "feature_pipeline.pkl"
            if pipeline_path.exists():
                import shutil

                shutil.copy(pipeline_path, model_output_dir / "xgboost" / "feature_pipeline.pkl")

            # Log final model to MLflow for promotion
            mlflow_run_id = None
            with mlflow.start_run(run_name=f"retraining_{self.timestamp}"):
                active_run = mlflow.active_run()
                if active_run is not None:
                    mlflow_run_id = active_run.info.run_id

                # Log metrics
                mlflow.log_metric("test_roc_auc", test_metrics.roc_auc)
                mlflow.log_metric("test_pr_auc", test_metrics.pr_auc)
                mlflow.log_metric("test_accuracy", test_metrics.accuracy)
                mlflow.log_metric("test_precision", test_metrics.precision)
                mlflow.log_metric("test_recall", test_metrics.recall)
                mlflow.log_metric("test_f1", test_metrics.f1)

                # Log parameters
                mlflow.log_param("timestamp", self.timestamp)
                mlflow.log_param("model_type", "xgboost")
                if trainer.best_params:
                    mlflow.log_params(trainer.best_params)

                # Log model
                mlflow.xgboost.log_model(trainer.model, "model")

                # Log artifacts
                if (model_output_dir / "xgboost" / "feature_pipeline.pkl").exists():
                    mlflow.log_artifact(
                        str(model_output_dir / "xgboost" / "feature_pipeline.pkl"),
                        artifact_path="artifacts",
                    )

                console.print(f"[dim]MLflow run ID: {mlflow_run_id}[/dim]")

            # Save summary
            summary = {
                "timestamp": self.timestamp,
                "model_type": "xgboost",
                "roc_auc": float(test_metrics.roc_auc),
                "accuracy": float(test_metrics.accuracy),
                "f1": float(test_metrics.f1),
                "precision": float(test_metrics.precision),
                "recall": float(test_metrics.recall),
                "mlflow_run_id": mlflow_run_id,
                "best_model": {
                    "model": "xgboost",
                    "roc_auc": float(test_metrics.roc_auc),
                    "accuracy": float(test_metrics.accuracy),
                },
            }

            summary_path = model_output_dir / "model_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            duration = (datetime.now(UTC) - start_time).total_seconds()
            console.print(
                f"[green][OK] Model training completed (ROC-AUC: {test_metrics.roc_auc:.4f})[/green]"
            )

            return TaskResult(
                task_name="model_training",
                status=TaskStatus.SUCCESS,
                message=f"Model trained with ROC-AUC: {test_metrics.roc_auc:.4f}",
                output_paths={
                    "model_path": model_path,
                    "summary_path": summary_path,
                    "pipeline_path": model_output_dir / "xgboost" / "feature_pipeline.pkl",
                },
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now(UTC) - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Model training failed: {error_msg}")
            console.print(f"[red][X] Model training failed: {error_msg}[/red]")

            return TaskResult(
                task_name="model_training",
                status=TaskStatus.FAILED,
                message="Model training failed",
                error=error_msg,
                duration_seconds=duration,
            )

    def _task_model_evaluation(
        self, model_path: Path, train_path: Path, target_path: Path
    ) -> TaskResult:
        """Task 5: Evaluate model performance and generate model card."""
        console.print("[cyan]Task 5: Model Evaluation[/cyan]")
        start_time = datetime.now(UTC)

        try:
            # Load model and data
            import joblib

            model = joblib.load(model_path)
            X = pd.read_parquet(train_path)
            y = pd.read_parquet(target_path).squeeze()

            if y.dtype == "object":
                y = (y == "Yes").astype(int)

            # Split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Evaluate
            evaluator = ModelEvaluator(model)
            test_metrics = evaluator.evaluate(X_test.values, y_test.values)

            # Save metrics
            metrics_path = model_path.parent.parent / "evaluation_metrics.json"
            test_metrics.to_json(metrics_path)

            # Generate lightweight model card in markdown for reporting
            reports_dir = Path("reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            model_card_path = reports_dir / f"model_card_{self.timestamp}.md"
            with open(model_card_path, "w") as f:
                f.write(f"# Telco Churn Model Card - {self.timestamp}\n\n")
                f.write("## Overview\n")
                f.write("- **Objective**: Predict customer churn with calibrated probabilities.\n")
                f.write(
                    "- **Model Type**: Gradient boosting (XGBoost) trained via Optuna within the RetrainingDAG.\n\n"
                )
                f.write("## Data\n")
                f.write(
                    "- **Source**: Cleaned Telco churn snapshot from Phase 1 ingestion/validation.\n"
                )
                f.write("- **Target**: Binary `Churn` label derived from original dataset.\n\n")
                f.write("## Metrics (Test Set)\n\n")
                f.write("| Metric | Value |\n")
                f.write("| --- | --- |\n")
                f.write(f"| ROC-AUC | {test_metrics.roc_auc:.4f} |\n")
                f.write(f"| PR-AUC | {test_metrics.pr_auc:.4f} |\n")
                f.write(f"| Accuracy | {test_metrics.accuracy:.4f} |\n")
                f.write(f"| Precision | {test_metrics.precision:.4f} |\n")
                f.write(f"| Recall | {test_metrics.recall:.4f} |\n")
                f.write(f"| F1 | {test_metrics.f1:.4f} |\n\n")
                f.write("## Top-Decile Focus\n")
                f.write(
                    "- Evaluation includes precision and recall at top-k percentiles to support retention playbooks.\n\n"
                )
                f.write("## Assumptions & Caveats\n")
                f.write(
                    "- Assumes feature pipeline from Phase 2 is applied consistently in training and serving.\n"
                )
                f.write(
                    "- Assumes monitored drift and performance degradation are addressed via the RetrainingDAG.\n"
                )
                f.write(
                    "- Model is intended for churn prioritization, not for making pricing or credit decisions.\n"
                )

            duration = (datetime.now(UTC) - start_time).total_seconds()
            console.print("[green][OK] Model evaluation completed[/green]")

            return TaskResult(
                task_name="model_evaluation",
                status=TaskStatus.SUCCESS,
                message="Model evaluation completed",
                output_paths={
                    "metrics_path": metrics_path,
                    "model_card_path": model_card_path,
                },
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now(UTC) - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Model evaluation failed: {error_msg}")
            console.print(f"[red][X] Model evaluation failed: {error_msg}[/red]")

            return TaskResult(
                task_name="model_evaluation",
                status=TaskStatus.FAILED,
                message="Model evaluation failed",
                error=error_msg,
                duration_seconds=duration,
            )

    def _task_model_promotion(self, summary_path: Path, metrics_path: Path) -> TaskResult:
        """Task 6: Promote model to production if metrics meet threshold.

        This task:
        1. Checks if metrics meet the promotion threshold
        2. Registers the model in MLflow Model Registry
        3. Transitions it to Production stage
        4. Archives any existing Production models
        """
        console.print("[cyan]Task 6: Model Promotion[/cyan]")
        start_time = datetime.now(UTC)

        try:
            # Load metrics
            with open(metrics_path) as f:
                metrics_data = json.load(f)

            roc_auc = metrics_data.get("roc_auc", 0.0)

            if roc_auc < self.min_roc_auc:
                message = f"Model does not meet promotion threshold (ROC-AUC: {roc_auc:.4f} < {self.min_roc_auc})"
                console.print(f"[yellow][WARN] {message}[/yellow]")

                return TaskResult(
                    task_name="model_promotion",
                    status=TaskStatus.SKIPPED,
                    message=message,
                    duration_seconds=(datetime.now(UTC) - start_time).total_seconds(),
                )

            # Load summary to get MLflow run ID
            with open(summary_path) as f:
                summary_data = json.load(f)

            mlflow_run_id = summary_data.get("mlflow_run_id")
            if not mlflow_run_id:
                raise ValueError(
                    "MLflow run ID not found in summary. Model must be logged to MLflow first."
                )

            # Set up MLflow
            mlflow.set_experiment(self.mlflow_experiment)
            from mlflow.tracking import MlflowClient

            client = MlflowClient()

            # Model registry name
            model_name = "telco_churn_xgboost"

            console.print(f"[dim]Registering model from run: {mlflow_run_id}[/dim]")

            # Register the model in MLflow Model Registry
            model_uri = f"runs:/{mlflow_run_id}/model"
            try:
                # Register model version (MLflow automatically assigns version numbers)
                model_version = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=mlflow_run_id,
                    description=f"Retraining run {self.timestamp}, ROC-AUC: {roc_auc:.4f}",
                    tags={
                        "timestamp": self.timestamp,
                        "roc_auc": str(roc_auc),
                        "model_type": "xgboost",
                    },
                )

                console.print(f"[green]Registered model version: {model_version.version}[/green]")

                # Archive any existing Production models
                try:
                    production_versions = client.get_latest_versions(
                        model_name, stages=["Production"]
                    )
                    for prod_version in production_versions:
                        console.print(
                            f"[yellow]Archiving previous Production version: {prod_version.version}[/yellow]"
                        )
                        client.transition_model_version_stage(
                            name=model_name,
                            version=prod_version.version,
                            stage="Archived",
                            archive_existing_versions=True,
                        )
                except Exception as e:
                    # No existing Production models or error archiving
                    logger.debug(f"No existing Production models to archive: {e}")

                # Transition new model to Staging first (best practice)
                console.print("[dim]Transitioning to Staging...[/dim]")
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging",
                    archive_existing_versions=False,
                )

                # Then transition to Production
                console.print("[dim]Transitioning to Production...[/dim]")
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Production",
                    archive_existing_versions=False,
                )

                duration = (datetime.now(UTC) - start_time).total_seconds()
                message = (
                    f"Model promoted to production (ROC-AUC: {roc_auc:.4f}, "
                    f"Version: {model_version.version})"
                )
                console.print(f"[green][OK] {message}[/green]")

                # Create promotion metadata file
                promotion_metadata = {
                    "model_registry_name": model_name,
                    "model_version": str(model_version.version),
                    "roc_auc": roc_auc,
                    "timestamp": self.timestamp,
                    "run_id": mlflow_run_id,
                }
                promotion_metadata_path = summary_path.parent / "promotion_metadata.json"
                with open(promotion_metadata_path, "w") as f:
                    json.dump(promotion_metadata, f, indent=2)

                return TaskResult(
                    task_name="model_promotion",
                    status=TaskStatus.SUCCESS,
                    message=message,
                    output_paths={
                        "promotion_metadata": promotion_metadata_path,
                    },
                    duration_seconds=duration,
                )

            except Exception as reg_error:
                # Model registration might fail if MLflow backend doesn't support it
                # (e.g., file-based tracking). Log warning but don't fail.
                error_msg = str(reg_error)
                logger.warning(f"Model registry not available: {error_msg}")
                logger.warning("Continuing with simplified promotion (file-based only)")

                # For file-based tracking, just log the promotion
                duration = (datetime.now(UTC) - start_time).total_seconds()
                message = (
                    f"Model ready for production (ROC-AUC: {roc_auc:.4f}). "
                    f"Model Registry not available (file-based tracking)."
                )
                console.print(f"[yellow][WARN] {message}[/yellow]")

                return TaskResult(
                    task_name="model_promotion",
                    status=TaskStatus.SUCCESS,
                    message=message,
                    duration_seconds=duration,
                )

        except Exception as e:
            duration = (datetime.now(UTC) - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Model promotion failed: {error_msg}")
            console.print(f"[red][X] Model promotion failed: {error_msg}[/red]")

            return TaskResult(
                task_name="model_promotion",
                status=TaskStatus.FAILED,
                message="Model promotion failed",
                error=error_msg,
                duration_seconds=duration,
            )

    def _build_summary(self) -> dict[str, Any]:
        """Build execution summary."""
        total_duration = sum(r.duration_seconds or 0 for r in self.results if r.duration_seconds)
        success_count = sum(1 for r in self.results if r.status == TaskStatus.SUCCESS)
        failed_count = sum(1 for r in self.results if r.status == TaskStatus.FAILED)

        summary = {
            "timestamp": self.timestamp,
            "status": "success" if failed_count == 0 else "failed",
            "total_tasks": len(self.results),
            "successful_tasks": success_count,
            "failed_tasks": failed_count,
            "total_duration_seconds": total_duration,
            "task_results": [r.to_dict() for r in self.results],
        }

        # Save summary
        summary_path = self.models_dir / self.timestamp / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        console.print("\n[bold]Pipeline Summary:[/bold]")
        console.print(f"  Status: {summary['status']}")
        console.print(f"  Tasks: {success_count}/{len(self.results)} successful")
        console.print(f"  Duration: {total_duration:.2f}s")

        return summary
