"""Phase 3 modeling workflow script."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.evaluation import ModelEvaluator
from src.models.explainability import ModelExplainer
from src.models.trainer import ModelTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 3 modeling workflow")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed feature datasets",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["baseline", "xgboost", "lightgbm", "all"],
        default="all",
        help="Type of model to train",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("configs/model_config.json"),
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained models and artifacts",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="telco_churn",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help="Use latest processed snapshot from input directory",
    )
    return parser.parse_args()


def load_latest_features(input_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the latest processed features.

    Args:
        input_dir: Directory containing processed feature datasets.

    Returns:
        Tuple of (features_df, target_series).
    """
    # Find all timestamped directories
    timestamp_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        raise FileNotFoundError(f"No processed snapshots found in {input_dir}")

    # Get latest by timestamp
    latest_dir = max(timestamp_dirs, key=lambda p: p.name)

    # Load features and target
    train_path = latest_dir / "train.parquet"
    target_path = latest_dir / "target.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Features file not found: {train_path}")

    X = pd.read_parquet(train_path)
    y = None

    if target_path.exists():
        y_df = pd.read_parquet(target_path)
        y = y_df.iloc[:, 0]  # Get first column

    if y is None:
        raise ValueError("Target not found. Please ensure target.parquet exists.")

    return X, y


def load_config(config_file: Path, model_type: str) -> TrainingConfig:
    """Load training configuration.

    Args:
        config_file: Path to configuration file.
        model_type: Model type to load config for.

    Returns:
        TrainingConfig object.
    """
    with open(config_file) as f:
        configs = json.load(f)

    if model_type not in configs:
        raise ValueError(f"Configuration for {model_type} not found in {config_file}")

    config_dict = configs[model_type]
    return TrainingConfig(**config_dict)


def main() -> None:
    """Run modeling workflow."""
    args = parse_args()
    console = Console()

    # Load data
    console.print("[bold cyan]Loading processed features...[/bold cyan]")
    X, y = load_latest_features(args.input_dir)
    console.print(f"[green]Loaded features: {X.shape}[/green]")
    console.print(f"[green]Target distribution: {y.value_counts().to_dict()}[/green]")

    # Convert to numpy arrays
    X_array = X.values
    y_array = y.values

    # Split data
    console.print("[bold cyan]Splitting data into train/test sets...[/bold cyan]")
    X_train, X_test, y_train, y_test = train_test_split(
        X_array,
        y_array,
        test_size=0.2,
        random_state=42,
        stratify=y_array,
    )
    console.print(f"[green]Train: {X_train.shape[0]} samples[/green]")
    console.print(f"[green]Test: {X_test.shape[0]} samples[/green]")

    # Further split train into train/val for hyperparameter tuning
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # Determine which models to train
    model_types = []
    if args.model_type == "all":
        model_types = ["baseline", "xgboost", "lightgbm"]
    else:
        model_types = [args.model_type]

    # Create output directory
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model_type in model_types:
        console.print(f"\n[bold cyan]Training {model_type} model...[/bold cyan]")

        # Load config
        config = load_config(args.config_file, model_type)

        # Train model
        trainer = ModelTrainer(config, mlflow_experiment_name=args.mlflow_experiment)
        model = trainer.train(
            X_train_fit,
            y_train_fit,
            X_val=X_val,
            y_val=y_val,
        )

        # Evaluate model
        console.print(f"[bold cyan]Evaluating {model_type} model...[/bold cyan]")
        evaluator = ModelEvaluator(model, feature_names=list(X.columns))
        test_metrics = evaluator.generate_report(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir / model_type,
            model_name=model_type,
        )

        # Generate explanations (skip for now if it fails)
        try:
            console.print(f"[bold cyan]Generating explanations for {model_type} model...[/bold cyan]")
            explainer = ModelExplainer(
                model=model,
                X_background=X_train[:1000],  # Sample for efficiency
                feature_names=list(X.columns),
            )
            explainer.generate_explanation_report(
                X=X_test,
                output_dir=output_dir / model_type,
                model_name=model_type,
                sample_size=500,  # Sample for efficiency
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate explanations: {e}[/yellow]")
            console.print("[yellow]Continuing without explanations...[/yellow]")

        # Save model
        import joblib

        model_path = output_dir / model_type / f"{model_type}_model.pkl"
        joblib.dump(model, model_path)
        console.print(f"[green]Saved model to: {model_path}[/green]")

        # Save feature pipeline (load from Phase 2 output)
        # Find the feature pipeline from the processed data directory
        processed_dir = args.input_dir
        timestamp_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        if timestamp_dirs:
            latest_processed_dir = max(timestamp_dirs, key=lambda p: p.name)
            pipeline_source = latest_processed_dir / "feature_pipeline.pkl"
            if pipeline_source.exists():
                pipeline_dest = output_dir / model_type / "feature_pipeline.pkl"
                import shutil

                shutil.copy2(pipeline_source, pipeline_dest)
                console.print(f"[green]Saved feature pipeline to: {pipeline_dest}[/green]")
            else:
                console.print(
                    f"[yellow]Warning: Feature pipeline not found at {pipeline_source}[/yellow]"
                )

        # Log to MLflow
        with mlflow.start_run(run_name=f"{model_type}_final"):
            mlflow.log_metric("test_roc_auc", test_metrics.roc_auc)
            mlflow.log_metric("test_pr_auc", test_metrics.pr_auc)
            mlflow.log_metric("test_accuracy", test_metrics.accuracy)
            mlflow.log_metric("test_precision", test_metrics.precision)
            mlflow.log_metric("test_recall", test_metrics.recall)
            mlflow.log_metric("test_f1", test_metrics.f1)

            # Log model
            if model_type == "baseline":
                mlflow.sklearn.log_model(model.get_model(), "model")
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, "model")

            # Log artifacts
            mlflow.log_artifacts(str(output_dir / model_type), artifact_path="evaluation")

        results.append(
            {
                "model": model_type,
                "roc_auc": test_metrics.roc_auc,
                "pr_auc": test_metrics.pr_auc,
                "accuracy": test_metrics.accuracy,
                "precision": test_metrics.precision,
                "recall": test_metrics.recall,
                "f1": test_metrics.f1,
            }
        )

        console.print(f"[green]✓ {model_type} model complete![/green]")
        console.print(f"  ROC-AUC: {test_metrics.roc_auc:.4f}")
        console.print(f"  PR-AUC: {test_metrics.pr_auc:.4f}")
        console.print(f"  F1: {test_metrics.f1:.4f}")

    # Display summary
    console.print("\n[bold green]Modeling Complete![/bold green]\n")

    table = Table(title="Model Performance Summary")
    table.add_column("Model", style="cyan")
    table.add_column("ROC-AUC", style="green")
    table.add_column("PR-AUC", style="green")
    table.add_column("F1", style="green")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")

    for result in results:
        table.add_row(
            result["model"],
            f"{result['roc_auc']:.4f}",
            f"{result['pr_auc']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
        )

    console.print(table)

    # Find best model
    best_model = max(results, key=lambda x: x["roc_auc"])
    console.print(
        f"\n[bold green]Best Model: {best_model['model']} (ROC-AUC: {best_model['roc_auc']:.4f})[/bold green]"
    )

    # Save summary
    summary_path = output_dir / "model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "results": results,
                "best_model": best_model,
            },
            f,
            indent=2,
        )
    console.print(f"[green]Saved summary to: {summary_path}[/green]")


if __name__ == "__main__":
    main()
