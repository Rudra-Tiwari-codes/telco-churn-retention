"""Phase 5 retraining pipeline script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rich.console import Console

from src.pipelines.retraining_dag import RetrainingDAG

console = Console()


def find_latest_raw_data(raw_dir: Path) -> Path | None:
    """Find the latest raw data CSV file in the directory.

    Args:
        raw_dir: Directory containing raw data files.

    Returns:
        Path to latest CSV file, or None if not found.
    """
    if not raw_dir.exists():
        return None

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        return None

    # Return the most recently modified file
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 5 retraining pipeline")

    # Find default raw data file (latest CSV in data/raw)
    raw_dir = Path("data/raw")
    default_raw_data = find_latest_raw_data(raw_dir)
    if default_raw_data is None:
        default_raw_data = Path("data/raw/telco_data_28_11_2025.csv")  # Fallback

    parser.add_argument(
        "--raw-data",
        type=Path,
        default=default_raw_data,
        help=f"Path to raw data file (default: latest CSV in data/raw or {default_raw_data})",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed data",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory for trained models",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("reports/validation"),
        help="Directory for validation reports",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="telco_churn",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--min-roc-auc",
        type=float,
        default=0.85,
        help="Minimum ROC-AUC for model promotion",
    )
    parser.add_argument(
        "--no-promotion",
        action="store_true",
        help="Disable automatic model promotion",
    )
    return parser.parse_args()


def main() -> None:
    """Run retraining pipeline."""
    args = parse_args()

    console.print("[bold blue]Phase 5: Retraining Pipeline[/bold blue]\n")

    # Initialize DAG
    dag = RetrainingDAG(
        raw_data_path=args.raw_data,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        validation_report_dir=args.validation_dir,
        mlflow_experiment=args.mlflow_experiment,
        min_roc_auc=args.min_roc_auc,
        enable_promotion=not args.no_promotion,
    )

    # Execute pipeline
    summary = dag.run()

    # Print final summary
    if summary["status"] == "success":
        console.print("\n[bold green][OK] Retraining pipeline completed successfully![/bold green]")
    else:
        console.print("\n[bold red][FAIL] Retraining pipeline failed![/bold red]")
        console.print("Check task results for details.")


if __name__ == "__main__":
    main()
