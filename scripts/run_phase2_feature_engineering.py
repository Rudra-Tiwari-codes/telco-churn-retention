"""Phase 2 feature engineering workflow script."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.ingestion import clean_dataset, load_raw_dataset
from src.features.pipeline import apply_feature_pipeline, create_feature_pipeline
from src.features.store import FeatureMetadata, FeatureStore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 2 feature engineering workflow")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed"),
        help="Path to input processed dataset or directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to store processed feature datasets",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/feature_metadata"),
        help="Directory to store feature metadata",
    )
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help="Use latest processed snapshot from input directory",
    )
    return parser.parse_args()


def load_latest_snapshot(processed_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Load the latest processed snapshot.

    Args:
        processed_dir: Directory containing processed snapshots.

    Returns:
        Tuple of (dataframe, snapshot_path).
    """
    # Find all timestamped directories
    timestamp_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        raise FileNotFoundError(f"No processed snapshots found in {processed_dir}")

    # Get latest by timestamp
    latest_dir = max(timestamp_dirs, key=lambda p: p.name)
    parquet_files = list(latest_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {latest_dir}")

    # Prefer telco_churn.parquet or any file that's not target.parquet
    # target.parquet only contains the target column, not the features
    snapshot_path = None
    for pf in parquet_files:
        if pf.name != "target.parquet":
            snapshot_path = pf
            break

    # If only target.parquet exists, that's an error - we need feature data
    if snapshot_path is None:
        raise FileNotFoundError(
            f"Only target.parquet found in {latest_dir}. "
            f"Need feature data file (telco_churn.parquet or train.parquet)."
        )

    df = pd.read_parquet(snapshot_path)

    return df, snapshot_path


def create_provenance_log(
    input_path: Path,
    output_path: Path,
    feature_count: int,
    pipeline_config: dict,
) -> dict:
    """Create provenance log for processed dataset.

    Args:
        input_path: Path to input dataset.
        output_path: Path to output dataset.
        feature_count: Number of features generated.
        pipeline_config: Pipeline configuration.

    Returns:
        Provenance log dictionary.
    """
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "feature_count": feature_count,
        "pipeline_config": pipeline_config,
        "created_at": datetime.now(UTC).isoformat(),
        "version": "1.0",
    }


def main() -> None:
    """Run feature engineering workflow."""
    args = parse_args()
    console = Console()

    # Load input data
    console.print("[bold cyan]Loading input data...[/bold cyan]")
    if args.use_latest or args.input_path.is_dir():
        df, input_path = load_latest_snapshot(args.input_path)
        console.print(f"[green]Loaded latest snapshot: {input_path}[/green]")
    elif args.input_path.suffix == ".parquet":
        df = pd.read_parquet(args.input_path)
        input_path = args.input_path
        console.print(f"[green]Loaded dataset: {input_path}[/green]")
    elif args.input_path.suffix == ".csv":
        # Load raw CSV and clean it
        df = load_raw_dataset(args.input_path)
        df = clean_dataset(df)
        input_path = args.input_path
        console.print(f"[green]Loaded and cleaned raw dataset: {input_path}[/green]")
    else:
        raise ValueError(f"Unsupported input file type: {args.input_path}")

    console.print(f"[dim]Input shape: {df.shape}[/dim]")

    # Create feature pipeline
    console.print("[bold cyan]Creating feature pipeline...[/bold cyan]")
    pipeline = create_feature_pipeline()

    # Fit pipeline
    console.print("[bold cyan]Fitting feature pipeline...[/bold cyan]")
    pipeline.fit(df)

    # Transform data
    console.print("[bold cyan]Applying feature transformations...[/bold cyan]")
    X_transformed, y = apply_feature_pipeline(df, pipeline, target_column="Churn")

    console.print(f"[green]Transformed shape: {X_transformed.shape}[/green]")
    console.print(f"[dim]Features: {list(X_transformed.columns[:10])}...[/dim]")

    # Create output directory with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature pipeline
    import joblib

    pipeline_path = output_dir / "feature_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    console.print(f"[green]Saved feature pipeline to: {pipeline_path}[/green]")

    # Save processed features
    output_path = output_dir / "train.parquet"
    X_transformed.to_parquet(output_path, index=False)
    console.print(f"[green]Saved processed features to: {output_path}[/green]")

    # Save target if present
    if y is not None:
        target_path = output_dir / "target.parquet"
        y.to_frame("Churn").to_parquet(target_path, index=False)
        console.print(f"[green]Saved target to: {target_path}[/green]")

    # Create provenance log
    provenance_log = create_provenance_log(
        input_path=input_path,
        output_path=output_path,
        feature_count=len(X_transformed.columns),
        pipeline_config={"pipeline_type": "sklearn", "version": "1.0"},
    )
    provenance_path = output_dir / "provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance_log, f, indent=2)
    console.print(f"[green]Saved provenance log to: {provenance_path}[/green]")

    # Register with feature store
    console.print("[bold cyan]Registering features with feature store...[/bold cyan]")
    feature_store = FeatureStore(args.metadata_dir)

    # Create feature metadata
    feature_definitions = [
        FeatureMetadata(
            name=col,
            dtype=str(X_transformed[col].dtype),
            description=f"Engineered feature: {col}",
        )
        for col in X_transformed.columns
    ]

    metadata = feature_store.register_feature_set(
        feature_set_name="telco_churn_features",
        df=X_transformed,
        feature_definitions=feature_definitions,
        transformer_config={"pipeline_type": "sklearn"},
        version=timestamp,
    )
    console.print(
        f"[green]Registered feature set: {metadata.feature_set_name} v{metadata.version}[/green]"
    )

    # Display summary
    console.print("\n[bold green]Feature Engineering Complete![/bold green]\n")

    table = Table(title="Feature Engineering Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input Shape", f"{df.shape[0]} rows × {df.shape[1]} cols")
    table.add_row("Output Shape", f"{X_transformed.shape[0]} rows × {X_transformed.shape[1]} cols")
    table.add_row("Features Generated", str(len(X_transformed.columns)))
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Feature Set Version", metadata.version)

    console.print(table)


if __name__ == "__main__":
    main()
