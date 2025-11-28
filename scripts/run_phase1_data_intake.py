from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from rich.console import Console

from src.data.eda import write_markdown_summary
from src.data.ingestion import IngestionConfig, run_ingestion
from src.data.validation import run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 data intake workflow")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/telco_data_28_11_2025.csv"),
        help="Path to the raw Telco CSV",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory that stores processed snapshots",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("reports/phase1_eda.md"),
        help="Where to write the EDA markdown summary",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("reports/validation/phase1.json"),
        help="Where to write the Great Expectations report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    config = IngestionConfig(
        raw_path=args.raw_path,
        processed_dir=args.processed_dir,
    )

    console.print(f"[bold cyan]Running ingestion from {config.raw_path}[/bold cyan]")
    snapshot_path = run_ingestion(config)
    console.print(f"[green]Snapshot materialized at {snapshot_path}[/green]")

    df = pd.read_parquet(snapshot_path)
    run_validation(df, args.validation_report)
    console.print(f"[green]Validation report stored at {args.validation_report}[/green]")

    write_markdown_summary(df, args.summary_path)
    console.print(f"[green]EDA summary written to {args.summary_path}[/green]")


if __name__ == "__main__":
    main()
