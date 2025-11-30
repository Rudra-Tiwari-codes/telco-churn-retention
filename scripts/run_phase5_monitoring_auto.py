"""Helper script for automatic monitoring workflow that finds latest processed data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rich.console import Console

console = Console()


def find_latest_processed_data(processed_dir: Path) -> tuple[Path | None, Path | None]:
    """Find the two most recent processed data directories with train.parquet.

    Args:
        processed_dir: Directory containing processed data snapshots.

    Returns:
        Tuple of (reference_data_path, current_data_path). Returns (None, None) if not enough data.
    """
    if not processed_dir.exists():
        console.print(f"[red]Error: Processed data directory does not exist: {processed_dir}[/red]")
        return None, None

    # Find all directories with train.parquet
    dirs_with_data = [
        d for d in processed_dir.iterdir()
        if d.is_dir() and (d / "train.parquet").exists()
    ]

    if not dirs_with_data:
        console.print(
            f"[red]Error: No processed directories with train.parquet found in {processed_dir}[/red]"
        )
        return None, None

    # Sort by directory name (timestamp)
    dirs_with_data = sorted(dirs_with_data, key=lambda p: p.name)

    if len(dirs_with_data) == 1:
        # Only one dataset available - use it for both reference and current
        console.print(
            f"[yellow]Warning: Only one processed dataset found. Using it for both reference and current.[/yellow]"
        )
        path = dirs_with_data[0] / "train.parquet"
        return path, path
    else:
        # Use second-to-last as reference, last as current
        reference_path = dirs_with_data[-2] / "train.parquet"
        current_path = dirs_with_data[-1] / "train.parquet"
        return reference_path, current_path


def main_auto() -> None:
    """Run monitoring workflow with automatic data discovery."""
    processed_dir = Path("data/processed")

    console.print("[bold cyan]Finding latest processed data...[/bold cyan]")
    reference_path, current_path = find_latest_processed_data(processed_dir)

    if reference_path is None or current_path is None:
        console.print("[red]Error: Cannot proceed without processed data.[/red]")
        sys.exit(1)

    console.print(f"[green]Reference data: {reference_path}[/green]")
    console.print(f"[green]Current data: {current_path}[/green]\n")

    # Modify sys.argv to pass the paths as command line arguments
    original_argv = sys.argv.copy()
    sys.argv = [
        "run_phase5_monitoring_auto.py",
        "--reference-data", str(reference_path),
        "--current-data", str(current_path),
    ]

    try:
        import scripts.run_phase5_monitoring as monitoring_module
        monitoring_module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main_auto()

