"""Phase 4 API service script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 4 API service")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    return parser.parse_args()


def main() -> None:
    """Run API service."""
    args = parse_args()

    # Set environment variables for model service
    import os

    os.environ["MODEL_DIR"] = str(args.model_dir.resolve())
    os.environ["PREDICTION_THRESHOLD"] = str(args.threshold)

    # Run uvicorn server
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
