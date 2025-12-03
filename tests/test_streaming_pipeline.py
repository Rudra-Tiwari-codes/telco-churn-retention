"""Test streaming pipeline."""

from pathlib import Path

import pytest

from src.api.streaming import run_streaming_pipeline

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streaming_pipeline_simulation() -> None:
    """Test streaming pipeline in simulation mode."""
    # Test with simulation mode
    model_dir = PROJECT_ROOT / "models"
    if not model_dir.exists():
        pytest.skip("Models directory not found - skipping streaming pipeline test")

    # Check if any models exist
    model_subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not model_subdirs:
        pytest.skip("No model directories found - skipping streaming pipeline test")

    try:
        run_streaming_pipeline(
            model_dir=model_dir,
            simulate=True,
            batch_size=5,  # Process 5 events
        )
    except Exception as e:
        pytest.fail(f"Streaming pipeline test failed: {e}")
