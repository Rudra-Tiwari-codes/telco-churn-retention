"""Test streaming pipeline."""

from pathlib import Path

import pytest

from src.api.streaming import run_streaming_pipeline

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streaming_pipeline_simulation():
    """Test streaming pipeline in simulation mode."""
    print("=" * 80)
    print("STREAMING PIPELINE TEST")
    print("=" * 80)

    # Test with simulation mode
    model_dir = PROJECT_ROOT / "models"
    if not model_dir.exists():
        pytest.skip("Models directory not found - skipping streaming pipeline test")

    print(f"\n[OK] Model directory found: {model_dir}")
    print("  Running streaming pipeline in simulation mode...\n")

    try:
        run_streaming_pipeline(
            model_dir=model_dir,
            simulate=True,
            batch_size=5,  # Process 5 events
        )
        print("\n[OK] Streaming pipeline test completed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Streaming pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Streaming pipeline test failed: {e}")
