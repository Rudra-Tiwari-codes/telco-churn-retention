"""Test streaming pipeline."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.streaming import run_streaming_pipeline

print("=" * 80)
print("STREAMING PIPELINE TEST")
print("=" * 80)

# Test with simulation mode
model_dir = PROJECT_ROOT / "models"
if not model_dir.exists():
    print("[FAIL] Models directory not found")
    sys.exit(1)

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
    sys.exit(1)
