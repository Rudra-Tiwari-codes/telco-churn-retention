"""Phase 4 streaming pipeline script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.api.streaming import run_streaming_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 4 streaming pipeline")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--kafka-topic",
        type=str,
        default="customer_events",
        help="Kafka topic to consume from",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of events to process in a batch",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use Kafka simulator (default: True if KAFKA_BOOTSTRAP_SERVERS not set)",
    )
    parser.add_argument(
        "--no-simulate",
        dest="simulate",
        action="store_false",
        help="Use actual Kafka (requires Kafka running)",
    )
    
    args = parser.parse_args()
    
    # Check if simulate was explicitly set via command line arguments
    # If --simulate or --no-simulate was provided, args.simulate will be True or False
    # If neither was provided, we need to check sys.argv to detect this
    import sys
    simulate_explicitly_set = "--simulate" in sys.argv or "--no-simulate" in sys.argv
    args._simulate_explicitly_set = simulate_explicitly_set
    
    return args


def main() -> None:
    """Run streaming pipeline."""
    import os
    
    args = parse_args()
    
    # Default to simulation if KAFKA_BOOTSTRAP_SERVERS is not set
    # If neither --simulate nor --no-simulate was provided, use environment-based default
    if not args._simulate_explicitly_set:
        # Neither flag was provided, use environment-based default
        simulate = not bool(os.getenv("KAFKA_BOOTSTRAP_SERVERS"))
    else:
        # User explicitly set --simulate or --no-simulate
        simulate = args.simulate

    run_streaming_pipeline(
        model_dir=args.model_dir,
        kafka_topic=args.kafka_topic,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        batch_size=args.batch_size,
        simulate=simulate,
    )


if __name__ == "__main__":
    main()

