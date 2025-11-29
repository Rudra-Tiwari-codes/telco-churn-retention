"""
Streaming pipeline for real-time churn prediction using Kafka and Redis.

This module implements a Kafka consumer that:
1. Consumes customer events from Kafka topics
2. Enriches events with features
3. Scores events using the trained model
4. Writes results to Redis (for fast access) and optionally Postgres (for persistence)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api.service import ModelService
from src.features.pipeline import apply_feature_pipeline

# Setup logging
logger = logging.getLogger(__name__)


class StreamingPipeline:
    """Streaming pipeline for real-time churn prediction."""

    def __init__(
        self,
        model_service: ModelService,
        redis_client: Any | None = None,
        postgres_client: Any | None = None,
    ) -> None:
        """Initialize streaming pipeline.

        Args:
            model_service: ModelService instance for making predictions.
            redis_client: Optional Redis client for caching results.
            postgres_client: Optional Postgres client for persistent storage.
        """
        self.model_service = model_service
        self.redis_client = redis_client
        self.postgres_client = postgres_client

    def process_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Process a single event and return prediction.

        Args:
            event: Customer event dictionary.

        Returns:
            Prediction result with customerID, churn_probability, timestamp, etc.
        """
        try:
            # Extract customer data from event
            customer_data = self._extract_customer_data(event)

            # Make prediction
            prediction = self.model_service.predict(customer_data, include_explanation=False)

            # Add metadata
            result = {
                **prediction,
                "event_timestamp": event.get("timestamp"),
                "processed_at": pd.Timestamp.now().isoformat(),
            }

            return result
        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)
            return {
                "customerID": event.get("customerID", "unknown"),
                "error": str(e),
                "processed_at": pd.Timestamp.now().isoformat(),
            }

    def _extract_customer_data(self, event: dict[str, Any]) -> dict[str, Any]:
        """Extract customer data from event.

        Args:
            event: Raw event dictionary.

        Returns:
            Customer data dictionary compatible with API models.
        """
        # Map event fields to customer data structure
        # Handle different event formats
        customer_data = {
            "customerID": event.get("customerID") or event.get("customer_id"),
            "gender": event.get("gender"),
            "SeniorCitizen": event.get("SeniorCitizen") or event.get("senior_citizen", 0),
            "Partner": event.get("Partner") or event.get("partner", "No"),
            "Dependents": event.get("Dependents") or event.get("dependents", "No"),
            "tenure": event.get("tenure", 0),
            "PhoneService": event.get("PhoneService") or event.get("phone_service", "No"),
            "MultipleLines": event.get("MultipleLines") or event.get("multiple_lines", "No"),
            "InternetService": event.get("InternetService") or event.get("internet_service", "No"),
            "OnlineSecurity": event.get("OnlineSecurity") or event.get("online_security", "No"),
            "OnlineBackup": event.get("OnlineBackup") or event.get("online_backup", "No"),
            "DeviceProtection": event.get("DeviceProtection") or event.get("device_protection", "No"),
            "TechSupport": event.get("TechSupport") or event.get("tech_support", "No"),
            "StreamingTV": event.get("StreamingTV") or event.get("streaming_tv", "No"),
            "StreamingMovies": event.get("StreamingMovies") or event.get("streaming_movies", "No"),
            "Contract": event.get("Contract") or event.get("contract", "Month-to-month"),
            "PaperlessBilling": event.get("PaperlessBilling") or event.get("paperless_billing", "No"),
            "PaymentMethod": event.get("PaymentMethod") or event.get("payment_method", "Electronic check"),
            "MonthlyCharges": float(event.get("MonthlyCharges") or event.get("monthly_charges", 0)),
            "TotalCharges": (
                float(event.get("TotalCharges") or event.get("total_charges", 0))
                if event.get("TotalCharges") or event.get("total_charges")
                else None
            ),
        }

        return customer_data

    def store_result(self, result: dict[str, Any], customer_id: str) -> None:
        """Store prediction result in Redis and/or Postgres.

        Args:
            result: Prediction result dictionary.
            customer_id: Customer identifier.
        """
        # Store in Redis for fast access
        if self.redis_client:
            try:
                key = f"churn_prediction:{customer_id}"
                value = json.dumps(result)
                self.redis_client.setex(key, 3600, value)  # Expire after 1 hour
                logger.debug(f"Stored result in Redis for customer {customer_id}")
            except Exception as e:
                logger.error(f"Error storing in Redis: {e}")

        # Store in Postgres for persistence
        if self.postgres_client:
            try:
                # This would insert into a predictions table
                # Implementation depends on your Postgres setup
                logger.debug(f"Would store result in Postgres for customer {customer_id}")
            except Exception as e:
                logger.error(f"Error storing in Postgres: {e}")


class KafkaSimulator:
    """Simulator for Kafka consumer (for testing without actual Kafka)."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        """Initialize Kafka simulator.

        Args:
            events: List of events to simulate.
        """
        self.events = events
        self.current_index = 0

    def consume(self, timeout: float = 1.0) -> dict[str, Any] | None:
        """Simulate consuming an event from Kafka.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Event dictionary or None if no more events.
        """
        if self.current_index >= len(self.events):
            return None

        event = self.events[self.current_index]
        self.current_index += 1
        return event

    def has_more(self) -> bool:
        """Check if there are more events to consume."""
        return self.current_index < len(self.events)


def run_streaming_pipeline(
    model_dir: Path,
    kafka_topic: str = "customer_events",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    batch_size: int = 100,
    simulate: bool = True,
    events: list[dict[str, Any]] | None = None,
) -> None:
    """Run the streaming pipeline.

    Args:
        model_dir: Directory containing trained models.
        kafka_topic: Kafka topic to consume from.
        redis_host: Redis host.
        redis_port: Redis port.
        batch_size: Number of events to process in a batch.
        simulate: Whether to use Kafka simulator (for testing).
        events: List of events for simulation.
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting streaming pipeline...")

    # Initialize model service
    model_service = ModelService(model_dir=model_dir)

    # Initialize Redis client (optional)
    redis_client = None
    if not simulate:
        try:
            import redis

            redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except ImportError:
            logger.warning("redis package not installed. Redis storage disabled.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Continuing without Redis.")

    # Initialize streaming pipeline
    pipeline = StreamingPipeline(model_service=model_service, redis_client=redis_client)

    # Initialize Kafka consumer or simulator
    if simulate:
        if events is None:
            logger.warning("No events provided for simulation. Creating sample events.")
            events = [
                {
                    "customerID": f"SIM-{i}",
                    "tenure": 12 + i,
                    "MonthlyCharges": 70.5 + i * 5,
                    "TotalCharges": 845.0 + i * 50,
                    "gender": "Male" if i % 2 == 0 else "Female",
                    "Partner": "Yes" if i % 3 == 0 else "No",
                    "Contract": "Month-to-month",
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
                for i in range(10)
            ]
        kafka_consumer = KafkaSimulator(events)
        logger.info(f"Using Kafka simulator with {len(events)} events")
    else:
        try:
            from kafka import KafkaConsumer

            kafka_consumer = KafkaConsumer(
                kafka_topic,
                bootstrap_servers=["localhost:9092"],
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
            )
            logger.info(f"Connected to Kafka topic: {kafka_topic}")
        except ImportError:
            logger.error("kafka-python package not installed. Install with: pip install kafka-python")
            return
        except Exception as e:
            logger.error(f"Could not connect to Kafka: {e}")
            return

    # Process events
    logger.info("Processing events...")
    processed_count = 0

    try:
        while True:
            if simulate:
                if not kafka_consumer.has_more():
                    break
                event = kafka_consumer.consume()
            else:
                # Get message from Kafka
                message = next(kafka_consumer)
                event = message.value

            if event is None:
                break

            # Process event
            result = pipeline.process_event(event)

            # Store result
            customer_id = result.get("customerID", "unknown")
            pipeline.store_result(result, customer_id)

            processed_count += 1
            logger.info(
                f"Processed event {processed_count}: Customer {customer_id}, "
                f"Churn probability: {result.get('churn_probability', 'N/A'):.4f}"
            )

            # Process in batches
            if processed_count >= batch_size:
                logger.info(f"Processed {processed_count} events. Continuing...")
                if simulate:
                    break  # Stop after batch in simulation

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.error(f"Error in streaming pipeline: {e}", exc_info=True)
    finally:
        logger.info(f"Streaming pipeline completed. Processed {processed_count} events.")
        if not simulate and hasattr(kafka_consumer, "close"):
            kafka_consumer.close()

