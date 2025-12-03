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
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.api.models import CustomerData
from src.api.service import ModelService
from src.utils.retry import retry_with_backoff

# Setup logging
logger = logging.getLogger(__name__)


def camel_to_snake_case(name: str) -> str:
    """Convert camelCase to snake_case.

    Handles special cases like:
    - "customerID" -> "customer_id"
    - "MonthlyCharges" -> "monthly_charges"
    - "TotalCharges" -> "total_charges"

    Args:
        name: camelCase string.

    Returns:
        snake_case string.
    """
    # Handle special case: ID at the end
    if name.endswith("ID"):
        name = name[:-2] + "Id"

    # Insert underscore before uppercase letters, then convert to lowercase
    # Pattern: split before uppercase letters that follow lowercase letters or numbers
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Handle consecutive uppercase letters followed by lowercase
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


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
        except ValueError as e:
            # Invalid input data
            logger.warning(f"Invalid event data: {e}")
            return {
                "customerID": event.get("customerID", "unknown"),
                "error": f"Invalid data: {str(e)}",
                "processed_at": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            # Unexpected error
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

        Raises:
            ValueError: If required fields are missing or data is invalid.
        """
        # Validate required fields are present
        required_fields = ["customerID", "tenure", "MonthlyCharges"]
        missing_fields = []

        for field in required_fields:
            # Check all possible naming variants
            variants = [
                field,  # camelCase (e.g., "MonthlyCharges")
                field.lower(),  # lowercase (e.g., "monthlycharges")
                camel_to_snake_case(field),  # snake_case (e.g., "monthly_charges", "customer_id")
            ]

            # Check if any variant exists in the event
            if not any(event.get(variant) is not None for variant in variants):
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Missing required fields in event: {missing_fields}. "
                f"Event keys: {list(event.keys())}"
            )

        # Map event fields to customer data structure
        # Handle different event formats
        customer_data = {
            "customerID": event.get("customerID") or event.get("customer_id"),
            "gender": event.get("gender") or event.get("Gender") or "Male",
            "SeniorCitizen": event.get("SeniorCitizen") or event.get("senior_citizen", 0),
            "Partner": event.get("Partner") or event.get("partner", "No"),
            "Dependents": event.get("Dependents") or event.get("dependents", "No"),
            "tenure": event.get("tenure", 0),
            "PhoneService": event.get("PhoneService") or event.get("phone_service", "No"),
            "MultipleLines": event.get("MultipleLines") or event.get("multiple_lines", "No"),
            "InternetService": event.get("InternetService") or event.get("internet_service", "No"),
            "OnlineSecurity": event.get("OnlineSecurity") or event.get("online_security", "No"),
            "OnlineBackup": event.get("OnlineBackup") or event.get("online_backup", "No"),
            "DeviceProtection": event.get("DeviceProtection")
            or event.get("device_protection", "No"),
            "TechSupport": event.get("TechSupport") or event.get("tech_support", "No"),
            "StreamingTV": event.get("StreamingTV") or event.get("streaming_tv", "No"),
            "StreamingMovies": event.get("StreamingMovies") or event.get("streaming_movies", "No"),
            "Contract": event.get("Contract") or event.get("contract", "Month-to-month"),
            "PaperlessBilling": event.get("PaperlessBilling")
            or event.get("paperless_billing", "No"),
            "PaymentMethod": event.get("PaymentMethod")
            or event.get("payment_method", "Electronic check"),
            "MonthlyCharges": float(event.get("MonthlyCharges") or event.get("monthly_charges", 0)),
            "TotalCharges": (
                float(event.get("TotalCharges") or event.get("total_charges", 0))
                if event.get("TotalCharges") or event.get("total_charges")
                else None
            ),
        }

        # Validate using Pydantic model to ensure data integrity
        try:
            validated = CustomerData(**customer_data)
            return validated.model_dump()
        except Exception as e:
            logger.error(f"Invalid customer data extracted from event: {e}")
            raise ValueError(f"Invalid customer data: {e}") from e

    def store_result(self, result: dict[str, Any], customer_id: str) -> None:
        """Store prediction result in Redis and/or Postgres.

        Args:
            result: Prediction result dictionary.
            customer_id: Customer identifier.
        """
        # Store in Redis for fast access
        if self.redis_client:
            self._store_in_redis(result, customer_id)

        # Store in Postgres for persistence
        if self.postgres_client:
            self._store_in_postgres(result, customer_id)

    @retry_with_backoff(
        max_retries=3,
        initial_delay=0.5,
        max_delay=10.0,
        exceptions=(Exception,),  # Redis exceptions are various
    )
    def _store_in_redis(self, result: dict[str, Any], customer_id: str) -> None:
        """Store result in Redis with retry logic."""
        if self.redis_client is None:
            return
        key = f"churn_prediction:{customer_id}"
        value = json.dumps(result)
        self.redis_client.setex(key, 3600, value)  # Expire after 1 hour
        logger.debug(f"Stored result in Redis for customer {customer_id}")

    @retry_with_backoff(
        max_retries=3,
        initial_delay=0.5,
        max_delay=10.0,
        exceptions=(Exception,),  # Postgres exceptions are various
    )
    def _store_in_postgres(self, result: dict[str, Any], customer_id: str) -> None:
        """Store result in Postgres with retry logic."""
        # Postgres storage implementation
        # Requires a postgres_client with execute() method
        # Example schema:
        # CREATE TABLE churn_predictions (
        #     customer_id VARCHAR(255) PRIMARY KEY,
        #     churn_probability FLOAT,
        #     churn_prediction BOOLEAN,
        #     prediction_timestamp TIMESTAMP,
        #     result JSONB
        # );
        if hasattr(self.postgres_client, "execute"):
            query = """
                INSERT INTO churn_predictions
                (customer_id, churn_probability, churn_prediction, prediction_timestamp, result)
                VALUES (%s, %s, %s, NOW(), %s)
                ON CONFLICT (customer_id)
                DO UPDATE SET
                    churn_probability = EXCLUDED.churn_probability,
                    churn_prediction = EXCLUDED.churn_prediction,
                    prediction_timestamp = EXCLUDED.prediction_timestamp,
                    result = EXCLUDED.result
            """
            if self.postgres_client is None:
                return
            self.postgres_client.execute(
                query,
                (
                    customer_id,
                    result.get("churn_probability"),
                    result.get("churn_prediction"),
                    json.dumps(result),
                ),
            )
            logger.debug(f"Stored result in Postgres for customer {customer_id}")
        else:
            logger.warning("Postgres client provided but does not have execute() method")


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


def check_readiness(
    model_service: ModelService,
    redis_client: Any | None = None,
    kafka_consumer: Any | None = None,
    require_redis: bool = False,
    require_kafka: bool = False,
) -> tuple[bool, list[str]]:
    """Check readiness of all pipeline components.

    Args:
        model_service: ModelService instance to check.
        redis_client: Optional Redis client to check.
        kafka_consumer: Optional Kafka consumer to check.
        require_redis: If True, Redis must be available.
        require_kafka: If True, Kafka must be available.

    Returns:
        Tuple of (is_ready, list of error messages).
    """
    errors = []

    # Check model service
    if not model_service.is_ready():
        errors.append("Model service is not ready (model or pipeline not loaded)")

    # Check Redis if required
    if require_redis:
        if redis_client is None:
            errors.append("Redis client is required but not initialized")
        else:
            try:
                redis_client.ping()
            except Exception as e:
                errors.append(f"Redis connection failed: {e}")

    # Check Kafka if required
    if require_kafka:
        if kafka_consumer is None:
            errors.append("Kafka consumer is required but not initialized")
        elif not hasattr(kafka_consumer, "bootstrap_connected"):
            # Try to check connection by accessing a property
            try:
                _ = kafka_consumer.config
            except Exception as e:
                errors.append(f"Kafka connection failed: {e}")

    return len(errors) == 0, errors


def run_streaming_pipeline(
    model_dir: Path,
    kafka_topic: str | None = None,
    redis_host: str | None = None,
    redis_port: int | None = None,
    batch_size: int = 100,
    simulate: bool = True,
    events: list[dict[str, Any]] | None = None,
    fail_fast: bool = False,
    require_redis: bool = False,
    require_kafka: bool = False,
) -> dict[str, Any]:
    """Run the streaming pipeline.

    Args:
        model_dir: Directory containing trained models.
        kafka_topic: Kafka topic to consume from. If None, uses KAFKA_TOPIC env var or "customer_events".
        redis_host: Redis host. If None, uses REDIS_HOST env var or "localhost".
        redis_port: Redis port. If None, uses REDIS_PORT env var or 6379.
        batch_size: Number of events to process in a batch.
        simulate: Whether to use Kafka simulator (for testing).
        events: List of events for simulation.
        fail_fast: If True, raise exception on connection failures instead of logging warnings.
        require_redis: If True, Redis must be available (fail if not).
        require_kafka: If True, Kafka must be available (fail if not).

    Returns:
        Dictionary with pipeline execution results including:
        - success: Boolean indicating if pipeline completed successfully
        - processed_count: Number of events processed
        - errors: List of error messages (if any)

    Raises:
        RuntimeError: If fail_fast=True and critical components are unavailable.
    """
    import logging
    import os

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting streaming pipeline...")

    # Load configuration from environment variables if not provided
    kafka_topic = kafka_topic or os.getenv("KAFKA_TOPIC", "customer_events")
    redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
    redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))

    # Initialize model service
    try:
        model_service = ModelService(model_dir=model_dir)
        if not model_service.is_ready():
            error_msg = f"Model service failed to initialize from {model_dir}"
            if fail_fast:
                raise RuntimeError(error_msg)
            logger.error(error_msg)
            return {"success": False, "processed_count": 0, "errors": [error_msg]}
    except Exception as e:
        error_msg = f"Failed to initialize model service: {e}"
        if fail_fast:
            raise RuntimeError(error_msg) from e
        logger.error(error_msg, exc_info=True)
        return {"success": False, "processed_count": 0, "errors": [error_msg]}

    # Initialize Redis client (optional)
    redis_client = None
    redis_error = None
    if not simulate:
        try:
            import redis

            redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except ImportError as err:
            redis_error = "redis package not installed. Redis storage disabled."
            if require_redis:
                error_msg = "Redis is required but package is not installed"
                if fail_fast:
                    raise RuntimeError(error_msg) from err
                logger.error(error_msg)
                return {"success": False, "processed_count": 0, "errors": [error_msg]}
            logger.warning(redis_error)
        except Exception as e:
            redis_error = f"Could not connect to Redis at {redis_host}:{redis_port}: {e}"
            if require_redis:
                if fail_fast:
                    raise RuntimeError(redis_error) from e
                logger.error(redis_error)
                return {"success": False, "processed_count": 0, "errors": [redis_error]}
            logger.warning(f"{redis_error}. Continuing without Redis.")

    # Initialize streaming pipeline
    pipeline = StreamingPipeline(model_service=model_service, redis_client=redis_client)

    # Initialize Kafka consumer or simulator
    kafka_consumer = None
    kafka_error = None
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

            # Load Kafka bootstrap servers from environment variable
            kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(
                ","
            )

            kafka_consumer = KafkaConsumer(
                kafka_topic,
                bootstrap_servers=kafka_bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
            )
            logger.info(f"Connected to Kafka topic: {kafka_topic} at {kafka_bootstrap_servers}")
        except ImportError as err:
            kafka_error = (
                "kafka-python package not installed. Install with: pip install kafka-python"
            )
            if require_kafka:
                if fail_fast:
                    raise RuntimeError(kafka_error) from err
                logger.error(kafka_error)
                return {"success": False, "processed_count": 0, "errors": [kafka_error]}
            logger.error(kafka_error)
            return {"success": False, "processed_count": 0, "errors": [kafka_error]}
        except Exception as e:
            kafka_error = f"Could not connect to Kafka at {kafka_bootstrap_servers}: {e}"
            if require_kafka:
                if fail_fast:
                    raise RuntimeError(kafka_error) from e
                logger.error(kafka_error)
                return {"success": False, "processed_count": 0, "errors": [kafka_error]}
            logger.error(kafka_error)
            return {"success": False, "processed_count": 0, "errors": [kafka_error]}

    # Perform readiness check
    is_ready, readiness_errors = check_readiness(
        model_service=model_service,
        redis_client=redis_client,
        kafka_consumer=kafka_consumer,
        require_redis=require_redis,
        require_kafka=require_kafka,
    )

    if not is_ready:
        error_msg = f"Pipeline not ready: {'; '.join(readiness_errors)}"
        if fail_fast:
            raise RuntimeError(error_msg)
        logger.error(error_msg)
        return {"success": False, "processed_count": 0, "errors": readiness_errors}

    # Process events
    logger.info("Processing events...")
    processed_count = 0
    errors = []

    try:
        while True:
            try:
                if simulate:
                    if not kafka_consumer.has_more():
                        break
                    event = kafka_consumer.consume()
                else:
                    # Get message from Kafka
                    # Type check: when simulate is False, kafka_consumer is KafkaConsumer
                    from typing import cast

                    from kafka import KafkaConsumer as KafkaConsumerType

                    consumer = cast(KafkaConsumerType, kafka_consumer)
                    message = next(consumer)
                    event = message.value

                if event is None:
                    break

                # Process event
                result = pipeline.process_event(event)

                # Check for errors in result
                if "error" in result:
                    error_msg = f"Error processing event for customer {result.get('customerID', 'unknown')}: {result['error']}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    if fail_fast:
                        raise RuntimeError(error_msg)
                    continue

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

            except StopIteration:
                # No more messages from Kafka
                logger.info("No more messages from Kafka")
                break
            except Exception as e:
                error_msg = f"Error processing event: {e}"
                errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
                if fail_fast:
                    raise
                # Continue processing other events

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
        return {
            "success": False,
            "processed_count": processed_count,
            "errors": errors + ["Pipeline interrupted by user"],
        }
    except Exception as e:
        error_msg = f"Fatal error in streaming pipeline: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "processed_count": processed_count,
            "errors": errors + [error_msg],
        }
    finally:
        logger.info(f"Streaming pipeline completed. Processed {processed_count} events.")
        if not simulate and kafka_consumer and hasattr(kafka_consumer, "close"):
            kafka_consumer.close()

    # Return structured result
    return {
        "success": len(errors) == 0,
        "processed_count": processed_count,
        "errors": errors if errors else None,
    }
