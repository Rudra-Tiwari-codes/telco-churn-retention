"""
FastAPI application for churn prediction API.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

from src.api.business_routes import router as business_router
from src.api.middleware import RateLimitMiddleware, verify_api_key
from src.api.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerData,
    HealthResponse,
    ModelMetadata,
    PredictionResponse,
)
from src.api.service import ModelService
from src.utils.logging_config import get_logger, setup_logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
setup_logging()
logger = get_logger(__name__)

# Global model service
model_service: ModelService | None = None

# Prometheus metrics (process-wide)
PREDICTION_REQUESTS = Counter(
    "telco_churn_prediction_requests_total",
    "Total number of prediction requests received",
    ["endpoint"],
)
PREDICTION_ERRORS = Counter(
    "telco_churn_prediction_errors_total",
    "Total number of prediction errors",
    ["endpoint", "reason"],
)
PREDICTION_LATENCY = Histogram(
    "telco_churn_prediction_latency_seconds",
    "Latency for prediction requests",
    ["endpoint"],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    global model_service

    # Startup: Load model and pipeline
    logger.info("Starting up API service...")
    model_dir = os.getenv("MODEL_DIR", "models")
    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

    try:
        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            error_msg = f"Model directory does not exist: {model_dir_path}"
            logger.error(error_msg)
            model_service = None
        else:
            model_service = ModelService(
                model_dir=model_dir_path,
                threshold=threshold,
            )
            if model_service.is_ready():
                logger.info(f"Model loaded successfully from {model_dir}")
            else:
                error_msg = "Model service initialized but not ready (model or pipeline not loaded)"
                logger.error(error_msg)
                model_service = None
    except FileNotFoundError as e:
        error_msg = f"Model file not found: {e}"
        logger.error(error_msg, exc_info=True)
        model_service = None
    except ValueError as e:
        error_msg = f"Invalid model configuration: {e}"
        logger.error(error_msg, exc_info=True)
        model_service = None
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        logger.error(error_msg, exc_info=True)
        # Don't raise - allow health check to show status
        model_service = None

    yield

    # Shutdown: Cleanup if needed
    logger.info("Shutting down API service...")
    model_service = None


# Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add rate limiting middleware
rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
if rate_limit_enabled:
    requests_per_minute = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
    app.add_middleware(RateLimitMiddleware, requests_per_minute=requests_per_minute)
    logger.info(f"Rate limiting enabled: {requests_per_minute} requests per minute")

# CORS middleware - configure via environment variable
# In production, default to empty list (no origins allowed) for security
is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
_cors_origins = os.getenv("CORS_ORIGINS")

if _cors_origins is None:
    # Default behavior: empty list in production, allow all in development
    cors_origins_list = [] if is_production else ["*"]
    if is_production:
        logger.warning(
            "CORS_ORIGINS not set in production. Defaulting to empty list (no origins allowed). "
            "Set CORS_ORIGINS environment variable to allow specific origins."
        )
else:
    # Validate that * is not used in production
    if _cors_origins == "*" and is_production:
        raise ValueError(
            "CORS_ORIGINS cannot be '*' in production environment. "
            "Please specify allowed origins explicitly."
        )
    # Parse comma-separated list of origins
    cors_origins_list = [origin.strip() for origin in _cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include business intelligence routes
app.include_router(business_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    model_loaded = model_service is not None and model_service.is_ready()
    pipeline_loaded = model_service is not None and model_service.pipeline is not None

    # Check external dependencies if configured
    external_deps_healthy = True
    dependency_errors = []

    # Check Redis if configured
    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        try:
            import redis

            redis_client = redis.Redis(
                host=redis_host,
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True,
                socket_connect_timeout=2,
            )
            redis_client.ping()
        except Exception as e:
            external_deps_healthy = False
            dependency_errors.append(f"Redis: {str(e)}")

    # Check Kafka if configured
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if kafka_bootstrap:
        try:
            from kafka import KafkaConsumer

            # Just check if we can create a consumer (doesn't actually connect)
            # Full connection check would require topic access
            _ = KafkaConsumer(
                bootstrap_servers=kafka_bootstrap.split(","),
                consumer_timeout_ms=100,
            )
        except Exception as e:
            # Kafka check is optional - just log warning
            logger.warning(f"Kafka health check warning: {e}")

    # Check MLflow if configured
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        try:
            import mlflow

            # Try to access MLflow tracking
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            # Just verify URI is accessible (doesn't require full connection)
        except Exception as e:
            # MLflow check is optional - just log warning
            logger.warning(f"MLflow health check warning: {e}")

    status_str = (
        "healthy" if (model_loaded and pipeline_loaded and external_deps_healthy) else "degraded"
    )

    if dependency_errors:
        logger.warning(f"External dependency health check failures: {dependency_errors}")

    return HealthResponse(
        status=status_str,
        version="1.0.0",
        model_loaded=model_loaded,
        pipeline_loaded=pipeline_loaded,
    )


@app.get("/metadata", response_model=ModelMetadata, tags=["Model"])
async def get_metadata(request: Request) -> ModelMetadata:
    """Get model metadata.

    Args:
        request: FastAPI request object (for authentication).
    """
    # Verify API key if required
    await verify_api_key(request)

    if model_service is None or not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )

    metadata = model_service.get_metadata()
    return ModelMetadata(**metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    customer: CustomerData,
    request: Request,
    include_explanation: bool = True,
) -> PredictionResponse:
    """Predict churn for a single customer.

    Args:
        customer: Customer data for prediction.
        request: FastAPI request object (for authentication).
        include_explanation: Whether to include SHAP explanations (default: True).
    """
    # Verify API key if required
    await verify_api_key(request)

    start_time = time.perf_counter()
    PREDICTION_REQUESTS.labels(endpoint="/predict").inc()

    if model_service is None or not model_service.is_ready():
        PREDICTION_ERRORS.labels(endpoint="/predict", reason="model_not_loaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )

    try:
        # Convert Pydantic model to dict
        customer_dict = customer.model_dump()

        # Make prediction
        result = model_service.predict(customer_dict, include_explanation=include_explanation)

        return PredictionResponse(**result)
    except ValueError as e:
        PREDICTION_ERRORS.labels(endpoint="/predict", reason="validation").inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        PREDICTION_ERRORS.labels(endpoint="/predict", reason="internal_error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e
    finally:
        duration = time.perf_counter() - start_time
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(duration)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    http_request: Request,
) -> BatchPredictionResponse:
    """Predict churn for multiple customers.

    Args:
        request: Batch prediction request with customer data.
        http_request: FastAPI request object (for authentication).
    """
    # Verify API key if required
    await verify_api_key(http_request)

    start_time = time.perf_counter()
    PREDICTION_REQUESTS.labels(endpoint="/predict/batch").inc()

    if model_service is None or not model_service.is_ready():
        PREDICTION_ERRORS.labels(endpoint="/predict/batch", reason="model_not_loaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )

    # Validate batch size
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))
    if len(request.customers) > MAX_BATCH_SIZE:
        PREDICTION_ERRORS.labels(endpoint="/predict/batch", reason="batch_size_exceeded").inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(request.customers)} exceeds maximum allowed size of {MAX_BATCH_SIZE}",
        )

    try:
        predictions = []
        errors = []

        # Check if explanations should be included (default to False for batch to improve performance)
        include_explanation = os.getenv("BATCH_INCLUDE_EXPLANATIONS", "false").lower() == "true"

        for customer in request.customers:
            try:
                customer_dict = customer.model_dump()
                result = model_service.predict(
                    customer_dict, include_explanation=include_explanation
                )
                predictions.append(PredictionResponse(**result))
            except Exception as e:
                errors.append(f"Customer {customer.customerID}: {str(e)}")

        if errors and not predictions:
            # All predictions failed
            PREDICTION_ERRORS.labels(endpoint="/predict/batch", reason="all_failed").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All predictions failed: {'; '.join(errors)}",
            )

        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(request.customers),
        )
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS.labels(endpoint="/predict/batch", reason="internal_error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        ) from e
    finally:
        duration = time.perf_counter() - start_time
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(duration)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors (Pydantic validation)."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions.

    In production, sanitizes error messages to avoid exposing internal details.
    """
    # Log the full exception for debugging
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Determine if we're in production (you can customize this check)
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

    if is_production:
        # Sanitize error message in production
        error_detail = (
            "An internal server error occurred. Please contact support if the issue persists."
        )
    else:
        # In development, show the actual error
        error_detail = str(exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": error_detail,
        },
    )


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> PlainTextResponse:
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type="text/plain; version=0.0.4")
