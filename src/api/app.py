"""
FastAPI application for churn prediction API.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# Configure logging
setup_logging()
logger = get_logger(__name__)

# Global model service
model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    global model_service

    # Startup: Load model and pipeline
    logger.info("Starting up API service...")
    model_dir = os.getenv("MODEL_DIR", "models")
    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

    try:
        model_service = ModelService(
            model_dir=Path(model_dir),
            threshold=threshold,
        )
        logger.info(f"Model loaded successfully from {model_dir}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
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

# CORS middleware - configure via environment variable
_cors_origins = os.getenv("CORS_ORIGINS", "*")
if _cors_origins == "*":
    # Allow all origins (development only)
    cors_origins_list = ["*"]
else:
    # Parse comma-separated list of origins
    cors_origins_list = [origin.strip() for origin in _cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    model_loaded = (
        model_service is not None and model_service.is_ready() if model_service else False
    )
    pipeline_loaded = (
        model_service is not None and model_service.pipeline is not None if model_service else False
    )

    status_str = "healthy" if (model_loaded and pipeline_loaded) else "degraded"

    return HealthResponse(
        status=status_str,
        version="1.0.0",
        model_loaded=model_loaded,
        pipeline_loaded=pipeline_loaded,
    )


@app.get("/metadata", response_model=ModelMetadata, tags=["Model"])
async def get_metadata() -> ModelMetadata:
    """Get model metadata."""
    if model_service is None or not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )

    metadata = model_service.get_metadata()
    return ModelMetadata(**metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerData) -> PredictionResponse:
    """Predict churn for a single customer."""
    if model_service is None or not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )

    try:
        # Convert Pydantic model to dict
        customer_dict = customer.model_dump()

        # Make prediction
        result = model_service.predict(customer_dict, include_explanation=True)

        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict churn for multiple customers."""
    if model_service is None or not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )

    try:
        predictions = []
        errors = []

        for customer in request.customers:
            try:
                customer_dict = customer.model_dump()
                result = model_service.predict(customer_dict, include_explanation=True)
                predictions.append(PredictionResponse(**result))
            except Exception as e:
                errors.append(f"Customer {customer.customerID}: {str(e)}")

        if errors and not predictions:
            # All predictions failed
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        ) from e


@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
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
