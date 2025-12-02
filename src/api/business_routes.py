"""
Business intelligence API endpoints for Phase 6.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.api.models import CustomerData
from src.api.service import ModelService
from src.business.cohorts import CohortAnalyzer
from src.business.kpis import KPITracker
from src.business.playbooks import RetentionPlaybookGenerator
from src.business.roi import ROICalculator

router = APIRouter(prefix="/business", tags=["Business Intelligence"])


class CohortAnalysisRequest(BaseModel):
    """Request for cohort analysis."""

    customers: list[CustomerData]
    churn_threshold: float = 0.5


class PlaybookRequest(BaseModel):
    """Request for retention playbook."""

    customer: CustomerData
    churn_probability: float | None = None


class ROIAnalysisRequest(BaseModel):
    """Request for ROI analysis."""

    customers: list[CustomerData]
    intervention_cost_per_customer: float = 50.0
    churn_threshold: float = 0.5


def get_model_service() -> ModelService:
    """Dependency to get model service."""
    import src.api.app as app_module

    global_model_service = app_module.model_service
    if global_model_service is None or not global_model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return global_model_service


@router.post("/cohorts")
async def analyze_cohorts(
    request: CohortAnalysisRequest,
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Analyze customers and create cohorts.

    Args:
        request: Cohort analysis request.
        model_service: Model service instance (injected by dependency).

    Returns:
        Cohort analysis results.
    """

    try:
        # Convert customers to DataFrame
        customers_df = pd.DataFrame([c.model_dump() for c in request.customers])

        # Get predictions
        predictions = []
        for customer in request.customers:
            customer_dict = customer.model_dump()
            result = model_service.predict(customer_dict, include_explanation=False)
            predictions.append(
                {
                    "customerID": customer_dict["customerID"],
                    "churn_probability": result["churn_probability"],
                }
            )

        predictions_df = pd.DataFrame(predictions)

        # Create cohorts
        analyzer = CohortAnalyzer()
        cohorts = analyzer.create_cohorts(customers_df, predictions_df)

        return {
            "cohorts": [cohort.to_dict() for cohort in cohorts],
            "total_cohorts": len(cohorts),
            "total_customers": len(customers_df),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cohort analysis failed: {str(e)}",
        ) from e


@router.post("/playbook")
async def generate_playbook(
    request: PlaybookRequest,
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Generate retention playbook for a customer.

    Args:
        request: Playbook request.
        model_service: Model service instance.

    Returns:
        Retention playbook.
    """
    if model_service is None or not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        customer_dict = request.customer.model_dump()

        # Get prediction if not provided
        if request.churn_probability is None:
            result = model_service.predict(customer_dict, include_explanation=True)
            churn_probability = result["churn_probability"]
            explanations = result.get("explanation", [])
        else:
            churn_probability = request.churn_probability
            result = model_service.predict(customer_dict, include_explanation=True)
            explanations = result.get("explanation", [])

        # Generate playbook
        generator = RetentionPlaybookGenerator()
        playbook = generator.generate_playbook(
            customer_data=customer_dict,
            churn_probability=churn_probability,
            monthly_revenue=customer_dict.get("MonthlyCharges", 0),
            tenure=customer_dict.get("tenure", 0),
            explanations=explanations,
        )

        return playbook.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Playbook generation failed: {str(e)}",
        ) from e


@router.post("/roi")
async def calculate_roi(
    request: ROIAnalysisRequest,
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Calculate ROI for retention initiatives.

    Args:
        request: ROI analysis request.
        model_service: Model service instance.

    Returns:
        ROI metrics.
    """
    if model_service is None or not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Convert customers to DataFrame
        customers_df = pd.DataFrame([c.model_dump() for c in request.customers])

        # Get predictions
        predictions = []
        for customer in request.customers:
            customer_dict = customer.model_dump()
            result = model_service.predict(customer_dict, include_explanation=False)
            predictions.append(
                {
                    "customerID": customer_dict["customerID"],
                    "churn_probability": result["churn_probability"],
                }
            )

        predictions_df = pd.DataFrame(predictions)

        # Calculate ROI
        calculator = ROICalculator()
        roi_metrics = calculator.calculate_roi(
            customers=customers_df,
            predictions=predictions_df,
            intervention_cost_per_customer=request.intervention_cost_per_customer,
            churn_threshold=request.churn_threshold,
        )

        return roi_metrics.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ROI calculation failed: {str(e)}",
        ) from e


@router.post("/kpis")
async def calculate_kpis(
    customers: list[CustomerData],
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Calculate business KPIs.

    Args:
        customers: List of customers.
        model_service: Model service instance.

    Returns:
        KPI metrics.
    """
    if model_service is None or not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Convert customers to DataFrame
        customers_df = pd.DataFrame([c.model_dump() for c in customers])

        # Get predictions
        predictions = []
        for customer in customers:
            customer_dict = customer.model_dump()
            result = model_service.predict(customer_dict, include_explanation=False)
            predictions.append(
                {
                    "customerID": customer_dict["customerID"],
                    "churn_probability": result["churn_probability"],
                }
            )

        predictions_df = pd.DataFrame(predictions)

        # Get model metrics
        metadata = model_service.get_metadata()
        model_metrics = metadata.get("performance_metrics", {})

        # Calculate KPIs
        tracker = KPITracker()
        kpi_metrics = tracker.calculate_kpis(
            customers=customers_df,
            predictions=predictions_df,
            model_metrics=model_metrics,
        )

        return kpi_metrics.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KPI calculation failed: {str(e)}",
        ) from e
