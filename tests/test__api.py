"""Comprehensive FAANG-level testing for Phase 4 API."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

# Create test client
client = TestClient(app)


def _check_model_loaded() -> bool:
    """Check if model is loaded."""
    health_response = client.get("/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        return health_data.get("model_loaded", False) and health_data.get("pipeline_loaded", False)
    return False


def test_health() -> None:
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "status" in data, "Missing status field"
    assert data["status"] in ["healthy", "degraded"], f"Unexpected status: {data['status']}"
    assert "model_loaded" in data, "Missing model_loaded field"
    assert "pipeline_loaded" in data, "Missing pipeline_loaded field"


@pytest.mark.requires_model
def test_metadata() -> None:
    """Test metadata endpoint."""
    if not _check_model_loaded():
        pytest.skip("Model not loaded - skipping metadata test")

    response = client.get("/metadata")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "model_type" in data, "Missing model_type"
    assert "feature_count" in data, "Missing feature_count"
    assert data["feature_count"] >= 0, "Feature count should be >= 0"


def _get_sample_customer() -> dict[str, Any]:
    """Get a sample customer for testing."""
    raw_path = Path("data/raw/telco_data_28_11_2025.csv")
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        df.columns = [c.strip() for c in df.columns]
        customer = df.iloc[0].to_dict()
        # Handle TotalCharges
        if pd.isna(customer.get("TotalCharges")):
            customer["TotalCharges"] = None
        else:
            customer["TotalCharges"] = float(customer["TotalCharges"])
        return customer
    else:
        # Use sample data
        return {
            "customerID": "1234-ABCD",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.5,
            "TotalCharges": 845.0,
        }


@pytest.mark.requires_model
def test_single_prediction() -> None:
    """Test single prediction endpoint."""
    if not _check_model_loaded():
        pytest.skip("Model not loaded - skipping prediction test")

    customer = _get_sample_customer()

    start_time = time.time()
    response = client.post("/predict", json=customer)
    elapsed = time.time() - start_time

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "churn_probability" in data, "Missing churn_probability"
    assert "churn_prediction" in data, "Missing churn_prediction"
    assert 0 <= data["churn_probability"] <= 1, "Probability should be between 0 and 1"
    assert isinstance(data["churn_prediction"], bool), "Prediction should be boolean"

    # FAANG latency check (non-blocking assertion for CI)
    assert elapsed < 1.0, f"Latency too high: {elapsed * 1000:.2f} ms"


@pytest.mark.requires_model
def test_batch_prediction() -> None:
    """Test batch prediction endpoint."""
    if not _check_model_loaded():
        pytest.skip("Model not loaded - skipping batch prediction test")

    customers = [
        {
            "customerID": f"TEST-{i}",
            "gender": "Male" if i % 2 == 0 else "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes" if i % 2 == 0 else "No",
            "Dependents": "No",
            "tenure": 12 + i * 5,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.5 + i * 10,
            "TotalCharges": 845.0 + i * 100,
        }
        for i in range(3)
    ]

    start_time = time.time()
    response = client.post("/predict/batch", json={"customers": customers})
    time.time() - start_time

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "predictions" in data, "Missing predictions"
    assert "total_customers" in data, "Missing total_customers"
    assert len(data["predictions"]) == 3, f"Expected 3 predictions, got {len(data['predictions'])}"
    assert data["total_customers"] == 3, f"Expected 3 total, got {data['total_customers']}"


def test_validation() -> None:
    """Test input validation."""
    # Test invalid tenure
    invalid_data = {"customerID": "test", "tenure": -1}
    response = client.post("/predict", json=invalid_data)
    assert (
        response.status_code == 422
    ), f"Expected 422 for validation error, got {response.status_code}"

    # Test missing fields
    incomplete_data = {"customerID": "test"}
    response = client.post("/predict", json=incomplete_data)
    assert (
        response.status_code == 422
    ), f"Expected 422 for missing fields, got {response.status_code}"

    # Test empty batch
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422, f"Expected 422 for empty batch, got {response.status_code}"


@pytest.mark.requires_model
def test_performance() -> None:
    """Test performance with multiple requests."""
    if not _check_model_loaded():
        pytest.skip("Model not loaded - skipping performance test")

    customer = _get_sample_customer()

    num_requests = 10
    latencies = []
    successes = 0

    for _i in range(num_requests):
        try:
            start_time = time.time()
            response = client.post("/predict", json=customer)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                latencies.append(elapsed * 1000)  # Convert to ms
                successes += 1
        except Exception:
            # Log but continue
            pass

    assert successes > 0, "No successful requests"
    assert len(latencies) == successes, "Latency count mismatch"

    avg_latency = np.mean(latencies)
    np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    np.percentile(latencies, 99)

    # Non-blocking assertions for CI (warnings rather than failures)
    # These can be made strict if needed
    assert avg_latency < 5000, f"Average latency too high: {avg_latency:.2f} ms"
    assert p95 < 10000, f"P95 latency too high: {p95:.2f} ms"


def test_health_without_model() -> None:
    """Test that health endpoint works even without model loaded."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "status" in data, "Missing status field"
    # Status can be "healthy" or "degraded" depending on model availability
    assert data["status"] in ["healthy", "degraded"], f"Unexpected status: {data['status']}"


def test_metadata_without_model() -> None:
    """Test that metadata endpoint returns 503 when model not loaded."""
    response = client.get("/metadata")
    # If model not loaded, should return 503
    if response.status_code == 503:
        # This is expected behavior when model is not available
        # HTTP exception handler converts 'detail' to 'error' in response
        assert "Model not loaded" in response.json().get("error", ""), "Missing error detail"
    else:
        # If model is loaded, should return 200
        assert response.status_code == 200, f"Expected 200 or 503, got {response.status_code}"
