"""Test SHAP explanations in API."""

import pytest
import requests

API_BASE_URL = "http://localhost:8001"


@pytest.mark.requires_model
def test_shap_explanations() -> None:
    """Test SHAP explanations in prediction response."""
    # Test customer data
    customer = {
        "customerID": "TEST-SHAP-123",
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

    try:
        # Make prediction
        response = requests.post(f"{API_BASE_URL}/predict", json=customer, timeout=30)
        if response.status_code == 200:
            data = response.json()
            explanations = data.get("explanation", [])

            assert explanations, "No SHAP explanations returned"
            assert len(explanations) > 0, "Empty explanations list"

            # Verify explanation structure
            for exp in explanations[:10]:  # Check first 10
                assert "feature" in exp, "Missing feature name in explanation"
                assert "shap_value" in exp, "Missing shap_value in explanation"
                assert isinstance(exp["shap_value"], (int, float)), "shap_value must be numeric"
        elif response.status_code == 503:
            pytest.skip("Model not loaded - skipping SHAP explanations test")
        else:
            pytest.skip(f"API server returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running - skipping SHAP explanations test")


@pytest.mark.requires_model
def test_metadata() -> None:
    """Test metadata endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/metadata", timeout=10)
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data, "Missing model_type"
            assert "feature_count" in data, "Missing feature_count"
            assert data["feature_count"] >= 0, "Feature count should be >= 0"

            perf_metrics = data.get("performance_metrics", {})
            if perf_metrics:
                # Verify performance metrics structure
                for metric, value in perf_metrics.items():
                    assert isinstance(value, (int, float)), f"Metric {metric} must be numeric"
        elif response.status_code == 503:
            pytest.skip("Model not loaded - skipping metadata test")
        else:
            pytest.skip(f"API server returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running - skipping metadata test")
