"""Comprehensive FAANG-level testing for Phase 4 API."""

import time
from pathlib import Path

import pandas as pd
import requests

API_BASE_URL = "http://localhost:8001"


def test_health():
    """Test health endpoint."""
    print("=" * 80)
    print("TEST 1: Health Check")
    print("=" * 80)
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy", f"Expected healthy, got {data['status']}"
    assert data["model_loaded"], "Model should be loaded"
    assert data["pipeline_loaded"], "Pipeline should be loaded"
    print("[OK] Health check passed")
    print(f"  Status: {data['status']}")
    print(f"  Model loaded: {data['model_loaded']}")
    print(f"  Pipeline loaded: {data['pipeline_loaded']}")
    return True


def test_metadata():
    """Test metadata endpoint."""
    print("\n" + "=" * 80)
    print("TEST 2: Metadata Endpoint")
    print("=" * 80)
    response = requests.get(f"{API_BASE_URL}/metadata", timeout=10)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "model_type" in data, "Missing model_type"
    assert "feature_count" in data, "Missing feature_count"
    assert data["feature_count"] > 0, "Feature count should be > 0"
    print("[OK] Metadata endpoint passed")
    print(f"  Model Type: {data['model_type']}")
    print(f"  Model Version: {data.get('model_version', 'N/A')}")
    print(f"  Feature Count: {data['feature_count']}")
    print(f"  Threshold: {data.get('threshold', 0.5)}")
    return True


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n" + "=" * 80)
    print("TEST 3: Single Prediction")
    print("=" * 80)

    # Load a real customer from data
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
    else:
        # Use sample data
        customer = {
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

    start_time = time.time()
    response = requests.post(f"{API_BASE_URL}/predict", json=customer, timeout=30)
    elapsed = time.time() - start_time

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "churn_probability" in data, "Missing churn_probability"
    assert "churn_prediction" in data, "Missing churn_prediction"
    assert 0 <= data["churn_probability"] <= 1, "Probability should be between 0 and 1"
    assert isinstance(data["churn_prediction"], bool), "Prediction should be boolean"

    print("[OK] Single prediction passed")
    print(f"  Customer ID: {data['customerID']}")
    print(f"  Churn Probability: {data['churn_probability']:.4f}")
    print(f"  Churn Prediction: {data['churn_prediction']}")
    print(f"  Latency: {elapsed*1000:.2f} ms")
    print(f"  Explanations: {len(data.get('explanation', []))} features")

    # FAANG latency check
    if elapsed < 0.1:
        print("  [OK] Latency < 100ms (FAANG standard)")
    else:
        print(f"  [WARN] Latency >= 100ms ({elapsed*1000:.2f} ms)")

    return True


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n" + "=" * 80)
    print("TEST 4: Batch Prediction")
    print("=" * 80)

    # Create batch of 3 customers
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
    response = requests.post(
        f"{API_BASE_URL}/predict/batch", json={"customers": customers}, timeout=60
    )
    elapsed = time.time() - start_time

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "predictions" in data, "Missing predictions"
    assert "total_customers" in data, "Missing total_customers"
    assert len(data["predictions"]) == 3, f"Expected 3 predictions, got {len(data['predictions'])}"
    assert data["total_customers"] == 3, f"Expected 3 total, got {data['total_customers']}"

    print("[OK] Batch prediction passed")
    print(f"  Total customers: {data['total_customers']}")
    print(f"  Predictions returned: {len(data['predictions'])}")
    print(f"  Latency: {elapsed*1000:.2f} ms")
    print(f"  Avg latency per customer: {elapsed*1000/3:.2f} ms")

    return True


def test_validation():
    """Test input validation."""
    print("\n" + "=" * 80)
    print("TEST 5: Input Validation")
    print("=" * 80)

    # Test invalid tenure
    invalid_data = {"customerID": "test", "tenure": -1}
    response = requests.post(f"{API_BASE_URL}/predict", json=invalid_data, timeout=10)
    assert (
        response.status_code == 422
    ), f"Expected 422 for validation error, got {response.status_code}"
    print("[OK] Invalid tenure rejected")

    # Test missing fields
    incomplete_data = {"customerID": "test"}
    response = requests.post(f"{API_BASE_URL}/predict", json=incomplete_data, timeout=10)
    assert (
        response.status_code == 422
    ), f"Expected 422 for missing fields, got {response.status_code}"
    print("[OK] Missing fields rejected")

    # Test empty batch
    response = requests.post(f"{API_BASE_URL}/predict/batch", json={"customers": []}, timeout=10)
    assert response.status_code == 422, f"Expected 422 for empty batch, got {response.status_code}"
    print("[OK] Empty batch rejected")

    return True


def test_performance():
    """Test performance with multiple requests."""
    print("\n" + "=" * 80)
    print("TEST 6: Performance Testing")
    print("=" * 80)

    customer = {
        "customerID": "PERF-TEST",
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

    num_requests = 10
    latencies = []
    successes = 0

    print(f"  Running {num_requests} requests...")
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/predict", json=customer, timeout=30)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                latencies.append(elapsed * 1000)  # Convert to ms
                successes += 1
        except Exception as e:
            print(f"  [WARN] Request {i+1} failed: {e}")

    if latencies:
        import numpy as np

        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        print("[OK] Performance test completed")
        print(f"  Successful requests: {successes}/{num_requests}")
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  P50 latency: {p50:.2f} ms")
        print(f"  P95 latency: {p95:.2f} ms")
        print(f"  P99 latency: {p99:.2f} ms")

        # FAANG standards
        if avg_latency < 100:
            print("  [OK] Average latency < 100ms (FAANG standard)")
        else:
            print("  [WARN] Average latency >= 100ms")

        if p95 < 200:
            print("  [OK] P95 latency < 200ms (FAANG standard)")
        else:
            print("  [WARN] P95 latency >= 200ms")
    else:
        print("  [FAIL] No successful requests")
        return False

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PHASE 4 COMPREHENSIVE API TESTING")
    print("=" * 80)

    tests = [
        ("Health Check", test_health),
        ("Metadata", test_metadata),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Input Validation", test_validation),
        ("Performance", test_performance),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} FAILED: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 4 is working correctly.")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Please review.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
