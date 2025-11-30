"""Test SHAP explanations in API."""

import requests

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

print("=" * 80)
print("SHAP EXPLANATIONS TEST")
print("=" * 80)

# Make prediction
response = requests.post("http://localhost:8001/predict", json=customer, timeout=30)
if response.status_code == 200:
    data = response.json()
    print("\n[OK] Prediction successful")
    print(f"  Customer ID: {data['customerID']}")
    print(f"  Churn Probability: {data['churn_probability']:.4f}")
    print(f"  Churn Prediction: {data['churn_prediction']}")
    print(f"  Threshold: {data['threshold']}")

    explanations = data.get("explanation", [])
    print(f"\n  SHAP Explanations: {len(explanations)} features")

    if explanations:
        print("\n  Top 10 Feature Contributions:")
        for i, exp in enumerate(explanations[:10], 1):
            shap_val = exp["shap_value"]
            sign = "+" if shap_val >= 0 else ""
            print(
                f"    {i:2d}. {exp['feature']:25s}: {sign}{shap_val:8.4f} (value: {exp.get('value', 'N/A')})"
            )
        print("\n[OK] SHAP explanations are working correctly!")
    else:
        print("\n[WARN] No SHAP explanations returned")
else:
    print(f"[FAIL] Prediction failed: {response.status_code}")
    print(f"  Error: {response.text}")

# Test metadata
print("\n" + "=" * 80)
print("METADATA TEST")
print("=" * 80)
response = requests.get("http://localhost:8001/metadata", timeout=10)
if response.status_code == 200:
    data = response.json()
    print("\n[OK] Metadata retrieved")
    print(f"  Model Type: {data['model_type']}")
    print(f"  Model Version: {data['model_version']}")
    print(f"  Feature Count: {data['feature_count']}")
    print(f"  Threshold: {data['threshold']}")

    perf_metrics = data.get("performance_metrics", {})
    if perf_metrics:
        print("\n  Performance Metrics:")
        for metric, value in perf_metrics.items():
            print(f"    {metric}: {value:.4f}")
else:
    print(f"[FAIL] Metadata failed: {response.status_code}")
