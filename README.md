# Telco Churn Retention Platform

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

## Problem Statement

Telecom companies lose **$15-25B annually** to customer churn. Traditional rule-based retention strategies react too late—by the time a customer calls to cancel, they've already decided to leave. Predicting churn 30-60 days in advance enables proactive intervention with targeted retention offers.

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | IBM Telco Customer Dataset |
| Records | 7,043 customers |
| Features | 21 (demographics, services, account info) |
| Target | Churn (Yes/No) - 26.5% positive class |
| Split | 70% train, 15% validation, 15% test |

**Key Features:** Contract type, tenure, monthly charges, payment method, tech support, online security

## Methodology

### ML Pipeline (6 Phases)
1. **Data Ingestion** — Automated validation with Great Expectations
2. **Feature Engineering** — 40+ engineered features with feature store
3. **Model Training** — XGBoost/LightGBM with Optuna hyperparameter tuning
4. **API Serving** — FastAPI with single/batch prediction endpoints
5. **Monitoring** — Drift detection (PSI, KS tests) + performance tracking
6. **Business Intelligence** — ROI calculator for retention campaigns

### MLOps Stack
- **Experiment Tracking**: MLflow
- **Feature Store**: Custom abstractions for batch/streaming
- **Model Explainability**: SHAP for every prediction
- **Streaming**: Kafka + Redis real-time scoring
- **Alerting**: Slack/email webhooks for model degradation

## Results

| Metric | XGBoost | LightGBM |
|--------|---------|----------|
| **ROC-AUC** | 0.87 | 0.86 |
| Precision | 0.72 | 0.70 |
| Recall | 0.68 | 0.71 |
| F1-Score | 0.70 | 0.70 |

### Business Impact
- **~$2.1M annual savings** from early intervention (estimated on 10K customer base)
- 30-day advance warning enables proactive retention offers
- SHAP explanations identify top churn drivers per customer

## Usage

```bash
# Setup
pip install -e .

# Run full pipeline
python scripts/run_phase1_data_intake.py      # Data validation
python scripts/run_phase2_feature_engineering.py
python scripts/run_phase3_modeling.py
python scripts/run_phase4_api.py              # Start API server

# Docker deployment
docker-compose up -d
```

### API Example
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "tenure": 12,
        "monthly_charges": 85.50,
        "contract": "Month-to-month",
        "tech_support": "No"
    }
)
# {"churn_probability": 0.73, "risk_level": "High", "top_factors": [...]}
```

## Project Structure

```
telco-churn-retention/
├── src/
│   ├── data/           # Ingestion, validation, EDA
│   ├── features/       # Feature engineering, feature store
│   ├── models/         # Training, evaluation, explainability
│   ├── api/            # FastAPI endpoints + streaming
│   ├── monitoring/     # Drift detection, alerting, dashboards
│   └── pipelines/      # Automated retraining orchestration
├── notebooks/          # 6 phase notebooks
├── configs/            # Model configuration
├── scripts/            # CLI utilities
└── tests/              # Comprehensive test suite
```

## Future Improvements

1. **A/B Testing Framework** — Measure actual retention campaign effectiveness
2. **Customer Lifetime Value Integration** — Prioritize high-value churners
3. **Real-time Feature Engineering** — Sub-second feature computation for streaming

---

**License**: MIT

[Rudra Tiwari](https://github.com/Rudra-Tiwari-codes)
