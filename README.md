## Telco Churn Retention Platform

ML platoform for predicting customer churn in telecommunications. Built with MLOps,, this system provides end-to-end capabilities from data ingestion and validation through model training, API serving, monitoring, and automated retraining.

### Objectives
- Predict churn with ROC-AUC ≥ 0.85 alongside calibrated precision/recall.
- Expose SHAP-based reason codes and business insights for targeted retention.
- Operate within a complete MLOps loop covering validation, training, deployment, monitoring, and automated retraining.

### Key Features
- **Data Quality & Validation**: Automated schema and distribution checks using Great Expectations
- **Feature Engineering**: Reusable transformers with feature store abstractions for batch and streaming
- **Model Training**: XGBoost/LightGBM models with Optuna hyperparameter tuning and MLflow tracking
- **Model Explainability**: SHAP-based explanations for every prediction
- **REST API**: FastAPI service with health checks, metadata endpoints, single and batch predictions
- **Streaming Pipeline**: Kafka/Redis-based real-time scoring simulator
- **Drift Detection**: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests for data, prediction, and label drift
- **Performance Monitoring**: Automated tracking of model performance metrics and degradation detection
- **Alerting System**: Configurable alerts via Slack/email webhooks for critical issues
- **Monitoring Dashboards**: Automated visualization of drift metrics and performance trends
- **Automated Retraining**: End-to-end DAG pipeline orchestrating data ingestion → validation → feature engineering → training → evaluation → promotion
- **Docker Support**: Containerized deployment with docker-compose for API, Kafka, and Redis

### Tech Stack
- **Language**: Python 3.11+ with pip and setuptools for dependency management (via pyproject.toml)
- **Data Processing**: Pandas, NumPy, PyArrow, Great Expectations
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **MLOps**: MLflow (experiment tracking & model registry), Optuna (hyperparameter tuning)
- **Explainability**: SHAP
- **API Framework**: FastAPI, Uvicorn, Pydantic
- **Streaming**: Kafka, Redis
- **Monitoring**: scipy (statistical tests), matplotlib, seaborn (visualizations)
- **Containerization**: Docker, docker-compose
- **Dev Tools**: Ruff (linting), Black (formatting), MyPy (type checking), Pytest (testing)

### Configuration
- Environment variables: See `.env.example` and `docs/environment_variables.md`
- Model configuration: `configs/model_config.json`
- All sensitive values should be set via environment variables or secrets management

### Repository Layout
- `src/`: Core application code organized by functionality
  - `src/data/`: Data ingestion, validation, and EDA modules
  - `src/features/`: Feature engineering pipelines, transformers, and feature store
  - `src/models/`: Model training, evaluation, explainability modules
  - `src/api/`: FastAPI application with prediction endpoints and streaming support
  - `src/monitoring/`: Drift detection, performance monitoring, alerting, dashboards
  - `src/pipelines/`: Automated retraining DAG orchestration
  - `src/utils/`: Shared utilities including logging configuration
- `notebooks/`: Jupyter notebooks for each phase (setup, EDA, feature engineering, modeling, API serving, monitoring)
- `configs/`: Model and experiment configuration files (JSON)
- `data/`: Data storage (gitignored)
  - `data/raw/`: Raw input datasets
  - `data/processed/`: Processed features with timestamps
  - `data/feature_metadata/`: Feature store metadata and versioning
- `models/`: Trained model artifacts with timestamps
- `mlruns/`: MLflow experiment tracking data
- `reports/`: Generated reports (EDA, validation, modeling, monitoring)
- `docs/`: Architecture, roadmap, modeling strategy, and environment variable documentation
- `scripts/`: CLI utilities for running each phase
  - `run_phase1_data_intake.py`: Data ingestion and validation
  - `run_phase2_feature_engineering.py`: Feature engineering pipeline
  - `run_phase3_modeling.py`: Model training and evaluation
  - `run_phase4_api.py`: API server
  - `run_phase4_streaming.py`: Streaming pipeline simulator
  - `run_phase5_monitoring.py`: Monitoring workflow
  - `run_phase5_monitoring_auto.py`: Automated monitoring
  - `run_phase5_retraining.py`: Automated retraining pipeline
- `tests/`: Unit and integration tests
- `Dockerfile`: Container definition for API service
- `docker-compose.yml`: Multi-service orchestration (API, Kafka, Redis)

### Project Structure Details

#### Data Flow
1. **Raw Data** → `data/raw/` (Telco CSV dataset)
2. **Validation** → Great Expectations suite ensures data quality
3. **Feature Engineering** → Processed features saved to `data/processed/<timestamp>/`
4. **Model Training** → Trained models saved to `models/<timestamp>/`
5. **API Serving** → Loads latest production model for predictions
6. **Monitoring** → Tracks drift and performance in `reports/monitoring/`

#### Model Artifacts
Each model training run produces:
- `model_summary.json`: Model metadata and performance metrics
- `pipeline_summary.json`: Feature pipeline configuration
- `evaluation_metrics.json`: Detailed evaluation results
- Model files (XGBoost/LightGBM): Serialized model objects
- SHAP explainer artifacts for predictions

#### Monitoring Reports
Monitoring workflows generate:
- `drift_report.json`: Data, prediction, and label drift metrics
- `performance_report.json`: Model performance tracking
- `alerts.json`: Triggered alerts and notifications
- `drift_metrics.png`: Visual drift analysis
- `monitoring_dashboard.png`: Comprehensive dashboard visualization

### Testing

The project includes comprehensive tests in `tests/`:
- Unit tests for data ingestion and validation
- Feature engineering pipeline tests
- Model evaluation and explainability tests
- API endpoint tests
- Monitoring module tests
- Streaming pipeline tests

Run tests with:
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html  # With coverage
```


