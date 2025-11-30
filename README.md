## Telco Churn Retention Platform

A production-ready machine learning platform for predicting customer churn in telecommunications. Built with MLOps best practices, this system provides end-to-end capabilities from data ingestion and validation through model training, API serving, monitoring, and automated retraining.

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

### Getting Started

#### Windows Setup

1. **Create Virtual Environment** (choose one method):
   - **Recommended**: Run `.\setup_venv.ps1` in PowerShell
   - **Alternative**: `python -m venv venv`

2. **Activate Virtual Environment** (choose one method):
   - **PowerShell**: `.\activate_venv.ps1` or `.\venv\Scripts\Activate.ps1`
   - **Command Prompt**: `activate_venv.bat` or `venv\Scripts\activate.bat`
   
   **If you get execution policy errors**, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e .
   pip install -e ".[eda,dev]"  # For development with notebooks
   ```

4. **Configure Environment Variables** (optional):
   ```bash
   # Copy the example environment file
   cp .env.example .env
   # Edit .env with your configuration
   ```
   See `docs/environment_variables.md` for detailed documentation.

5. Download the Telco dataset into `data/raw/` (refer to `docs/roadmap.md` for phase instructions).

6. Run `make qa` for linting, typing, and tests before committing.

7. **Start API Service** (after completing Phases 1-3):
   ```bash
   make phase4
   # Or with custom port: python scripts/run_phase4_api.py --port 8001
   ```
   
   API will be available at http://localhost:8000 (default)
   - **Swagger UI**: http://localhost:8000/docs
   - **ReDoc**: http://localhost:8000/redoc
   - **Health Check**: http://localhost:8000/health
   - **Root**: http://localhost:8000/
   
   **Note**: Notebooks may use port 8001 by default. Adjust the port using the `--port` flag if needed.
   
   **API Endpoints:**
   - `GET /health`: Service health check and model loading status
   - `GET /metadata`: Model metadata (version, features, performance metrics)
   - `POST /predict`: Single customer churn prediction with SHAP explanations
   - `POST /predict/batch`: Batch predictions for multiple customers

8. **Run Streaming Pipeline** (after completing Phases 1-4):
   ```bash
   make phase4-streaming
   # Or: python scripts/run_phase4_streaming.py --simulate
   ```
   Simulates real-time scoring via Kafka/Redis for demonstration purposes.

9. **Run Monitoring** (after completing Phases 1-4):
   ```bash
   make phase5-monitoring
   # Or: python scripts/run_phase5_monitoring_auto.py
   ```
   
   This will:
   - Detect data drift using PSI and KS tests
   - Monitor prediction and label drift
   - Track performance degradation
   - Generate monitoring dashboards
   - Create alerts for critical issues
   
   Reports are saved to `reports/monitoring/`.

10. **Run Retraining Pipeline** (after completing Phases 1-3):
    ```bash
    make phase5-retraining
    # Or: python scripts/run_phase5_retraining.py
    ```
    
    Orchestrates the complete retraining workflow:
    - Data ingestion and validation
    - Feature engineering
    - Model training with hyperparameter tuning
    - Model evaluation and comparison
    - Automatic promotion if performance improves

#### Linux/macOS Setup

1. Create and activate a Python 3.11+ environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   pip install -e ".[eda,dev]"  # For development with notebooks
   ```

3. **Configure Environment Variables** (optional):
   ```bash
   # Copy the example environment file
   cp .env.example .env
   # Edit .env with your configuration
   ```
   See `docs/environment_variables.md` for detailed documentation.

4. Download the Telco dataset into `data/raw/` (refer to `docs/roadmap.md` for phase instructions).

5. Run `make qa` for linting, typing, and tests before committing.

### Docker Deployment

The platform can be deployed using Docker for containerized environments.

#### Build and Run API Service
```bash
# Build the Docker image
docker build -t telco-churn-api .

# Run the API service
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  -e MODEL_DIR=/app/models \
  -e PREDICTION_THRESHOLD=0.5 \
  telco-churn-api
```

#### Run Full Stack with Docker Compose
```bash
# Start all services (API, Kafka, Redis, Zookeeper)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

The docker-compose setup includes:
- **API Service**: FastAPI application on port 8000
- **Redis**: For streaming pipeline (port 6379)
- **Kafka**: Message broker (port 9092)
- **Zookeeper**: Kafka coordination (port 2181)

### Usage Examples

#### Phase-by-Phase Execution
Execute each phase in sequence using Make commands or direct Python scripts:

```bash
# Phase 1: Data intake and validation
make phase1

# Phase 2: Feature engineering
make phase2

# Phase 3: Model training
make phase3

# Phase 4: Start API service
make phase4

# Phase 5: Run monitoring
make phase5-monitoring

# Phase 5: Run retraining pipeline
make phase5-retraining
```

#### Using Jupyter Notebooks
Each phase has a corresponding notebook in `notebooks/`:
- `phase0_setup.ipynb`: Environment setup and configuration
- `phase1_eda.ipynb`: Exploratory data analysis
- `phase2_feature_engineering.ipynb`: Feature engineering and feature store
- `phase3_modeling.ipynb`: Model training, tuning, and evaluation
- `phase4_api_serving.ipynb`: API testing and validation (includes automatic server startup)
- `phase5_monitoring.ipynb`: Monitoring workflows and dashboards

#### API Usage Examples

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Model Metadata:**
```bash
curl http://localhost:8000/metadata
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      { "customerID": "7590-VHVEG", ... },
      { "customerID": "5575-GNVDE", ... }
    ]
  }'
```

### Phase Tracking

| Phase | Scope | Components | Status |
| --- | --- | --- | --- |
| 0 | Repo scaffolding, docs, tooling | Project structure, pyproject.toml, Makefile, pre-commit hooks, documentation framework | ✅ Completed (28 Nov 2025) |
| 1 | Data intake, validation, EDA | Data ingestion (`src/data/ingestion.py`), Great Expectations validation (`src/data/validation.py`), EDA notebook with profiling | ✅ Completed (28 Nov 2025) |
| 2 | Feature engineering, feature store | Feature transformers (`src/features/transformers.py`), feature pipeline (`src/features/pipeline.py`), feature store with metadata (`src/features/store.py`) | ✅ Completed (28 Nov 2025) |
| 3 | Modeling, tuning, explainability | Baseline models, XGBoost/LightGBM training, Optuna hyperparameter tuning, MLflow tracking, SHAP explainability (`src/models/explainability.py`) | ✅ Completed (29 Nov 2025) |
| 4 | API, streaming pipeline | FastAPI application (`src/api/app.py`), model service (`src/api/service.py`), batch/single predictions, streaming simulator with Kafka/Redis (`src/api/streaming.py`) | ✅ Completed (29 Nov 2025) |
| 5 | Monitoring, retraining automation | Drift detection (`src/monitoring/drift.py`), performance monitoring (`src/monitoring/performance.py`), alerting (`src/monitoring/alerts.py`), dashboards (`src/monitoring/dashboard.py`), retraining DAG (`src/pipelines/retraining_dag.py`) | ✅ Completed (29 Nov 2025) |
| 6 | Business insights, packaging | Retention playbooks, ROI estimates, executive presentations | 🔄 Pending |

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

### Development Workflow

#### Quality Assurance
```bash
# Run all QA checks (lint, format, type-check, tests)
make qa

# Individual checks
make lint          # Run Ruff linter
make format        # Format code with Black
make format-check  # Check formatting without changes
make type-check    # Run MyPy type checking
make test          # Run pytest
make test-cov      # Run tests with coverage report
```

#### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
make pre-commit
```

### Configuration Files

- `pyproject.toml`: Project metadata, dependencies, tool configurations (Ruff, Black, MyPy, Pytest)
- `configs/model_config.json`: Model training configurations
- `.env.example`: Template for environment variables
- `Makefile`: Common development and execution commands
- `Dockerfile`: Container definition for API service
- `docker-compose.yml`: Multi-service orchestration

### Documentation

- `docs/architecture.md`: System architecture and design decisions
- `docs/roadmap.md`: Detailed phase-by-phase execution plan
- `docs/modeling.md`: Modeling strategy and approach
- `docs/environment_variables.md`: Complete environment variable reference

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

### Contribution Conventions

- **Branch naming**: `feat/<scope>`, `fix/<scope>`, `chore/<scope>`
- **Commits**: Atomic commits with imperative mood messages
- **Quality gates**: All work must pass lint, tests, and data-quality checks before PR
- **Traceability**: Tag issues/tasks inside commit bodies

Refer to `docs/roadmap.md` for the detailed execution plan and `docs/architecture.md` for system design details.


