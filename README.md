## Telco Churn Retention Platform

Work-in-progress churn prediction pipeline I’m building on the Telco Customer Churn dataset. The focus is on solid modeling, future real-time scoring, and turning insights into practical retention moves.

### Objectives
- Predict churn with ROC-AUC ≥ 0.85 alongside calibrated precision/recall.
- Expose reason codes and business insights for targeted retention.
- Aim to operate within an MLOps loop covering validation, training, deployment, and monitoring.

### Tech Stack
- Python 3.11, pip and setuptools for dependency management (via pyproject.toml).
- Pandas, scikit-learn, XGBoost, LightGBM, Optuna, MLflow.
- FastAPI, Uvicorn, Docker, GitHub Actions.
- Great Expectations, feature abstractions modeled after Feast, basic Kafka/Redis hooks for streaming.

### Configuration
- Environment variables: See `.env.example` and `docs/environment_variables.md`
- Model configuration: `configs/model_config.json`
- All sensitive values should be set via environment variables or secrets management

### Repository Layout
- `src/`: core application code (data, features, models, api, monitoring, pipelines).
- `notebooks/`: exploratory analysis and research.
- `configs/`: experiment and pipeline configuration files.
- `data/`: raw and processed artifacts (kept out of git; tracked locally only).
- `docs/`: architecture, roadmap, and business documentation.
- `scripts/`: CLI utilities for ingestion, validation, training, and eventual deployment.
- `tests/`: unit and integration tests.

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
   # Or: python scripts/run_phase4_api.py
   ```
   API will be available at http://localhost:8000
   - Swagger UI: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

8. **Run Monitoring** (after completing Phases 1-4):
   ```bash
   make phase5-monitoring
   # Or: python scripts/run_phase5_monitoring.py --reference-data <path> --current-data <path>
   ```

9. **Run Retraining Pipeline** (after completing Phases 1-3):
   ```bash
   make phase5-retraining
   # Or: python scripts/run_phase5_retraining.py
   ```

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

### Phase Tracking
| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repo scaffolding, docs, tooling |  Completed (28 Nov 2025) |
| 1 | Data intake, validation, EDA |  Completed (28 Nov 2025) |
| 2 | Feature engineering, feature store | Completed (28 Nov 2025) |
| 3 | Modeling, tuning, explainability | Completed (29 Nov 2025) |
| 4 | API, streaming pipeline | Completed (29 Nov 2025) |
| 5 | Monitoring, retraining automation | Completed (29 Nov 2025) |
| 6 | Business insights, packaging | Pending |

### Contribution Conventions
- Branch naming: `feat/<scope>`, `fix/<scope>`, `chore/<scope>`.
- Atomic commits with imperative mood messages.
- All work must pass lint, tests, and data-quality checks before PR.
- Tag issues/tasks inside commit bodies for traceability.

Refer to `docs/roadmap.md` for the detailed execution plan.


