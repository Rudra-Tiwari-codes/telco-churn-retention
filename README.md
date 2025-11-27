## Telco Churn Retention Platform

Work-in-progress churn prediction pipeline I’m building on the Telco Customer Churn dataset. The focus is on solid modeling, future real-time scoring, and turning insights into practical retention moves.

### Objectives
- Predict churn with ROC-AUC ≥ 0.85 alongside calibrated precision/recall.
- Expose reason codes and business insights for targeted retention.
- Aim to operate within an MLOps loop covering validation, training, deployment, and monitoring.

### Tech Stack
- Python 3.11, Poetry/uv for dependency management.
- Pandas, scikit-learn, XGBoost, LightGBM, Optuna, MLflow.
- FastAPI, Uvicorn, Docker, GitHub Actions.
- Great Expectations, feature abstractions modeled after Feast, basic Kafka/Redis hooks for streaming.

### Repository Layout
- `src/`: core application code (data, features, models, api, monitoring, pipelines).
- `notebooks/`: exploratory analysis and research.
- `configs/`: experiment and pipeline configuration files.
- `data/`: raw and processed artifacts (kept out of git; tracked locally only).
- `docs/`: architecture, roadmap, and business documentation.
- `scripts/`: CLI utilities for ingestion, validation, training, and eventual deployment.
- `tests/`: unit and integration tests.

### Getting Started
1. Create and activate a Python 3.11+ environment.
2. Install dependencies once `pyproject.toml`/`poetry.lock` are committed.
3. Download the Telco dataset into `data/raw/` (refer to `docs/roadmap.md` for phase instructions).
4. Run `make qa` (to be defined) for linting, typing, and tests before committing.

### Phase Tracking
| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repo scaffolding, docs, tooling | In progress |
| 1 | Data intake, validation, EDA | Pending |
| 2 | Feature engineering, feature store | Pending |
| 3 | Modeling, tuning, explainability | Pending |
| 4 | API, streaming pipeline | Pending |
| 5 | Monitoring, retraining automation | Pending |
| 6 | Business insights, packaging | Pending |

### Contribution Conventions
- Branch naming: `feat/<scope>`, `fix/<scope>`, `chore/<scope>`.
- Atomic commits with imperative mood messages.
- All work must pass lint, tests, and data-quality checks before PR.
- Tag issues/tasks inside commit bodies for traceability.

Refer to `docs/roadmap.md` for the detailed execution plan.

