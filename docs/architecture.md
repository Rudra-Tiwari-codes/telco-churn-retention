## System Architecture

- **Data**: Telco churn CSV lands in `data/raw/`; validated snapshots move to `data/processed/<date>/`. Great Expectations runs before training.
- **Features**: sklearn pipelines in `src/features/` generate both batch datasets and streaming-friendly transforms; metadata recorded in lightweight YAML for future feature-store migration.
- **Modeling**: `src/models/` drives experiments configured via `configs/`. Optuna handles tuning, MLflow logs runs and artifacts.
- **Serving**: FastAPI app (`src/api/app.py`) will load whichever preprocessing + model bundle is marked current; planned endpoints `/health`, `/metadata`, `/predict`, plus optional Kafka→Redis stream.
- **Ops**: Airflow/Dagster DAG (Phase 5) will orchestrate ingest→validate→train→evaluate→promote→monitor. Monitoring jobs compute PSI/KS and export metrics for dashboards/alerts.
- **Security**: Secrets supplied via env vars or cloud secret stores. Dataset lacks direct PII, but access controls and audit notes will be captured as the project grows.

