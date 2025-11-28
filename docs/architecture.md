## System Architecture

### Data Layer
- **Source**: Telco Customer Churn CSV plus optional behavioral augmentations.
- **Landing**: `data/raw/` (non-versioned), validated snapshots promoted to `data/processed/<date>/`.
- **Quality**: Great Expectations suites run before every training cycle; failures block downstream tasks.

### Feature Layer
- `src/features/` houses composable transformers assembled via sklearn pipelines.
- Feature metadata (owner, definition, data types) recorded in a lightweight registry (YAML/JSON) to ease migration to Feast or Tecton.
- Supports batch processing (offline training) and low-latency derivations for streaming inputs.

### Modeling Layer
- Training orchestrated from `src/models/` with configuration files in `configs/`.
- MLflow tracks params, metrics, artifacts, and model lineage; best models promoted to registry stages.
- Hyperparameter search handled by Optuna with early stopping and pruning.

### Serving Layer
- FastAPI service (`src/api/app.py`) loads the latest production bundle (preprocess + model).
- REST endpoints: `/health`, `/metadata`, `/predict`. Responses include probability, label, SHAP reason codes.
- Optional streaming connector consumes Kafka topics, enriches features, and writes scores to Redis/Postgres.

### Orchestration & Automation
- Pipelines coordinated by Airflow or Dagster once we reach Phase 5.
- Tasks: ingest → validate → feature build → train → evaluate → deploy → monitor → retrain.

### Observability & Monitoring
- Data drift (PSI), prediction drift, and label drift tracked via scheduled jobs.
- Model performance dashboards and alerting integrate with Slack/email webhooks.
- Application metrics (latency, throughput) exported via Prometheus-compatible endpoints.

### Security & Compliance
- Secrets managed via environment variables or Azure Key Vault/AWS Secrets Manager in deployment.
- PII handling documented; dataset does not include sensitive identifiers but policies will be outlined in `docs/operations.md`.
