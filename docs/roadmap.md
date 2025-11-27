## Roadmap

| Phase | Scope | Key Deliverables |
| --- | --- | --- |
| 0. Foundation | Repo layout, tooling, docs skeleton | pyproject/env setup, lint/test harness, CI placeholder |
| 1. Data Intake | Dataset ingestion + validation | `data/raw/` snapshot, GE checks, EDA summary |
| 2. Features | Core transformers + processed exports | sklearn pipelines, provenance logs, sample processed set |
| 3. Modeling | Baseline + boosted models | ≥0.85 ROC-AUC candidate, MLflow tracking, SHAP artifacts |
| 4. Serving | API + streaming draft | FastAPI skeleton, Kafka/Redis simulator, Docker assets |
| 5. Monitoring | Drift + retraining loop | PSI/KS jobs, Airflow/Dagster DAG, alert hooks |
| 6. Business Readout | Insights + comms | Cohort recommendations, KPI sheet, exec-ready summary |

Progress notes and branch references will continue to live here as work advances.

