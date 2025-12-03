## Modeling Strategy

### Objectives
- Deliver calibrated churn probabilities with ROC-AUC ≥ 0.85.
- Maintain high recall on top-decile customers to maximize retention lift.
- Provide explanation artifacts consumable by ops teams and APIs.

### Baseline
- Logistic Regression with class weighting and standardized numeric features.
- Evaluate using stratified 5-fold CV and hold-out test set; serve as benchmark for future gains.

### Advanced Models
- **Implemented**: Gradient Boosting models (XGBoost, LightGBM, CatBoost) with Optuna hyperparameter tuning.
- **Implemented**: Deep tabular model (feedforward neural network) with PyTorch, tuned via Optuna.
- **Note**: The deep tabular implementation is a custom feedforward network, not TabNet/FT-Transformer. TabNet/FT-Transformer are more advanced architectures that could be added in the future.
- **Future work**: Ensemble stacker combining calibrated outputs from multiple models.

### Imbalance Handling
- Techniques implemented: **class weights** (baseline logistic regression) and optional **SMOTE** oversampling.
- Planned/optional techniques: SMOTE variants, focal loss (for deep models), and advanced resampling can be layered on later.
- Control: set `use_smote` in `TrainingConfig` / `configs/model_config.json` to enable SMOTE on the training split only.
- Decision criteria: best recall@precision≥0.5 on validation plus ROC-AUC.

### Evaluation Suite
- Metrics: ROC-AUC, PR-AUC, F1, precision/recall at top-k, lift charts, calibration curves.
- Reporting: automated `reports/model_card_<timestamp>.md` summarizing performance, assumptions, and caveats.

### Experiment Tracking
- MLflow captures experiment metadata, artifacts, SHAP values, confusion matrices.
- By default, the project uses **file-based MLflow tracking** (`MLFLOW_TRACKING_URI=file:./mlruns`), which does **not** provide a Model Registry.
- When pointed at a tracking server that supports the Model Registry, promotion uses stages: `Staging` (under evaluation) → `Production` (serving) → `Archived`.
- In file-based mode, the retraining DAG logs a clear warning and falls back to “file-only” promotion (artifacts/metrics are still logged, but no registry state is created).
- Promotion checklist ensures monitoring hooks and rollback plan defined before deployment.

