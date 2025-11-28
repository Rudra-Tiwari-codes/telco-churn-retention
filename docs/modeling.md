## Modeling Strategy

### Objectives
- Deliver calibrated churn probabilities with ROC-AUC ≥ 0.85.
- Maintain high recall on top-decile customers to maximize retention lift.
- Provide explanation artifacts consumable by ops teams and APIs.

### Baseline
- Logistic Regression with class weighting and standardized numeric features.
- Evaluate using stratified 5-fold CV and hold-out test set; serve as benchmark for future gains.

### Advanced Models
- Gradient Boosting: XGBoost, LightGBM, CatBoost tuned via Optuna.
- Deep Tabular Model: TabNet or FT-Transformer for nonlinear interactions.
- Ensemble stacker combining calibrated outputs if it exceeds baseline metrics.

### Imbalance Handling
- Techniques compared: class weights, SMOTE variants, focal loss (deep model), and probability calibration.
- Decision criteria: best recall@precision≥0.5 on validation plus ROC-AUC.

### Evaluation Suite
- Metrics: ROC-AUC, PR-AUC, F1, precision/recall at top-k, lift charts, calibration curves.
- Reporting: automated `reports/model_card_<timestamp>.md` summarizing performance, assumptions, and caveats.

### Experiment Tracking
- MLflow captures experiment metadata, artifacts, SHAP values, confusion matrices.
- Model registry stages: `Staging` (under evaluation) → `Production` (serving) → `Archived`.
- Promotion checklist ensures monitoring hooks and rollback plan defined before deployment.

