"""Model training utilities with Optuna hyperparameter tuning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.models.baseline import BaselineModel


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: str  # "baseline", "xgboost", "lightgbm"
    random_state: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    use_smote: bool = False
    calibrate: bool = True
    n_trials: int = 50  # For Optuna
    timeout: int | None = None  # For Optuna (seconds)


class ModelTrainer:
    """Trainer for churn prediction models with hyperparameter tuning."""

    def __init__(
        self,
        config: TrainingConfig,
        mlflow_experiment_name: str = "telco_churn",
    ) -> None:
        """Initialize model trainer.

        Args:
            config: Training configuration.
            mlflow_experiment_name: MLflow experiment name.
        """
        self.config = config
        self.mlflow_experiment_name = mlflow_experiment_name
        self.model: Any = None
        self.best_params: dict[str, Any] | None = None

        # Setup MLflow
        mlflow.set_experiment(mlflow_experiment_name)

    def train_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaselineModel:
        """Train baseline logistic regression model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).

        Returns:
            Trained baseline model.
        """
        with mlflow.start_run(run_name="baseline_logistic_regression"):
            model = BaselineModel(
                class_weight="balanced",
                random_state=self.config.random_state,
                calibrate=self.config.calibrate,
            )

            model.fit(X_train, y_train)

            # Log parameters
            mlflow.log_params(
                {
                    "model_type": "baseline",
                    "class_weight": "balanced",
                    "calibrate": self.config.calibrate,
                }
            )

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                from sklearn.metrics import roc_auc_score

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, y_pred_proba)
                mlflow.log_metric("val_roc_auc", val_auc)

            # Cross-validation on training set
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state
            )
            cv_scores = cross_val_score(
                model.get_model(), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            mlflow.log_metrics(
                {
                    "cv_roc_auc_mean": cv_scores.mean(),
                    "cv_roc_auc_std": cv_scores.std(),
                }
            )

            self.model = model
            return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> XGBClassifier:
        """Train XGBoost model with Optuna hyperparameter tuning.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).

        Returns:
            Trained XGBoost model.
        """
        # Use validation set for early stopping if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": self.config.random_state,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
            }

            model = XGBClassifier(**params, n_jobs=-1)

            if eval_set:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=50,
                    verbose=False,
                )
                # Get best score from validation set
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                from sklearn.metrics import roc_auc_score

                score = roc_auc_score(y_val, y_pred_proba)
            else:
                # Use cross-validation
                cv = StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state,
                )
                scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
                )
                score = scores.mean()

            return score

        # Run Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name="xgboost_optimization",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        # Train final model with best parameters
        best_params = study.best_params.copy()
        best_params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": self.config.random_state,
                "n_jobs": -1,
            }
        )

        final_model = XGBClassifier(**best_params)

        if eval_set:
            final_model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False,
            )
        else:
            final_model.fit(X_train, y_train)

        # Log to MLflow
        with mlflow.start_run(run_name="xgboost_optimized"):
            mlflow.log_params({"model_type": "xgboost", **best_params})
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.log_params({"n_trials": self.config.n_trials})

            if X_val is not None and y_val is not None:
                from sklearn.metrics import roc_auc_score

                y_pred_proba = final_model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, y_pred_proba)
                mlflow.log_metric("val_roc_auc", val_auc)

            # Log model
            mlflow.xgboost.log_model(final_model, "model")

        self.model = final_model
        self.best_params = best_params
        return final_model

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Any:
        """Train LightGBM model with Optuna hyperparameter tuning.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).

        Returns:
            Trained LightGBM model.
        """
        import lightgbm as lgb

        # Use validation set for early stopping if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "random_state": self.config.random_state,
                "verbosity": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
            }

            model = lgb.LGBMClassifier(**params, n_jobs=-1)

            if eval_set:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                from sklearn.metrics import roc_auc_score

                score = roc_auc_score(y_val, y_pred_proba)
            else:
                cv = StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state,
                )
                scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
                )
                score = scores.mean()

            return score

        # Run Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name="lightgbm_optimization",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        # Train final model with best parameters
        best_params = study.best_params.copy()
        best_params.update(
            {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "random_state": self.config.random_state,
                "verbosity": -1,
                "n_jobs": -1,
            }
        )

        final_model = lgb.LGBMClassifier(**best_params)

        if eval_set:
            final_model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
        else:
            final_model.fit(X_train, y_train)

        # Log to MLflow
        with mlflow.start_run(run_name="lightgbm_optimized"):
            mlflow.log_params({"model_type": "lightgbm", **best_params})
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.log_params({"n_trials": self.config.n_trials})

            if X_val is not None and y_val is not None:
                from sklearn.metrics import roc_auc_score

                y_pred_proba = final_model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, y_pred_proba)
                mlflow.log_metric("val_roc_auc", val_auc)

            # Log model
            mlflow.lightgbm.log_model(final_model, "model")

        self.model = final_model
        self.best_params = best_params
        return final_model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Any:
        """Train model based on config.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).

        Returns:
            Trained model.
        """
        if self.config.model_type == "baseline":
            return self.train_baseline(X_train, y_train, X_val, y_val)
        elif self.config.model_type == "xgboost":
            return self.train_xgboost(X_train, y_train, X_val, y_val)
        elif self.config.model_type == "lightgbm":
            return self.train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

