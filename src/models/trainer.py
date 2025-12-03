"""Model training utilities with Optuna hyperparameter tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.models.baseline import BaselineModel


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: str  # "baseline", "xgboost", "lightgbm", "catboost", "deep_tabular"
    random_state: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    use_smote: bool = False
    calibrate: bool = True
    n_trials: int = 20  # For Optuna
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
                    "use_smote": self.config.use_smote,
                }
            )

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
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
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
            }

            # For XGBoost 3.x, early stopping API has changed
            # Fit without early stopping for Optuna trials (still works fine)
            model = XGBClassifier(**params, n_jobs=-1)
            
            if eval_set:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    verbose=False,
                )
                # Get best score from validation set
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
            else:
                # Use cross-validation
                model.fit(X_train, y_train)
                cv = StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state,
                )
                scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
                )
                score = float(scores.mean())

            return score  # type: ignore[no-any-return]

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

        # For XGBoost 3.x, early stopping API has changed
        # Fit final model without early stopping (still works fine)
        final_model = XGBClassifier(**best_params)
        if eval_set:
            final_model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            final_model.fit(X_train, y_train)

        # Log to MLflow
        with mlflow.start_run(run_name="xgboost_optimized"):
            mlflow.log_params(
                {
                    "model_type": "xgboost",
                    "use_smote": self.config.use_smote,
                    **best_params,
                }
            )
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.log_params({"n_trials": self.config.n_trials})

            if X_val is not None and y_val is not None:
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
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
            }

            model = lgb.LGBMClassifier(**params, n_jobs=-1)  # type: ignore[arg-type]

            if eval_set:
                from typing import cast
                from collections.abc import Sequence

                # Type cast to satisfy mypy's variance requirements
                eval_set_typed = cast(
                    Sequence[tuple[np.ndarray, np.ndarray]], eval_set
                )
                model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set_typed,  # type: ignore[arg-type]
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
                y_pred_proba = model.predict_proba(X_val)[:, 1]
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
                score = float(scores.mean())

            return score  # type: ignore[no-any-return]

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
            from typing import cast
            from collections.abc import Sequence

            # Type cast to satisfy mypy's variance requirements
            eval_set_typed = cast(
                Sequence[tuple[np.ndarray, np.ndarray]], eval_set
            )
            final_model.fit(
                X_train,
                y_train,
                eval_set=eval_set_typed,  # type: ignore[arg-type]
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
        else:
            final_model.fit(X_train, y_train)

        # Log to MLflow
        with mlflow.start_run(run_name="lightgbm_optimized"):
            mlflow.log_params(
                {
                    "model_type": "lightgbm",
                    "use_smote": self.config.use_smote,
                    **best_params,
                }
            )
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.log_params({"n_trials": self.config.n_trials})

            if X_val is not None and y_val is not None:
                y_pred_proba = final_model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, y_pred_proba)
                mlflow.log_metric("val_roc_auc", val_auc)

            # Log model
            mlflow.lightgbm.log_model(final_model, "model")

        self.model = final_model
        self.best_params = best_params
        return final_model

    def train_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> CatBoostClassifier:
        """Train CatBoost model with Optuna hyperparameter tuning."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": self.config.random_state,
                "verbose": False,
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1.0, 10.0, log=True
                ),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 1.0
                ),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }

            model = CatBoostClassifier(
                **params,
                thread_count=-1,
            )

            if X_val is not None and y_val is not None:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                )
                y_pred_proba = model.predict_proba(X_val)[:, 1]
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
                score = float(scores.mean())

            return score  # type: ignore[no-any-return]

        study = optuna.create_study(
            direction="maximize",
            study_name="catboost_optimization",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        best_params = study.best_params.copy()
        best_params.update(
            {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": self.config.random_state,
                "verbose": False,
            }
        )

        final_model = CatBoostClassifier(**best_params, thread_count=-1)
        if X_val is not None and y_val is not None:
            final_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
        else:
            final_model.fit(X_train, y_train)

        with mlflow.start_run(run_name="catboost_optimized"):
            mlflow.log_params(
                {
                    "model_type": "catboost",
                    "use_smote": self.config.use_smote,
                    **best_params,
                }
            )
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.log_params({"n_trials": self.config.n_trials})

            if X_val is not None and y_val is not None:
                y_pred_proba = final_model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, y_pred_proba)
                mlflow.log_metric("val_roc_auc", val_auc)

            mlflow.catboost.log_model(final_model, "model")

        self.model = final_model
        self.best_params = best_params
        return final_model

    def train_deep_tabular(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Any:
        """Train a simple deep tabular model (feedforward network) tuned via Optuna."""
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if X_val is None or y_val is None:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train,
                y_train,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y_train,
            )
        else:
            X_train_split, y_train_split = X_train, y_train
            X_val_split, y_val_split = X_val, y_val

        X_train_t = torch.tensor(X_train_split, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_split, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_split, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_split, dtype=torch.float32)

        def create_model(trial: optuna.Trial, n_features: int) -> nn.Module:
            layers: list[nn.Module] = []
            n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
            input_dim = n_features
            for i in range(n_hidden_layers):
                hidden_dim = trial.suggest_int(f"hidden_dim_{i}", 32, 256)
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                dropout = trial.suggest_float(f"dropout_{i}", 0.0, 0.5)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            return nn.Sequential(*layers)

        def objective(trial: optuna.Trial) -> float:
            model = create_model(trial, X_train_t.shape[1]).to(device)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            batch_size = trial.suggest_int("batch_size", 64, 512)
            epochs = trial.suggest_int("epochs", 10, 40)

            pos_weight_value = float(
                (len(y_train_split) - y_train_split.sum())
                / max(y_train_split.sum(), 1)
            )
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight_value], device=device)
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            model.train()
            for _ in range(epochs):
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb).squeeze(1)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_t.to(device)).squeeze(1)
                probs_val = torch.sigmoid(logits_val).cpu().numpy()

            return float(roc_auc_score(y_val_split, probs_val))

        study = optuna.create_study(
            direction="maximize",
            study_name="deep_tabular_optimization",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )
        study.optimize(
            objective,
            n_trials=min(self.config.n_trials, 25),
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        best_params = study.best_params

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from optuna.trial import FixedTrial

        fixed_trial: optuna.Trial = FixedTrial(best_params)  # type: ignore[assignment]
        best_model = create_model(fixed_trial, X_train.shape[1]).to(device)
        lr = best_params["lr"]
        weight_decay = best_params["weight_decay"]
        batch_size = best_params["batch_size"]
        epochs = best_params["epochs"]

        pos_weight_value = float(
            (len(y_train_split) - y_train_split.sum())
            / max(y_train_split.sum(), 1)
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], device=device)
        )
        optimizer = torch.optim.Adam(
            best_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = best_model(xb).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        class DeepTabularWrapper:
            """Wrapper to provide sklearn-like predict_proba API."""

            def __init__(self, net: Any, device: Any) -> None:
                self.net = net
                self.device = device

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                self.net.eval()
                import torch

                with torch.no_grad():
                    x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                    logits = self.net(x_t).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                probs = probs.reshape(-1, 1)
                return np.hstack([1 - probs, probs])

        wrapped_model = DeepTabularWrapper(best_model, device)

        with mlflow.start_run(run_name="deep_tabular_optimized"):
            mlflow.log_params(
                {
                    "model_type": "deep_tabular",
                    "use_smote": self.config.use_smote,
                    **best_params,
                }
            )
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.log_params({"n_trials": min(self.config.n_trials, 25)})

        self.model = wrapped_model
        self.best_params = best_params
        return wrapped_model

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> Any:
        """Train model based on config.

        Args:
            X_train: Training features (DataFrame or numpy array).
            y_train: Training targets (Series or numpy array).
            X_val: Validation features (optional, DataFrame or numpy array).
            y_val: Validation targets (optional, Series or numpy array).

        Returns:
            Trained model.
        """
        # Convert DataFrames/Series to numpy arrays for baseline model.
        # XGBoost and LightGBM can handle DataFrames directly, but we standardize on numpy arrays.
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train

        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
        else:
            y_train_array = y_train

        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_array = X_val.values
            else:
                X_val_array = X_val
        else:
            X_val_array = None

        if y_val is not None:
            if isinstance(y_val, pd.Series):
                y_val_array = y_val.values
            else:
                y_val_array = y_val
        else:
            y_val_array = None

        # Optional class-imbalance handling on the training split only.
        if self.config.use_smote:
            smote = SMOTE(random_state=self.config.random_state)
            X_train_array, y_train_array = smote.fit_resample(X_train_array, y_train_array)

        if self.config.model_type == "baseline":
            return self.train_baseline(X_train_array, y_train_array, X_val_array, y_val_array)
        elif self.config.model_type == "xgboost":
            return self.train_xgboost(X_train_array, y_train_array, X_val_array, y_val_array)
        elif self.config.model_type == "lightgbm":
            return self.train_lightgbm(X_train_array, y_train_array, X_val_array, y_val_array)
        elif self.config.model_type == "catboost":
            return self.train_catboost(X_train_array, y_train_array, X_val_array, y_val_array)
        elif self.config.model_type == "deep_tabular":
            return self.train_deep_tabular(X_train_array, y_train_array, X_val_array, y_val_array)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

