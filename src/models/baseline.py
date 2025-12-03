"""Baseline logistic regression model with class imbalance handling."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class BaselineModel:
    """Baseline logistic regression model with class weighting and calibration."""

    def __init__(
        self,
        class_weight: str | dict[int, float] = "balanced",
        random_state: int = 42,
        max_iter: int = 1000,
        calibrate: bool = True,
    ) -> None:
        """Initialize baseline model.

        Args:
            class_weight: Class weight strategy or dict. Default: "balanced".
            random_state: Random seed for reproducibility.
            max_iter: Maximum iterations for logistic regression.
            calibrate: Whether to calibrate probabilities using Platt scaling.
        """
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.model: Pipeline | LogisticRegression | CalibratedClassifierCV | None = None

    def build(self) -> Pipeline | LogisticRegression | CalibratedClassifierCV:
        """Build the baseline model pipeline.

        Returns:
            Fitted model pipeline.
        """
        base_model = LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            solver="lbfgs",
        )

        if self.calibrate:
            # Use isotonic calibration for better probability estimates
            self.model = CalibratedClassifierCV(
                base_model, method="isotonic", cv=5, n_jobs=-1
            )
        else:
            self.model = base_model

        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselineModel:
        """Fit the baseline model.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Self for method chaining.
        """
        if self.model is None:
            self.model = self.build()

        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        result = self.model.predict(X)
        return np.asarray(result)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Predicted probabilities for each class.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        result = self.model.predict_proba(X)
        return np.asarray(result)

    def get_model(self) -> Pipeline | LogisticRegression | CalibratedClassifierCV:
        """Get the underlying model object.

        Returns:
            The model object.
        """
        if self.model is None:
            self.model = self.build()
        return self.model

