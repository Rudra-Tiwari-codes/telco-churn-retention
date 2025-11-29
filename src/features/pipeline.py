"""
Feature engineering pipeline using sklearn pipelines.

This module provides a complete feature engineering pipeline that can be
used for both batch processing and streaming inference.
"""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.transformers import (
    CustomerLifetimeValueTransformer,
    RevenueSignalTransformer,
    ServiceCountTransformer,
    TenureBucketTransformer,
)


def create_feature_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    use_target_encoding: bool = False,
) -> Pipeline:
    """Create feature engineering pipeline.

    The pipeline first applies custom transformers to create new features,
    then applies scaling and encoding to all features.

    Args:
        numeric_features: List of numeric feature names after custom transformations.
                         If None, will be determined dynamically after custom features are created.
        categorical_features: List of categorical feature names after custom transformations.
                            If None, will be determined dynamically after custom features are created.
        use_target_encoding: Whether to use target encoding for categoricals (not yet implemented).

    Returns:
        sklearn Pipeline for feature engineering.
    """
    # Base feature lists (features that exist in raw data)
    base_numeric_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    base_categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    # Features created by custom transformers
    engineered_numeric_features = [
        "service_count",
        "charges_ratio",
        "avg_monthly_charge",
        "charge_increase_flag",
        "clv_approximation",
    ]

    engineered_categorical_features = [
        "tenure_bucket",
    ]

    # Create custom transformers pipeline
    custom_transformers = Pipeline(
        [
            ("tenure_buckets", TenureBucketTransformer()),
            ("service_count", ServiceCountTransformer()),
            ("revenue_signals", RevenueSignalTransformer()),
            ("clv", CustomerLifetimeValueTransformer()),
        ]
    )

    # Use provided feature lists or combine base + engineered
    if numeric_features is None:
        numeric_features = base_numeric_features + engineered_numeric_features

    if categorical_features is None:
        categorical_features = base_categorical_features + engineered_categorical_features

    # Create preprocessing pipeline
    # Note: ColumnTransformer will be applied to data after custom features are created
    # Add imputer for numeric features to handle NaN values
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ],
        remainder="drop",  # Drop any remaining columns not explicitly handled
    )

    # Combine into full pipeline
    full_pipeline = Pipeline(
        [
            ("custom_features", custom_transformers),
            ("preprocessor", preprocessor),
        ]
    )

    return full_pipeline


def apply_feature_pipeline(
    df: pd.DataFrame,
    pipeline: Pipeline,
    target_column: str = "Churn",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Apply feature pipeline to dataframe.

    Args:
        df: Input dataframe.
        pipeline: Fitted feature pipeline.
        target_column: Name of target column (if present).

    Returns:
        Tuple of (transformed_features_df, target_series).
    """
    # Separate target if present
    target = None
    if target_column in df.columns:
        if target_column == "Churn":
            target = df[target_column].map({"Yes": 1, "No": 0})
        else:
            target = df[target_column]
        df_features = df.drop(columns=[target_column])
    else:
        df_features = df.copy()

    # Ensure required columns exist for custom transformers
    required_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    missing_columns = [col for col in required_columns if col not in df_features.columns]
    if missing_columns:
        raise ValueError(
            f"Required columns missing from dataframe: {missing_columns}. "
            f"Available columns: {list(df_features.columns)}"
        )

    # Apply custom transformers first (these modify dataframe in place)
    # Use pipeline's internal _transform method or access steps directly
    # to avoid FutureWarning about accessing named_steps before fitting check

    # Suppress FutureWarning globally for this section since we know pipeline is fitted
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.pipeline")

        # Access steps using steps_ attribute (set after fitting) to avoid property access
        # that triggers the fitted check
        if hasattr(pipeline, "steps_"):
            # Pipeline is fitted, use steps_ directly
            steps_list = pipeline.steps_
        elif hasattr(pipeline, "_sklearn_fitted") and pipeline._sklearn_fitted:
            # Alternative check for fitted state
            steps_list = pipeline.steps
        else:
            # Last resort: try to access named_steps with warning suppressed
            try:
                steps_dict = dict(pipeline.named_steps.items())
                custom_pipeline = steps_dict["custom_features"]
                preprocessor = steps_dict["preprocessor"]
            except (AttributeError, KeyError) as err:
                raise ValueError("Pipeline is not fitted. Call pipeline.fit() first.") from err
            else:
                # Successfully got steps from named_steps
                df_with_custom = custom_pipeline.transform(df_features)
                transformed = preprocessor.transform(df_with_custom)
                # Convert to DataFrame and return early
                if isinstance(transformed, pd.DataFrame):
                    transformed_df = transformed
                else:
                    if hasattr(preprocessor, "get_feature_names_out"):
                        feature_names = preprocessor.get_feature_names_out()
                    else:
                        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
                    transformed_df = pd.DataFrame(
                        transformed, columns=feature_names, index=df_features.index
                    )
                return transformed_df, target

        # Convert steps_list to dict
        steps_dict = dict(steps_list)
        custom_pipeline = steps_dict["custom_features"]
        preprocessor = steps_dict["preprocessor"]

        # Transform using custom pipeline
        df_with_custom = custom_pipeline.transform(df_features)

        # Transform using preprocessor
        transformed = preprocessor.transform(df_with_custom)

    # Convert to DataFrame
    if isinstance(transformed, pd.DataFrame):
        transformed_df = transformed
    else:
        # Get feature names from preprocessor
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            # Fallback: generate names based on shape
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

        transformed_df = pd.DataFrame(transformed, columns=feature_names, index=df_features.index)

    return transformed_df, target
