"""
Feature engineering pipeline using sklearn pipelines.

This module provides a complete feature engineering pipeline that can be
used for both batch processing and streaming inference.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.transformers import (
    CustomerLifetimeValueTransformer,
    RevenueSignalTransformer,
    ServiceCountTransformer,
    TenureBucketTransformer,
)


def create_feature_pipeline(
    numeric_features: List[str] | None = None,
    categorical_features: List[str] | None = None,
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
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
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
        target = df[target_column].map({"Yes": 1, "No": 0}) if target_column == "Churn" else df[target_column]
        df_features = df.drop(columns=[target_column])
    else:
        df_features = df.copy()

    # Apply custom transformers first (these modify dataframe in place)
    custom_pipeline = pipeline.named_steps["custom_features"]
    df_with_custom = custom_pipeline.transform(df_features)

    # Apply preprocessor (this returns numpy array)
    preprocessor = pipeline.named_steps["preprocessor"]
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

