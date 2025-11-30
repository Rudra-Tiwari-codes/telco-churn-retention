"""
Pydantic models for API request and response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerData(BaseModel):
    """Customer data for churn prediction."""

    customerID: str = Field(..., description="Unique customer identifier")
    gender: str = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen")
    Partner: str = Field(..., description="Whether customer has partner")
    Dependents: str = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, description="Number of months customer has been with company")
    PhoneService: str = Field(..., description="Whether customer has phone service")
    MultipleLines: str = Field(..., description="Whether customer has multiple lines")
    InternetService: str = Field(..., description="Type of internet service")
    OnlineSecurity: str = Field(..., description="Whether customer has online security")
    OnlineBackup: str = Field(..., description="Whether customer has online backup")
    DeviceProtection: str = Field(..., description="Whether customer has device protection")
    TechSupport: str = Field(..., description="Whether customer has tech support")
    StreamingTV: str = Field(..., description="Whether customer has streaming TV")
    StreamingMovies: str = Field(..., description="Whether customer has streaming movies")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Whether customer has paperless billing")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float | None = Field(None, ge=0, description="Total charges (can be null)")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Validate gender field."""
        if v not in ["Male", "Female"]:
            raise ValueError(f"Must be 'Male' or 'Female', got '{v}'")
        return v

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling")
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        """Validate Yes/No fields."""
        if v not in ["Yes", "No"]:
            raise ValueError(f"Must be 'Yes' or 'No', got '{v}'")
        return v

    @field_validator("MultipleLines")
    @classmethod
    def validate_multiple_lines(cls, v: str) -> str:
        """Validate MultipleLines field."""
        if v not in ["Yes", "No", "No phone service"]:
            raise ValueError(f"Must be 'Yes', 'No', or 'No phone service', got '{v}'")
        return v

    @field_validator(
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    @classmethod
    def validate_service_fields(cls, v: str) -> str:
        """Validate service fields."""
        if v not in ["Yes", "No", "No internet service"]:
            raise ValueError(f"Must be 'Yes', 'No', or 'No internet service', got '{v}'")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        """Validate internet service field."""
        valid_values = ["DSL", "Fiber optic", "No"]
        if v not in valid_values:
            raise ValueError(f"Must be one of {valid_values}, got '{v}'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        """Validate contract field."""
        valid_values = ["Month-to-month", "One year", "Two year"]
        if v not in valid_values:
            raise ValueError(f"Must be one of {valid_values}, got '{v}'")
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment_method(cls, v: str) -> str:
        """Validate payment method field."""
        valid_values = [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
        if v not in valid_values:
            raise ValueError(f"Must be one of {valid_values}, got '{v}'")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "1234-ABCD",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.5,
                "TotalCharges": 845.0,
            }
        }
    )


class FeatureExplanation(BaseModel):
    """Feature explanation from SHAP."""

    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value for this feature")
    value: float = Field(..., description="Feature value")


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    customerID: str = Field(..., description="Customer identifier")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    threshold: float = Field(..., description="Threshold used for prediction")
    explanation: list[FeatureExplanation] = Field(
        default_factory=list, description="Top feature explanations from SHAP"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "1234-ABCD",
                "churn_probability": 0.73,
                "churn_prediction": True,
                "threshold": 0.5,
                "explanation": [
                    {"feature": "tenure", "shap_value": -0.15, "value": 12.0},
                    {"feature": "MonthlyCharges", "shap_value": 0.12, "value": 70.5},
                ],
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    customers: list[CustomerData] = Field(
        ..., min_length=1, description="List of customers to predict"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customers": [
                    {
                        "customerID": "1234-ABCD",
                        "gender": "Male",
                        "SeniorCitizen": 0,
                        "Partner": "Yes",
                        "Dependents": "No",
                        "tenure": 12,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "DSL",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "Yes",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "No",
                        "StreamingMovies": "No",
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 70.5,
                        "TotalCharges": 845.0,
                    }
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    total_customers: int = Field(..., description="Total number of customers processed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "customerID": "1234-ABCD",
                        "churn_probability": 0.73,
                        "churn_prediction": True,
                        "threshold": 0.5,
                        "explanation": [],
                    }
                ],
                "total_customers": 1,
            }
        }
    )


class ModelMetadata(BaseModel):
    """Model metadata information."""

    model_type: str = Field(..., description="Type of model (baseline, xgboost, lightgbm)")
    model_version: str = Field(..., description="Model version/timestamp")
    feature_count: int = Field(..., description="Number of features")
    feature_names: list[str] = Field(..., description="List of feature names")
    performance_metrics: dict[str, float] = Field(..., description="Model performance metrics")
    threshold: float = Field(..., description="Default prediction threshold")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "xgboost",
                "model_version": "20250127T120000Z",
                "feature_count": 45,
                "feature_names": ["tenure", "MonthlyCharges", ...],
                "performance_metrics": {"roc_auc": 0.87, "pr_auc": 0.65, "f1": 0.72},
                "threshold": 0.5,
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    pipeline_loaded: bool = Field(..., description="Whether feature pipeline is loaded")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "pipeline_loaded": True,
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Error detail")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Validation error",
                "detail": "Field 'tenure' must be >= 0",
            }
        }
    )
