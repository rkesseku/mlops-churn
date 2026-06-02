"""Pydantic request and response schemas for the serving API.

These do double duty:
  1. Runtime validation: malformed requests get a 422 before they reach
     the model. XGBoost will not protect itself; we have to.
  2. OpenAPI / Swagger documentation: FastAPI reads these to render
     /docs and /redoc automatically.

The field list mirrors the FEATURE set produced by the Spark pipeline,
NOT the raw schema. Clients send already-featurized rows. (In a fuller
production system, the API would call the feature pipeline itself on
raw input — that's a Phase 9+ choice we're not making here.)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Allowed literal values for categorical features. Same as what the data
# generator produces; Pydantic rejects anything else with a clear error.
ContractType = Literal["month-to-month", "one-year", "two-year"]
PaymentMethod = Literal["echeck", "mailcheck", "banktx", "creditcard"]
InternetService = Literal["DSL", "Fiber", "None"]
TenureBucket = Literal["new", "mid", "loyal"]


class CustomerFeatures(BaseModel):
    """One customer's featurized row. Matches the Spark output schema."""

    # Allow extra examples in the OpenAPI docs.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure_months": 12,
                "contract_type": "month-to-month",
                "monthly_charges": 65.50,
                "payment_method": "echeck",
                "internet_service": "Fiber",
                "num_support_calls": 3,
                "num_logins_30d": 5,
                "avg_session_minutes": 12.5,
                "days_since_last_login": 10,
                "has_paperless_billing": True,
                "auto_renew": False,
                "discount_applied": 0.0,
                "charge_ratio": 1.05,
                "tenure_bucket": "mid",
                "engagement_score": 6.25,
                "support_calls_per_login": 0.5,
                "risk_flag_no_renew_m2m": True,
                "total_charges_filled": 786.0,
                "nps_filled": 4.0,
                "nps_was_null": False,
            }
        }
    )

    # Raw features
    tenure_months: int = Field(..., ge=0, le=72)
    contract_type: ContractType
    monthly_charges: float = Field(..., ge=15.0, le=120.0)
    payment_method: PaymentMethod
    internet_service: InternetService
    num_support_calls: int = Field(..., ge=0)
    num_logins_30d: int = Field(..., ge=0)
    avg_session_minutes: float = Field(..., ge=0.0)
    days_since_last_login: int = Field(..., ge=0, le=365)
    has_paperless_billing: bool
    auto_renew: bool
    discount_applied: float = Field(..., ge=0.0, le=0.5)

    # Derived features (computed by the Spark pipeline normally;
    # client is responsible for sending them in v1 of the API).
    charge_ratio: float
    tenure_bucket: TenureBucket
    engagement_score: float
    support_calls_per_login: float
    risk_flag_no_renew_m2m: bool
    total_charges_filled: float
    nps_filled: float
    nps_was_null: bool


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label: bool
    model_name: str
    model_version: str


class BatchPredictRequest(BaseModel):
    instances: list[CustomerFeatures] = Field(..., min_length=1, max_length=10000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    n: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["ok", "degraded"]
    model_name: str
    model_version: str | None
    model_loaded: bool
