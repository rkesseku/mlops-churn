"""Load the churn model from the MLflow registry at FastAPI startup.

Singleton pattern: the model is heavy (loading takes ~2 seconds), so we
load it once per process and share. Per-request loading would be a major
performance bug.

The model is referenced by *alias*, not version. The training script
points `staging` at whatever version it just registered, so this code
always gets "the latest staging model" without knowing the version
number ahead of time.

In production:
  - The serving container reloads on deploy (k8s rolling restart)
  - To promote a new model, training updates the alias and CI redeploys
  - For zero-downtime model swap, you'd add a /reload endpoint that
    re-fetches by alias — out of scope for v1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from churn_platform.config import settings

log = logging.getLogger("model_loader")

MODEL_NAME = "churn_xgboost"
DEFAULT_ALIAS = "staging"


@dataclass
class LoadedModel:
    """The model and metadata about it. Returned from load()."""

    model: Any  # The actual mlflow.pyfunc.PyFuncModel
    name: str
    version: str
    alias: str

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Return P(churn=1) for each row.

        The pyfunc wrapper returns predictions; for XGBoost binary classifiers,
        that's `predict_proba`-style probabilities of the positive class.

        XGBoost is strict about column order; we reindex the incoming DataFrame
        to match the order the model was trained on. This is more robust than
        depending on Pydantic schema field order matching training order.
        """
        df = features.copy()

        # Re-apply categorical dtypes — pandas drops them through dict
        # round-trips. The model was trained with these columns as category.
        for col in ["contract_type", "payment_method", "internet_service", "tenure_bucket"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Reindex to the booster's expected column order, if available.
        # mlflow.pyfunc wraps the model; the booster sits one level in.
        booster = getattr(self.model, "_model_impl", None)
        expected = None
        if booster is not None:
            inner = getattr(booster, "xgb_model", None) or getattr(booster, "model", None)
            if inner is not None:
                try:
                    expected = list(inner.get_booster().feature_names)
                except Exception:
                    expected = None
        if expected:
            df = df[expected]

        raw = self.model.predict(df)
        # mlflow.xgboost wraps as pyfunc and returns ndarray of probabilities
        # for binary classification (shape (n,) or (n, 2)). Normalize to a
        # Series of positive-class probability.
        import numpy as np
        arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return pd.Series(arr[:, 1])
        return pd.Series(arr.flatten())


def load(alias: str = DEFAULT_ALIAS) -> LoadedModel:
    """Load the model identified by `models:/<name>@<alias>` from MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{MODEL_NAME}@{alias}"
    log.info("Loading model from %s", uri)

    model = mlflow.pyfunc.load_model(uri)

    # Look up the underlying version number for /health.
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)

    log.info("Loaded %s v%s (run_id=%s)", MODEL_NAME, mv.version, mv.run_id)
    return LoadedModel(
        model=model,
        name=MODEL_NAME,
        version=mv.version,
        alias=alias,
    )
