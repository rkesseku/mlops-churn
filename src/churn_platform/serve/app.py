"""FastAPI app serving churn predictions with Prometheus metrics.

Endpoints:
  GET  /            — landing page with links
  GET  /health      — liveness + model version
  POST /predict     — single-row prediction
  POST /predict_batch — many rows in one call
  GET  /metrics     — Prometheus exposition (auto-added by Instrumentator)

The model is loaded once at startup via the lifespan context manager,
then shared across all requests via app.state.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from churn_platform.monitor.drift import score_drift
from churn_platform.serve.model_loader import LoadedModel, load
from churn_platform.serve.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    HealthResponse,
    PredictResponse,
)

log = logging.getLogger("serve")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# ----- Prometheus metrics --------------------------------------------------
#
# Layered on top of the default HTTP metrics (latency histogram, request
# count by status) added by Instrumentator below.
#
# Naming convention: snake_case, suffix denotes type:
#   _total  -> Counter
#   (none)  -> Gauge or Histogram

PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Number of churn predictions served, broken down by predicted label.",
    labelnames=["churn_label"],
)

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of predicted churn probabilities. Drifts here can "
    "indicate input drift or model staleness.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_VERSION_INFO = Gauge(
    "model_version_info",
    "Always 1; the model version is encoded as a label.",
    labelnames=["name", "version"],
)


# ----- Lifespan ------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup. Don't crash if it fails — /health reports degraded."""
    log.info("Starting churn-serving")
    try:
        app.state.loaded = load()
        log.info("Model loaded successfully")
        MODEL_VERSION_INFO.labels(
            name=app.state.loaded.name,
            version=app.state.loaded.version,
        ).set(1)
    except Exception as e:
        log.error("Failed to load model on startup: %s", e)
        app.state.loaded = None
    yield
    log.info("Shutting down churn-serving")


app = FastAPI(
    title="Churn Prediction Service",
    description=(
        "Predicts customer churn probability using an XGBoost model loaded "
        "from the MLflow registry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Wire up default HTTP metrics + the /metrics endpoint.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ----- Dependencies --------------------------------------------------------


def get_model(request: Request) -> LoadedModel:
    """Inject the loaded model into endpoints. 503 if loading failed."""
    loaded: LoadedModel | None = request.app.state.loaded
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health and serving logs.",
        )
    return loaded


# ----- Endpoints -----------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h1>Churn Prediction Service</h1>
    <ul>
      <li><a href="/docs">Swagger UI</a></li>
      <li><a href="/health">Health check</a></li>
      <li><a href="/metrics">Prometheus metrics</a></li>
      <li><a href="/redoc">ReDoc</a></li>
    </ul>
    """


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    loaded: LoadedModel | None = request.app.state.loaded
    if loaded is None:
        return HealthResponse(
            status="degraded",
            model_name="churn_xgboost",
            model_version=None,
            model_loaded=False,
        )
    return HealthResponse(
        status="ok",
        model_name=loaded.name,
        model_version=loaded.version,
        model_loaded=True,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    features: CustomerFeatures,
    model: Annotated[LoadedModel, Depends(get_model)],
):
    """Predict churn probability for one customer."""
    df = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(df).iloc[0])
    label = proba >= 0.5

    CHURN_PROBABILITY.observe(proba)
    PREDICTIONS_TOTAL.labels(churn_label=str(label).lower()).inc()

    return PredictResponse(
        churn_probability=proba,
        churn_label=label,
        model_name=model.name,
        model_version=model.version,
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(
    body: BatchPredictRequest,
    model: Annotated[LoadedModel, Depends(get_model)],
):
    """Predict churn for many customers in one call."""
    rows = [c.model_dump() for c in body.instances]
    df = pd.DataFrame(rows)
    probas = model.predict_proba(df).tolist()

    predictions = []
    for p in probas:
        label = p >= 0.5
        CHURN_PROBABILITY.observe(p)
        PREDICTIONS_TOTAL.labels(churn_label=str(label).lower()).inc()
        predictions.append(PredictResponse(
            churn_probability=p,
            churn_label=label,
            model_name=model.name,
            model_version=model.version,
        ))
    return BatchPredictResponse(predictions=predictions, n=len(predictions))


# ----- Drift endpoint ------------------------------------------------------


@app.post("/drift/score")
def drift_score(body: dict):
    """Score drift between a baseline parquet and a current parquet.

    Request body:
        {
          "baseline_path": "/path/to/train.parquet",
          "current_path":  "/path/to/recent.parquet"
        }

    Both paths must be reachable from inside the serving container. Use the
    bind-mounted /opt/jobs/data path for local files.

    Response: drift report summary + per-feature scores.
    """
    import pandas as pd

    baseline_path = body.get("baseline_path")
    current_path = body.get("current_path")
    if not baseline_path or not current_path:
        raise HTTPException(
            status_code=400,
            detail="Both 'baseline_path' and 'current_path' are required.",
        )

    try:
        baseline_df = pd.read_parquet(baseline_path)
        current_df = pd.read_parquet(current_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read parquet: {e}")

    report = score_drift(baseline_df, current_df)
    return {
        "n_features_checked": len({r.feature for r in report.results}),
        "severe_features": report.severe_features,
        "moderate_features": report.moderate_features,
        "has_severe": report.has_severe,
        "should_retrain": report.has_severe,
        "results": [
            {
                "feature": r.feature,
                "method": r.method,
                "score": round(r.score, 4),
                "pvalue": round(r.pvalue, 4) if r.pvalue is not None else None,
                "severity": r.severity,
            }
            for r in report.results
        ],
    }
