"""FastAPI app serving churn predictions.

Endpoints:
  GET  /            — landing page with links
  GET  /health      — liveness + model version
  POST /predict     — single-row prediction
  POST /predict_batch — many rows in one call
  GET  /metrics     — Prometheus exposition format (wired in Phase 7)

The model is loaded once at startup via the lifespan context manager,
then shared across all requests via app.state.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks. Load model at startup; nothing to clean at shutdown."""
    log.info("Starting churn-serving")
    try:
        app.state.loaded = load()
        log.info("Model loaded successfully")
    except Exception as e:
        # Don't crash the app — let /health report degraded so ops can see it.
        log.error("Failed to load model on startup: %s", e)
        app.state.loaded = None
    yield
    log.info("Shutting down churn-serving")


app = FastAPI(
    title="Churn Prediction Service",
    description="Predicts customer churn probability using an XGBoost model "
                "loaded from the MLflow registry.",
    version="1.0.0",
    lifespan=lifespan,
)


# ----- Dependency: model access ---------------------------------------------


def get_model(request: Request) -> LoadedModel:
    """Inject the loaded model into endpoints. 503 if loading failed."""
    loaded: LoadedModel | None = request.app.state.loaded
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health and serving logs.",
        )
    return loaded


# ----- Endpoints ------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h1>Churn Prediction Service</h1>
    <ul>
      <li><a href="/docs">Swagger UI</a></li>
      <li><a href="/health">Health check</a></li>
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
    return PredictResponse(
        churn_probability=proba,
        churn_label=proba >= 0.5,
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
    predictions = [
        PredictResponse(
            churn_probability=p,
            churn_label=p >= 0.5,
            model_name=model.name,
            model_version=model.version,
        )
        for p in probas
    ]
    return BatchPredictResponse(predictions=predictions, n=len(predictions))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus exposition. Stub for now; wired in Phase 7."""
    return "# Prometheus metrics will be exposed here in Phase 7.\n"
