# Slim serving image for the churn FastAPI app.
# Intentionally narrower than the dev env: only the deps needed to load
# the model from MLflow and serve predictions over HTTP.

FROM python:3.10-slim-bookworm

# Don't write .pyc files; flush stdout immediately so docker logs are live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: curl is for the healthcheck, libgomp for XGBoost's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Pin every package — reproducible builds beat the latest-and-greatest.
# These versions mirror what's in pyproject.toml for the matching libraries.
RUN pip install --no-cache-dir \
        numpy==2.1.3 \
        pandas==2.2.3 \
        pyarrow==17.0.0 \
        pydantic==2.9.2 \
        pydantic-settings==2.6.1 \
        scikit-learn==1.5.2 \
        xgboost==2.1.2 \
        mlflow-skinny==2.17.2 \
        boto3==1.35.60 \
        fastapi==0.115.5 \
        "uvicorn[standard]==0.32.1"

# Copy only the package source. The MLflow artifact (the model itself) is
# pulled from the registry at startup — we never bake it into the image.
COPY src/churn_platform /app/churn_platform

# Run as a non-root user. Standard hardening: even if the container is
# compromised, the attacker doesn't get root.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# uvicorn defaults are fine for local dev; production would tune --workers.
CMD ["uvicorn", "churn_platform.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
