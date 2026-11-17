# Retrain job image: full training stack (XGBoost, Optuna, MLflow) + the package.
# Build from the repo root:  docker build -f docker/retrain/Dockerfile -t churn-retrain:latest .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --uid 1000 appuser
USER appuser

CMD ["python", "-m", "churn_platform.retrain"]
