# Slim serving image for the churn FastAPI app.
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

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
        "uvicorn[standard]==0.32.1" \
        prometheus-client==0.21.0 \
        prometheus-fastapi-instrumentator==7.0.0

COPY src/churn_platform /app/churn_platform

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "churn_platform.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
