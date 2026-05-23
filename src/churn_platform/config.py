"""Centralized settings for the churn platform.

Every module reads its configuration from `settings`, never from raw env vars
or hard-coded paths. This makes the platform reconfigurable for different
environments (local dev, CI, production) by changing environment variables —
no code changes required.

Usage:
    from churn_platform.config import settings
    print(settings.minio_endpoint)

Environment variables override defaults. For local dev, put them in a `.env`
file at the repo root; Pydantic will load it automatically.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, resolved from this file's location: src/churn_platform/config.py
# -> parents[2] climbs up: churn_platform -> src -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration. Values can be overridden via env vars
    (e.g. setting `MLFLOW_TRACKING_URI=http://prod.example.com:5000` in the
    environment overrides the default below)."""

    # ----- Paths --------------------------------------------------------------
    # Where generated data lives. Gitignored; safe to wipe and regenerate.
    data_dir: Path = REPO_ROOT / "data"
    raw_data_dir: Path = REPO_ROOT / "data" / "raw"
    features_dir: Path = REPO_ROOT / "data" / "features"
    samples_dir: Path = REPO_ROOT / "data" / "samples"

    # ----- Data generation ----------------------------------------------------
    default_n_rows: int = 1_000_000
    # Fixed seed makes the dataset reproducible across machines and CI runs.
    # Change only if you deliberately want a different dataset.
    default_seed: int = 42

    # ----- MLflow -------------------------------------------------------------
    # When running locally, MLflow is at localhost:5000.
    # When running from inside another Docker container, it's at http://mlflow:5000
    # (the docker-compose service name).
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )

    # ----- MinIO (S3-compatible) ---------------------------------------------
    # These are the same credentials we set in docker-compose.yml.
    # In production, these would come from a secrets manager.
    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket_artifacts: str = Field(default="mlflow-artifacts")
    minio_bucket_raw: str = Field(default="raw-data")
    minio_bucket_features: str = Field(default="features")

    # ----- Postgres -----------------------------------------------------------
    # Used by MLflow as its metadata backend. Most app code shouldn't touch this
    # directly — go through MLflow's Python API instead.
    postgres_url: str = Field(
        default="postgresql://mlflow:mlflow@localhost:5432/mlflow",
    )

    # Pydantic settings config: read from .env file at repo root if present.
    # extra="ignore" means unknown env vars don't crash us.
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton. Import this everywhere; don't instantiate Settings()
# again, or you'll re-parse env vars unnecessarily.
settings = Settings()
