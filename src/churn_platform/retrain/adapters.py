"""Real MLflow/XGBoost backend for the retrain orchestrator.

Heavy imports (mlflow, xgboost, sklearn, train.train) are done lazily inside
methods so the orchestrator and its tests import without them.

Registry convention:
  - candidate gets alias `candidate` while being evaluated
  - champion = alias `production`, falling back to `staging` (what the serving
    app loads today) if `production` has never been set
  - promotion repoints BOTH `production` and `staging` to the new version, so
    serving picks it up on its next (re)start; rejected versions are tagged
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import subprocess
from pathlib import Path

log = logging.getLogger("retrain.adapters")

MODEL_NAME = "churn_xgboost"


class MlflowBackend:
    def __init__(
        self,
        features_dir: Path,
        n_trials: int,
        sample_size: int,
        features_cmd: str = "",
        model_name: str = MODEL_NAME,
        holdout_split: str = "gate",
    ) -> None:
        if holdout_split == "test":
            raise ValueError(
                "the promotion gate must not score on the final test split; "
                "use the dedicated 'gate' split written by the features job"
            )
        self.holdout_split = holdout_split
        self.features_dir = Path(features_dir)
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.features_cmd = features_cmd
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import mlflow
            from mlflow.tracking import MlflowClient

            from churn_platform.config import settings

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            self._client = MlflowClient()
        return self._client

    def champion_version(self) -> str | None:
        for alias in ("production", "staging"):
            try:
                return str(self.client.get_model_version_by_alias(self.model_name, alias).version)
            except Exception:
                continue
        return None

    def train_candidate(self) -> str:
        if self.features_cmd:
            log.info("Refreshing features: %s", self.features_cmd)
            subprocess.run(shlex.split(self.features_cmd), check=True)
        from churn_platform.train.train import run as train_run

        version = str(
            train_run(
                n_trials=self.n_trials,
                sample_size=self.sample_size,
                features_dir=self.features_dir,
                set_staging_alias=False,  # don't touch what serving loads
            )
        )
        self.client.set_registered_model_alias(self.model_name, "candidate", version)
        return version

    def holdout_predictions(self, version: str):
        """(y, proba) for a registered version on the gate holdout
        (features_dir/<holdout_split>.parquet). Training never sees it, and it is
        separate from the final test split, which train.run touches once."""
        import mlflow.xgboost
        import pandas as pd

        from churn_platform.train.train import _prepare_features

        path = self.features_dir / f"{self.holdout_split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; re-run the features job, which writes a "
                f"'{self.holdout_split}' split for the promotion gate"
            )
        model = mlflow.xgboost.load_model(f"models:/{self.model_name}/{version}")
        X, y = _prepare_features(pd.read_parquet(path))
        with contextlib.suppress(Exception):
            X = X[list(model.get_booster().feature_names)]
        return y.to_numpy(), model.predict_proba(X)[:, 1]

    def promote(self, version: str) -> None:
        for alias in ("production", "staging"):
            self.client.set_registered_model_alias(self.model_name, alias, version)
        self.client.set_model_version_tag(self.model_name, version, "retrain_decision", "promoted")
        log.info("Promoted %s v%s", self.model_name, version)

    def reject(self, version: str) -> None:
        self.client.set_model_version_tag(self.model_name, version, "retrain_decision", "rejected")
        log.info("Rejected candidate %s v%s", self.model_name, version)
