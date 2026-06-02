"""XGBoost training with Optuna hyperparameter search and MLflow tracking.

Usage:
    python -m churn_platform.train.train
    python -m churn_platform.train.train --n-trials 100
    python -m churn_platform.train.train --sample-size 50000  # faster but lower-quality search

After the run:
    - Open http://localhost:5000 to see all trials in MLflow
    - The best model is registered as "churn_xgboost" in the Model Registry
      with stage = "Staging"
    - Final test-set metrics are logged on the parent run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

from churn_platform.config import settings
from churn_platform.train.search_space import DEFAULT_N_TRIALS, suggest_params

log = logging.getLogger("train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


# Columns to exclude from the feature matrix.
# - customer_id: identifier, not predictive
# - event_timestamp: leak risk (we'd use it for time-based splits, not as a feature)
# - churned: the label
# - total_charges, nps_score: replaced by *_filled counterparts
NON_FEATURE_COLS = {
    "customer_id", "event_timestamp", "churned",
    "total_charges", "nps_score",
}

# Columns that should be treated as categorical by XGBoost.
CATEGORICAL_COLS = [
    "contract_type", "payment_method", "internet_service", "tenure_bucket",
]

MODEL_NAME = "churn_xgboost"


# ----- Data loading ---------------------------------------------------------


@dataclass
class Datasets:
    """All three splits in one place. Each .X is a DataFrame, .y a Series."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a single DataFrame into (X, y) and apply dtype conversions.

    Categorical columns must be pandas Categorical dtype for XGBoost's
    enable_categorical=True path to work. Boolean columns are fine as-is.
    """
    y = df["churned"].astype(int)
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y


def load_datasets(features_dir: Path = Path("data/features")) -> Datasets:
    """Load train/val/test parquets and return them as a Datasets bundle."""
    log.info("Loading datasets from %s", features_dir)
    train_df = pd.read_parquet(features_dir / "train.parquet")
    val_df = pd.read_parquet(features_dir / "val.parquet")
    test_df = pd.read_parquet(features_dir / "test.parquet")

    X_train, y_train = _prepare_features(train_df)
    X_val, y_val = _prepare_features(val_df)
    X_test, y_test = _prepare_features(test_df)

    log.info(
        "Loaded train=%d  val=%d  test=%d  features=%d",
        len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )
    return Datasets(X_train, y_train, X_val, y_val, X_test, y_test)


# ----- MLflow setup ---------------------------------------------------------


def configure_mlflow() -> None:
    """Point MLflow at our tracking server and ensure the experiment exists."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # set_experiment creates the experiment if it doesn't exist already.
    mlflow.set_experiment("churn-xgboost")
    log.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)


# ----- Optuna objective -----------------------------------------------------


def make_objective(
    datasets: Datasets,
    sample_size: int,
    seed: int = 42,
):
    """Build the Optuna objective callable.

    We curry the datasets into the closure so Optuna's `study.optimize`
    can call objective(trial) without extra arguments.

    Each trial:
      1. Suggests hyperparameters from search_space.
      2. Subsamples the training data to `sample_size` rows (speed).
      3. Trains XGBoost with early stopping on validation set.
      4. Reports validation AUC to MLflow and back to Optuna.
    """
    # Pre-sample once if sample_size < full size, for reproducibility across trials.
    if sample_size and sample_size < len(datasets.X_train):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(datasets.X_train), size=sample_size, replace=False)
        X_tr = datasets.X_train.iloc[idx]
        y_tr = datasets.y_train.iloc[idx]
    else:
        X_tr = datasets.X_train
        y_tr = datasets.y_train

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        # nested=True: this run becomes a child of the active parent run.
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_param("n_train_rows", len(X_tr))

            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                enable_categorical=True,
                early_stopping_rounds=50,
                random_state=seed,
                verbosity=0,
            )

            t0 = time.perf_counter()
            model.fit(
                X_tr, y_tr,
                eval_set=[(datasets.X_val, datasets.y_val)],
                verbose=False,
            )
            fit_seconds = time.perf_counter() - t0

            val_pred_proba = model.predict_proba(datasets.X_val)[:, 1]
            val_auc = roc_auc_score(datasets.y_val, val_pred_proba)
            val_logloss = log_loss(datasets.y_val, val_pred_proba)

            mlflow.log_metrics({
                "val_auc": val_auc,
                "val_logloss": val_logloss,
                "fit_seconds": fit_seconds,
                "best_iteration": model.best_iteration or model.n_estimators,
            })

            log.info(
                "trial %3d  val_auc=%.4f  logloss=%.4f  best_iter=%d  %.1fs",
                trial.number, val_auc, val_logloss,
                model.best_iteration or model.n_estimators, fit_seconds,
            )
            return val_auc

    return objective


# ----- Final fit + registration ---------------------------------------------


def fit_final_model(
    best_params: dict[str, Any],
    datasets: Datasets,
    seed: int = 42,
) -> XGBClassifier:
    """Re-fit XGBoost with the best params on the FULL training set.

    The Optuna study used a subsample for speed. The model we register and
    deploy should use everything.
    """
    log.info("Fitting final model with best params on full training set (%d rows)",
             len(datasets.X_train))
    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=seed,
        verbosity=0,
    )
    t0 = time.perf_counter()
    model.fit(
        datasets.X_train, datasets.y_train,
        eval_set=[(datasets.X_val, datasets.y_val)],
        verbose=False,
    )
    log.info("Final fit done in %.1fs (best_iteration=%s)",
             time.perf_counter() - t0, model.best_iteration)
    return model


def evaluate_on_test(model: XGBClassifier, datasets: Datasets) -> dict[str, float]:
    """Score the model on the holdout test set. Touched exactly once."""
    test_pred_proba = model.predict_proba(datasets.X_test)[:, 1]
    return {
        "test_auc": float(roc_auc_score(datasets.y_test, test_pred_proba)),
        "test_logloss": float(log_loss(datasets.y_test, test_pred_proba)),
        "test_pr_auc": float(
            average_precision_score(datasets.y_test, test_pred_proba)
        ),
    }


# ----- Orchestration --------------------------------------------------------


def run(n_trials: int, sample_size: int, features_dir: Path) -> None:
    configure_mlflow()
    datasets = load_datasets(features_dir)

    with mlflow.start_run(run_name=f"study_{int(time.time())}") as parent_run:
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_param("n_train_rows_full", len(datasets.X_train))
        mlflow.log_param("n_val_rows", len(datasets.X_val))
        mlflow.log_param("n_test_rows", len(datasets.X_test))
        mlflow.log_param("n_features", datasets.X_train.shape[1])

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f"churn-{int(time.time())}",
        )

        objective = make_objective(datasets, sample_size=sample_size)
        log.info("Starting Optuna study: %d trials", n_trials)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        log.info("Best val AUC: %.4f", study.best_value)
        log.info("Best params: %s", study.best_params)

        mlflow.log_metric("best_val_auc", study.best_value)
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # Refit on full training set, evaluate, register.
        final_model = fit_final_model(study.best_params, datasets)
        test_metrics = evaluate_on_test(final_model, datasets)
        log.info("Test metrics: %s", test_metrics)
        mlflow.log_metrics(test_metrics)

        # Register the model in MLflow Model Registry.
        # We skip MLflow's input_example / auto-signature because XGBoost's
        # categorical columns don't survive the round-trip cleanly. The
        # FastAPI serving layer in Phase 6 will do request validation
        # explicitly via Pydantic, so we don't need MLflow's signature.
        mv = mlflow.xgboost.log_model(
            xgb_model=final_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        log.info("Registered model %s (version derived from run %s)",
                 MODEL_NAME, parent_run.info.run_id)

        # MLflow's "stages" (Staging/Production) are deprecated; use aliases
        # instead. Aliases are mutable named pointers to a specific version:
        # "staging" -> v3 today, v5 tomorrow. Production promotion just
        # repoints "production" to a new version, leaving the old one queryable.
        client = mlflow.tracking.MlflowClient()
        version = mv.registered_model_version
        client.set_registered_model_alias(MODEL_NAME, "staging", version)
        log.info("Set alias %s@staging -> v%s", MODEL_NAME, version)


# ----- CLI ------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Train churn XGBoost via Optuna + MLflow.")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Rows per trial during the Optuna study. Full training is used for the final fit.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/features"),
    )
    args = parser.parse_args()
    run(
        n_trials=args.n_trials,
        sample_size=args.sample_size,
        features_dir=args.features_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
