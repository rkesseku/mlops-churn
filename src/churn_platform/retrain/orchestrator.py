"""Drift-triggered retraining orchestrator.

One cycle:
  1. load reference + a recent window of current scoring data
  2. run the drift check (monitor.drift.score_drift)
  3. if drift is severe: train a candidate (reusing train.train), register it
     in MLflow, score candidate AND champion on the same gate holdout (a
     dedicated split, not the final test split)
  4. promote only if the lower bound of a paired-bootstrap CI for the
     improvement clears the margin
  5. write a JSON decision record (always) and optionally push Prometheus metrics

Idempotency: each cycle derives a `run_id` from the data window fingerprints,
the decision-relevant config. If a non-error record
with that id already exists, the cycle is skipped, so re-running a CronJob on
unchanged data does nothing.

All MLflow/XGBoost access lives behind the `Backend` protocol (see
`adapters.py`), so this module and `decision.py` are testable with fakes.

Usage:
    python -m churn_platform.retrain --reference data/raw/customers.parquet \
        --current data/raw/customers_drifted.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from churn_platform.retrain.decision import (
    DecisionRecord,
    RetrainConfig,
    decide_drift,
    decide_promotion,
    holdout_metrics,
    paired_bootstrap_improvement,
)

log = logging.getLogger("retrain")


class Backend(Protocol):
    def champion_version(self) -> str | None: ...
    def train_candidate(self) -> str: ...
    def holdout_predictions(self, version: str) -> tuple[np.ndarray, np.ndarray]:
        """(labels, predicted churn probability) on the gate holdout."""
        ...
    def promote(self, version: str) -> None: ...
    def reject(self, version: str) -> None: ...


class DecisionStore:
    """One JSON file per run_id in a directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.json"

    def get(self, run_id: str) -> dict[str, Any] | None:
        p = self.path(run_id)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, record: DecisionRecord) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        p = self.path(record.run_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True, default=str))
        tmp.replace(p)  # atomic: a crash never leaves a half-written record
        return p


# ----- data windowing -------------------------------------------------------


def select_recent_window(
    df: pd.DataFrame,
    days: int | None,
    max_rows: int | None,
    now: datetime,
    ts_col: str = "event_timestamp",
    seed: int = 42,
) -> pd.DataFrame:
    """Keep rows from the last `days` days (if a timestamp column exists), then
    cap at `max_rows` with a seeded sample so the result is reproducible."""
    out = df
    if days and ts_col in out.columns:
        ts = pd.to_datetime(out[ts_col])
        cutoff = pd.Timestamp(now)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        cutoff = cutoff - pd.Timedelta(days=days)
        if ts.dt.tz is None:
            cutoff = cutoff.tz_localize(None)
        out = out[ts >= cutoff]
    if max_rows and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return out.reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(str(len(df)).encode())
    if len(df):
        h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def compute_run_id(reference_fp: str, current_fp: str, cfg: RetrainConfig) -> str:
    """Identity of a decision: the data windows plus decision-relevant config.

    The champion version is deliberately NOT part of the id: after a promotion
    the champion changes, and re-running on the same data must not retrain again.
    """
    payload = json.dumps(
        [reference_fp, current_fp, cfg.min_severe_features,
         cfg.promotion_margin, cfg.metric, cfg.force, cfg.dry_run,
         cfg.n_bootstrap, cfg.ci_level, cfg.bootstrap_seed],
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ----- the cycle ------------------------------------------------------------


def run_cycle(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: RetrainConfig,
    backend: Backend,
    store: DecisionStore,
    drift_fn: Callable[[pd.DataFrame, pd.DataFrame], Any] | None = None,
    now: datetime | None = None,
) -> DecisionRecord:
    """Execute one decision cycle. Persists a record for every outcome, except
    that a completed record for identical inputs short-circuits to `duplicate`."""
    if drift_fn is None:
        from churn_platform.monitor.drift import score_drift as drift_fn
    now = now or datetime.now(timezone.utc)

    champion = backend.champion_version()
    run_id = compute_run_id(fingerprint(reference), fingerprint(current), cfg)
    base = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "champion_version": champion,
        "config": asdict(cfg),
    }

    existing = store.get(run_id)
    if existing is not None and existing.get("status") != "error":
        log.info("Run %s already recorded (status=%s); skipping", run_id, existing["status"])
        return DecisionRecord(**{
            **existing,
            "status": "duplicate",
            "detail": "identical inputs already handled "
                      f"(original status: {existing['status']})",
        })

    if len(current) < cfg.min_rows or len(reference) < cfg.min_rows:
        rec = DecisionRecord(
            **base, status="insufficient_data",
            detail=f"need >= {cfg.min_rows} rows; "
                   f"reference={len(reference)} current={len(current)}",
        )
        store.put(rec)
        return rec

    report = drift_fn(reference, current)
    drift = decide_drift(report, cfg)
    drift_dict = asdict(drift)
    log.info("Drift decision: %s (%s)", drift.should_retrain, drift.reason)

    if not drift.should_retrain:
        rec = DecisionRecord(**base, status="no_drift", drift=drift_dict, detail=drift.reason)
        store.put(rec)
        return rec

    if cfg.dry_run:
        rec = DecisionRecord(
            **base, status="dry_run", drift=drift_dict,
            detail="drift gate passed; would retrain (dry run, nothing trained)",
        )
        store.put(rec)
        return rec

    candidate: str | None = None
    try:
        candidate = backend.train_candidate()
        promo = _compare(backend, candidate, champion, cfg)
        if promo.promote:
            backend.promote(candidate)
        else:
            backend.reject(candidate)
    except Exception as exc:
        log.exception("Retrain cycle failed")
        rec = DecisionRecord(
            **base, status="error", drift=drift_dict, candidate_version=candidate,
            detail=f"{type(exc).__name__}: {exc}",
        )
        store.put(rec)
        return rec

    rec = DecisionRecord(
        **base, status="promoted" if promo.promote else "rejected",
        drift=drift_dict, promotion=asdict(promo), candidate_version=candidate,
        detail=promo.reason,
    )
    store.put(rec)
    return rec


def _compare(backend: Backend, candidate: str, champion: str | None, cfg: RetrainConfig):
    """Score both models on the gate holdout and apply the promotion rule."""
    y, p_cand = backend.holdout_predictions(candidate)
    y, p_cand = np.asarray(y), np.asarray(p_cand)
    cand_value = holdout_metrics(y, p_cand)[cfg.metric]
    if champion is None:
        return decide_promotion(cand_value, None, cfg, n_holdout=len(y))
    y_champ, p_champ = backend.holdout_predictions(champion)
    if not np.array_equal(y, np.asarray(y_champ)):
        raise ValueError("candidate and champion were scored on different holdout rows")
    p_champ = np.asarray(p_champ)
    champ_value = holdout_metrics(y, p_champ)[cfg.metric]
    ci = None
    if cfg.n_bootstrap > 0:
        _, lo, hi = paired_bootstrap_improvement(
            y, p_cand, p_champ, cfg.metric, n_boot=cfg.n_bootstrap, ci_level=cfg.ci_level,
            seed=cfg.bootstrap_seed,
        )
        ci = (lo, hi)
    return decide_promotion(cand_value, champ_value, cfg, ci=ci, n_holdout=len(y))


# ----- CLI ------------------------------------------------------------------


def _env(name: str, default: Any) -> Any:
    return os.environ.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m churn_platform.retrain",
        description="Check drift and, if severe, retrain and conditionally promote.",
    )
    p.add_argument("--reference", type=Path, default=_env("RETRAIN_REFERENCE_PATH", None),
                   help="Baseline (training-time) raw parquet.")
    p.add_argument("--current", type=Path, default=_env("RETRAIN_CURRENT_PATH", None),
                   help="Recent scoring-data raw parquet.")
    p.add_argument("--features-dir", type=Path,
                   default=Path(_env("RETRAIN_FEATURES_DIR", "data/features")))
    p.add_argument("--decision-dir", type=Path,
                   default=Path(_env("RETRAIN_DECISION_DIR", "data/retrain_decisions")))
    p.add_argument("--window-days", type=int, default=int(_env("RETRAIN_WINDOW_DAYS", 30)))
    p.add_argument("--max-rows", type=int, default=int(_env("RETRAIN_MAX_ROWS", 200_000)))
    p.add_argument("--min-rows", type=int, default=int(_env("RETRAIN_MIN_ROWS", 1_000)))
    p.add_argument("--min-severe-features", type=int,
                   default=int(_env("RETRAIN_MIN_SEVERE_FEATURES", 1)))
    p.add_argument("--promotion-margin", type=float,
                   default=float(_env("RETRAIN_PROMOTION_MARGIN", 0.002)))
    p.add_argument("--metric", default=_env("RETRAIN_METRIC", "auc"),
                   choices=["auc", "pr_auc", "logloss"])
    p.add_argument("--holdout-split", default=_env("RETRAIN_HOLDOUT_SPLIT", "gate"),
                   help="Features split used by the promotion gate (<features-dir>/<name>"
                        ".parquet). Must not be the final 'test' split.")
    p.add_argument("--n-bootstrap", type=int,
                   default=int(_env("RETRAIN_N_BOOTSTRAP", 1_000)),
                   help="Paired-bootstrap draws for the improvement CI; 0 = point estimate.")
    p.add_argument("--ci-level", type=float, default=float(_env("RETRAIN_CI_LEVEL", 0.95)))
    p.add_argument("--n-trials", type=int, default=int(_env("RETRAIN_N_TRIALS", 30)))
    p.add_argument("--sample-size", type=int, default=int(_env("RETRAIN_SAMPLE_SIZE", 100_000)))
    p.add_argument("--features-cmd", default=_env("RETRAIN_FEATURES_CMD", ""),
                   help="Optional command (no shell) run after the drift gate and before "
                        "training, e.g. a spark-submit that refreshes --features-dir.")
    p.add_argument("--pushgateway-url", default=_env("PUSHGATEWAY_URL", ""))
    p.add_argument("--force", action="store_true", help="Retrain even without severe drift.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the drift gate only; never train or promote.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if not args.reference or not args.current:
        print("error: --reference and --current (or RETRAIN_REFERENCE_PATH / "
              "RETRAIN_CURRENT_PATH) are required", file=sys.stderr)
        return 2

    cfg = RetrainConfig(
        min_severe_features=args.min_severe_features,
        promotion_margin=args.promotion_margin,
        metric=args.metric,
        window_days=args.window_days or None,
        max_rows=args.max_rows or None,
        min_rows=args.min_rows,
        force=args.force,
        dry_run=args.dry_run,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
    )
    now = datetime.now(timezone.utc)
    reference = select_recent_window(pd.read_parquet(args.reference), None, cfg.max_rows, now)
    current = select_recent_window(
        pd.read_parquet(args.current), cfg.window_days, cfg.max_rows, now
    )

    from churn_platform.retrain.adapters import MlflowBackend  # lazy: heavy deps

    backend = MlflowBackend(
        features_dir=args.features_dir,
        n_trials=args.n_trials,
        sample_size=args.sample_size,
        features_cmd=args.features_cmd,
        holdout_split=args.holdout_split,
    )
    record = run_cycle(reference, current, cfg, backend, DecisionStore(args.decision_dir), now=now)
    print(json.dumps(record.to_dict(), indent=2, default=str))

    if args.pushgateway_url:
        from churn_platform.retrain.metrics import RetrainMetrics

        m = RetrainMetrics()
        m.observe(record.to_dict(), time.time())
        m.push(args.pushgateway_url)
    return 1 if record.status == "error" else 0
