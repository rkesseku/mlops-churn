"""Pure decision logic for the retrain loop. No MLflow / Spark / network imports.

Two decisions are made per cycle:

1. Drift gate: is drift severe enough to justify a retrain?
   -> at least `min_severe_features` distinct features with severity "severe"
      (PSI > 0.25, or a KS test that is both significant and large; see
      `churn_platform.monitor.drift`).
2. Promotion gate (champion/challenger): candidate and champion are scored on
   the SAME dedicated gate holdout (never the final test split). A paired
   bootstrap over the holdout rows gives a confidence interval for the
   sign-corrected metric difference, and the candidate is promoted only if the
   LOWER bound clears `promotion_margin` (set `n_bootstrap=0` to fall back to
   the point estimate).
   -> if there is no champion at all, the candidate is promoted.

The metric functions here are small numpy/scipy implementations so this module
stays free of MLflow, Spark and network imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.stats import rankdata

# Metrics where larger is better. Anything else (e.g. logloss) is lower-is-better.
HIGHER_IS_BETTER = {"auc", "pr_auc"}
SUPPORTED_METRICS = HIGHER_IS_BETTER | {"logloss"}


@dataclass(frozen=True)
class RetrainConfig:
    min_severe_features: int = 1
    promotion_margin: float = 0.002  # absolute improvement required on `metric`
    metric: str = "auc"
    window_days: int | None = 30
    max_rows: int | None = 200_000
    min_rows: int = 1_000
    force: bool = False
    dry_run: bool = False
    # Paired bootstrap for the candidate-minus-champion difference on the gate
    # holdout. Promotion needs the lower bound of the `ci_level` interval to be
    # >= promotion_margin. n_bootstrap=0 disables the interval (point estimate).
    n_bootstrap: int = 1_000
    ci_level: float = 0.95
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}")
        if self.promotion_margin < 0:
            raise ValueError("promotion_margin must be >= 0")
        if self.min_severe_features < 1:
            raise ValueError("min_severe_features must be >= 1")
        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be >= 0")
        if not 0.5 <= self.ci_level < 1:
            raise ValueError("ci_level must be in [0.5, 1)")


@dataclass(frozen=True)
class DriftDecision:
    should_retrain: bool
    severe_features: list[str]
    moderate_features: list[str]
    reason: str


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    metric: str
    champion_value: float | None
    candidate_value: float
    improvement: float | None  # positive = candidate is better, sign-corrected
    margin: float
    reason: str
    ci_lower: float | None = None  # paired-bootstrap interval for `improvement`
    ci_upper: float | None = None
    ci_level: float | None = None
    n_holdout: int | None = None


def decide_drift(report: Any, cfg: RetrainConfig) -> DriftDecision:
    """`report` is a `monitor.drift.DriftReport` (duck-typed)."""
    severe = sorted(set(report.severe_features))
    moderate = sorted(set(report.moderate_features) - set(severe))
    if cfg.force:
        return DriftDecision(True, severe, moderate, "forced by configuration")
    if len(severe) >= cfg.min_severe_features:
        return DriftDecision(
            True, severe, moderate,
            f"{len(severe)} severe feature(s) >= threshold {cfg.min_severe_features}",
        )
    return DriftDecision(
        False, severe, moderate,
        f"{len(severe)} severe feature(s) < threshold {cfg.min_severe_features}",
    )


# ----- holdout metrics and the paired bootstrap ------------------------------


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (step-wise PR area, as in sklearn), ties grouped."""
    y = np.asarray(y).astype(float)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    p = np.asarray(p)
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]  # end of each tie group
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def logloss_score(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


METRIC_FNS = {"auc": auc_score, "pr_auc": pr_auc_score, "logloss": logloss_score}


def holdout_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {name: fn(y, p) for name, fn in METRIC_FNS.items()}


def improvement_of(candidate_value: float, champion_value: float, metric: str) -> float:
    """Sign-corrected difference: positive means the candidate is better."""
    if metric in HIGHER_IS_BETTER:
        return candidate_value - champion_value
    return champion_value - candidate_value


def paired_bootstrap_improvement(
    y: np.ndarray,
    p_candidate: np.ndarray,
    p_champion: np.ndarray,
    metric: str,
    n_boot: int = 1_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of the sign-corrected improvement.

    Both models are scored on the SAME resampled rows in every draw (paired),
    so the interval reflects the uncertainty of the difference, not of each
    model separately. Draws containing a single class are skipped.
    """
    y = np.asarray(y)
    p_candidate, p_champion = np.asarray(p_candidate), np.asarray(p_champion)
    if not (len(y) == len(p_candidate) == len(p_champion)):
        raise ValueError("y and both prediction vectors must have the same length")
    fn = METRIC_FNS[metric]
    point = improvement_of(fn(y, p_candidate), fn(y, p_champion), metric)
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        diffs.append(improvement_of(fn(yb, p_candidate[idx]), fn(yb, p_champion[idx]), metric))
    if not diffs:
        raise ValueError("no usable bootstrap draws (holdout has a single class)")
    a = (1 - ci_level) / 2
    lo, hi = np.quantile(diffs, [a, 1 - a])
    return float(point), float(lo), float(hi)


def decide_promotion(
    candidate_value: float,
    champion_value: float | None,
    cfg: RetrainConfig,
    ci: tuple[float, float] | None = None,
    n_holdout: int | None = None,
) -> PromotionDecision:
    """Promote if the improvement (or, when `ci` is given, its lower bound) >= margin."""
    if champion_value is None:
        return PromotionDecision(
            True, cfg.metric, None, candidate_value, None, cfg.promotion_margin,
            "no champion exists; promoting candidate", n_holdout=n_holdout,
        )
    improvement = improvement_of(candidate_value, champion_value, cfg.metric)
    if ci is None:
        promote = improvement >= cfg.promotion_margin
        verb = "beats" if promote else "does not beat"
        return PromotionDecision(
            promote, cfg.metric, champion_value, candidate_value, improvement,
            cfg.promotion_margin,
            f"candidate {verb} champion by {improvement:+.5f} "
            f"(required >= {cfg.promotion_margin}; point estimate, no interval)",
            n_holdout=n_holdout,
        )
    lo, hi = ci
    promote = lo >= cfg.promotion_margin
    verb = "clears" if promote else "does not clear"
    pct = round(100 * cfg.ci_level)
    return PromotionDecision(
        promote, cfg.metric, champion_value, candidate_value, improvement,
        cfg.promotion_margin,
        f"improvement {improvement:+.5f}, {pct}% CI [{lo:+.5f}, {hi:+.5f}]; lower bound "
        f"{verb} margin {cfg.promotion_margin}",
        ci_lower=lo, ci_upper=hi, ci_level=cfg.ci_level, n_holdout=n_holdout,
    )


@dataclass
class DecisionRecord:
    """JSON-serialisable audit record, one file per run_id."""

    run_id: str
    timestamp: str
    status: str  # no_drift | insufficient_data | dry_run | promoted | rejected | duplicate | error
    drift: dict[str, Any] | None = None
    promotion: dict[str, Any] | None = None
    champion_version: str | None = None
    candidate_version: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
