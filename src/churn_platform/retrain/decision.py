"""Pure decision logic for the retrain loop. No MLflow / Spark / network imports.

Two decisions are made per cycle:

1. Drift gate: is drift severe enough to justify a retrain?
   -> at least `min_severe_features` distinct features with severity "severe"
      (PSI > 0.25, or a KS test that is both significant and large; see
      `churn_platform.monitor.drift`).
2. Promotion gate (champion/challenger): does the candidate beat the current
   champion on the SAME holdout by at least `promotion_margin` on `metric`?
   -> if there is no champion at all, the candidate is promoted (bootstrap).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

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

    def __post_init__(self) -> None:
        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}")
        if self.promotion_margin < 0:
            raise ValueError("promotion_margin must be >= 0")
        if self.min_severe_features < 1:
            raise ValueError("min_severe_features must be >= 1")


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


def decide_promotion(
    candidate_value: float,
    champion_value: float | None,
    cfg: RetrainConfig,
) -> PromotionDecision:
    if champion_value is None:
        return PromotionDecision(
            True, cfg.metric, None, candidate_value, None, cfg.promotion_margin,
            "no champion exists; promoting candidate",
        )
    if cfg.metric in HIGHER_IS_BETTER:
        improvement = candidate_value - champion_value
    else:
        improvement = champion_value - candidate_value
    promote = improvement >= cfg.promotion_margin
    verb = "beats" if promote else "does not beat"
    return PromotionDecision(
        promote, cfg.metric, champion_value, candidate_value, improvement,
        cfg.promotion_margin,
        f"candidate {verb} champion by {improvement:+.5f} (required >= {cfg.promotion_margin})",
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
