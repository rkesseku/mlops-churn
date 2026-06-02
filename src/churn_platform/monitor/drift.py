"""Distribution drift detection for tabular features.

Two complementary methods:

  - PSI (Population Stability Index): bin-based. Robust, well-known in
    industry. Thresholds (rule of thumb):
      < 0.1   no significant drift
      0.1-0.25 moderate drift, monitor
      > 0.25   severe drift, retrain candidate

  - KS test: non-parametric two-sample test for continuous features.
    Returns a p-value; low p-value means the two distributions differ.

Workflow:
  - At training time, save a baseline summary of the training feature
    distributions (`compute_baseline`).
  - At monitoring time, score the current production sample against the
    baseline (`score_drift`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Features we don't drift-check: identifiers, timestamps, and labels.
SKIP_COLUMNS = {"customer_id", "event_timestamp", "churned"}

# Default PSI bin count. 10 is the standard banking default.
DEFAULT_BINS = 10


@dataclass
class DriftResult:
    feature: str
    method: Literal["psi", "ks"]
    score: float        # PSI value, or KS statistic
    pvalue: float | None  # only meaningful for KS
    severity: Literal["none", "moderate", "severe"]


@dataclass
class DriftReport:
    results: list[DriftResult] = field(default_factory=list)

    @property
    def severe_features(self) -> list[str]:
        return [r.feature for r in self.results if r.severity == "severe"]

    @property
    def moderate_features(self) -> list[str]:
        return [r.feature for r in self.results if r.severity == "moderate"]

    @property
    def has_severe(self) -> bool:
        return len(self.severe_features) > 0

    def summary(self) -> str:
        n = len(self.results)
        n_sev = len(self.severe_features)
        n_mod = len(self.moderate_features)
        lines = [
            f"Drift report: {n} features checked",
            f"  severe:    {n_sev} -> {', '.join(self.severe_features) if n_sev else '(none)'}",
            f"  moderate:  {n_mod} -> {', '.join(self.moderate_features) if n_mod else '(none)'}",
            f"  unchanged: {n - n_sev - n_mod}",
        ]
        return "\n".join(lines)


# ----- PSI (numeric features) ------------------------------------------------


def _psi(baseline: np.ndarray, current: np.ndarray, bins: int = DEFAULT_BINS) -> float:
    """Population Stability Index between two numeric distributions.

    Builds quantile bins on the baseline (so each bin has roughly equal mass
    in baseline), then scores how the current sample distributes across those
    bins. Returns a non-negative number; 0 = identical, larger = more drift.
    """
    # Quantile-based binning of baseline.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(baseline, quantiles)
    # Make edges strictly increasing (handle constant features).
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    edges = np.unique(edges)
    if len(edges) < 3:
        # Degenerate feature (nearly constant). No meaningful PSI.
        return 0.0

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    # Convert to proportions, with a tiny epsilon to avoid log(0).
    eps = 1e-6
    bp = (baseline_counts / max(baseline_counts.sum(), 1)) + eps
    cp = (current_counts / max(current_counts.sum(), 1)) + eps

    return float(np.sum((cp - bp) * np.log(cp / bp)))


# ----- KS test (continuous features) -----------------------------------------


def _ks(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    if len(baseline) < 2 or len(current) < 2:
        return 0.0, 1.0
    res = ks_2samp(baseline, current)
    return float(res.statistic), float(res.pvalue)


# ----- Categorical drift -----------------------------------------------------


def _categorical_psi(baseline: pd.Series, current: pd.Series) -> float:
    """PSI for categorical / boolean features. Same formula, but the bins
    are the unique categories instead of quantile cuts."""
    all_categories = pd.Index(set(baseline.dropna().unique()) | set(current.dropna().unique()))
    bp = (baseline.value_counts(normalize=True).reindex(all_categories).fillna(0) + 1e-6)
    cp = (current.value_counts(normalize=True).reindex(all_categories).fillna(0) + 1e-6)
    return float(np.sum((cp - bp) * np.log(cp / bp)))


# ----- Top-level scoring -----------------------------------------------------


def _classify(psi: float) -> Literal["none", "moderate", "severe"]:
    if psi < 0.10:
        return "none"
    if psi < 0.25:
        return "moderate"
    return "severe"


def score_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    bins: int = DEFAULT_BINS,
) -> DriftReport:
    """Score every shared column for drift.

    Numeric columns -> PSI + KS. Categorical / boolean -> categorical PSI.
    Skips identifiers, timestamps, and the label.
    """
    results: list[DriftResult] = []
    shared = set(baseline.columns) & set(current.columns) - SKIP_COLUMNS

    for col in sorted(shared):
        b = baseline[col].dropna()
        c = current[col].dropna()
        if len(b) == 0 or len(c) == 0:
            continue

        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            psi = _psi(b.to_numpy(), c.to_numpy(), bins=bins)
            ks_stat, ks_p = _ks(b.to_numpy(), c.to_numpy())
            results.append(DriftResult(
                feature=col, method="psi",
                score=psi, pvalue=None,
                severity=_classify(psi),
            ))
            results.append(DriftResult(
                feature=col, method="ks",
                score=ks_stat, pvalue=ks_p,
                # KS severity inferred from p-value: a very low p with a
                # meaningful KS stat is at least moderate drift.
                severity=(
                    "severe" if (ks_p < 0.001 and ks_stat > 0.1)
                    else "moderate" if (ks_p < 0.05 and ks_stat > 0.05)
                    else "none"
                ),
            ))
        else:
            psi = _categorical_psi(b, c)
            results.append(DriftResult(
                feature=col, method="psi",
                score=psi, pvalue=None,
                severity=_classify(psi),
            ))

    return DriftReport(results=results)
