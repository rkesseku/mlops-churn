"""Synthetic customer churn data generator.

Produces a Parquet file with realistic customer-snapshot rows for training
churn models. The relationships between features and the `churned` label are
deterministic given the seed, but include controlled noise so the resulting
ML problem has a realistic ceiling (~0.87 AUC) rather than being trivially
separable.

Design (see Phase 3 design walkthrough for the full reasoning):
  - 15 features mixing numeric, categorical, boolean, and timestamp types
  - Churn is a logistic combination of weighted features + tenure×contract
    interaction + gaussian noise
  - Base churn rate is calibrated to ~27% (matches Telco-like datasets)
  - Realistic missingness: ~30% of nps_score is null; total_charges is null
    for tenure_months==0
  - Four drift modes (none / covariate / label / concept) for the monitoring
    phase later

Usage (CLI):
    python -m churn_platform.generate.synthesize \
        --rows 1000000 --out data/raw/customers.parquet

Usage (Python):
    from churn_platform.generate.synthesize import generate, GenParams
    df = generate(GenParams(n_rows=10_000, seed=42))
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from churn_platform.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("synthesize")


DriftMode = Literal["none", "covariate", "label", "concept"]


# ----- Generator parameters -------------------------------------------------


@dataclass
class GenParams:
    """All knobs that control the generator.

    Drift modes mutate selected weights/distributions before sampling. Anything
    not mutated stays at its baseline value, so a `covariate`-drifted run still
    has the same churn-from-features causal structure as a `none` run.
    """

    n_rows: int = 1_000_000
    seed: int = 42
    drift_mode: DriftMode = "none"
    # 0.0 = no drift, 1.0 = maximum drift. Most realistic drift is 0.2–0.5.
    drift_strength: float = 0.0

    # Base churn rate target (the logistic intercept is calibrated to hit this).
    base_churn_rate: float = 0.27

    # Tenure distribution: exponential decay with mean ~24 months, capped at 72.
    tenure_mean: float = 24.0
    tenure_max: int = 72

    # Monthly charges: log-normal in dollars, mean ≈ $65.
    charges_log_mu: float = 4.15
    charges_log_sigma: float = 0.45

    # Categorical mix probabilities. Must sum to 1.0 in each tuple.
    contract_probs: tuple[float, float, float] = (0.55, 0.25, 0.20)  # m2m, 1yr, 2yr
    payment_probs: tuple[float, float, float, float] = (0.35, 0.20, 0.25, 0.20)
    internet_probs: tuple[float, float, float] = (0.35, 0.45, 0.20)  # DSL, Fiber, None

    # Feature weights in the churn log-odds. Sign matters; magnitude is in
    # log-odds units. Tuned to give a realistic ~0.87 AUC ceiling.
    w_tenure: float = -0.04        # longer tenure -> less churn
    w_contract_m2m: float = 0.9    # month-to-month is high risk
    w_contract_2yr: float = -1.5   # two-year is low risk
    w_charges: float = 0.012       # higher price -> more churn
    w_support_calls: float = 0.20  # complaints -> churn
    w_logins: float = -0.05        # disengagement -> churn (negative weight on logins)
    w_no_autorenew: float = 1.1    # turning off auto-renew is the smoking gun
    w_low_nps: float = 0.15        # per point below 5
    # Interaction: a new (low-tenure) customer on month-to-month is the highest-
    # risk combination. Trees pick this up; additive models miss it.
    w_interaction_new_m2m: float = 0.8

    # Noise added to the log-odds before sigmoid. Higher = less learnable.
    noise_std: float = 0.6

    # NPS survey response rate; 1 - this fraction will be null.
    nps_response_rate: float = 0.70

    # Reference "now" for the event_timestamp column. Defaults to a fixed
    # date so that same-seed runs produce identical timestamps (essential for
    # reproducibility). Pass an explicit value when you actually want "now".
    reference_time: datetime = field(
        default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )


# ----- Drift application ----------------------------------------------------


def _apply_drift(params: GenParams) -> GenParams:
    """Return a copy of params with drift-mode mutations applied.

    Reasoning by mode:
      - covariate: shift feature distributions (e.g. tenure mean up, charges
        down) without changing the churn weights — P(y|x) stable.
      - label: scale the base churn rate without touching features.
      - concept: change the weight on price; same customers, different reasons
        to churn now.
    """
    if params.drift_mode == "none" or params.drift_strength == 0.0:
        return params

    p = GenParams(**{**params.__dict__})  # shallow copy
    s = params.drift_strength

    if params.drift_mode == "covariate":
        # Customer base ages: more long-tenure, fewer m2m contracts.
        p.tenure_mean = params.tenure_mean * (1.0 + 0.5 * s)
        p.contract_probs = (
            max(0.05, params.contract_probs[0] - 0.3 * s),
            params.contract_probs[1] + 0.15 * s,
            params.contract_probs[2] + 0.15 * s,
        )
        # Renormalize (probabilities must sum to 1).
        total = sum(p.contract_probs)
        p.contract_probs = tuple(x / total for x in p.contract_probs)

    elif params.drift_mode == "label":
        # Macro event: churn rate jumps.
        p.base_churn_rate = min(0.6, params.base_churn_rate * (1.0 + s))

    elif params.drift_mode == "concept":
        # Market becomes more price-sensitive; the weight on charges grows.
        p.w_charges = params.w_charges * (1.0 + 2.0 * s)
        p.w_support_calls = params.w_support_calls * (1.0 + s)

    return p


# ----- Core sampling --------------------------------------------------------


def _sample_features(p: GenParams, rng: np.random.Generator) -> pd.DataFrame:
    """Sample all feature columns. Label is added by `_sample_label`."""
    n = p.n_rows

    # Tenure: exponential, clipped to [0, tenure_max].
    tenure = rng.exponential(scale=p.tenure_mean, size=n).clip(0, p.tenure_max)
    tenure = tenure.round().astype(np.int32)

    # Monthly charges: log-normal.
    monthly = rng.lognormal(mean=p.charges_log_mu, sigma=p.charges_log_sigma, size=n)
    monthly = monthly.clip(15.0, 120.0).round(2)

    # Total charges = tenure * monthly + multiplicative noise.
    # NaN for tenure_months == 0 to mimic the classic Telco dataset quirk.
    total = tenure * monthly * rng.normal(1.0, 0.05, size=n)
    total = total.round(2)
    total = np.where(tenure == 0, np.nan, total)

    # Categorical features.
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n, p=p.contract_probs,
    )
    payment = rng.choice(
        ["echeck", "mailcheck", "banktx", "creditcard"],
        size=n, p=p.payment_probs,
    )
    internet = rng.choice(
        ["DSL", "Fiber", "None"],
        size=n, p=p.internet_probs,
    )

    # Behavioral features.
    # Support calls: poisson, slightly heavier-tailed for show.
    support_calls = rng.poisson(lam=1.2, size=n).astype(np.int32)
    # Logins: poisson around 12 per month.
    logins = rng.poisson(lam=12.0, size=n).astype(np.int32)
    # Average session minutes: gamma (right-skewed).
    session_minutes = rng.gamma(shape=2.0, scale=8.0, size=n).round(2)
    # Days since last login: bigger for disengaged users.
    days_since = rng.exponential(scale=5.0, size=n).clip(0, 90).round().astype(np.int32)

    # Booleans.
    paperless = rng.random(size=n) < 0.6
    auto_renew = rng.random(size=n) < 0.7

    # Discount: most customers have none; some have a moderate discount.
    discount = np.where(
        rng.random(size=n) < 0.3,
        rng.uniform(0.05, 0.5, size=n),
        0.0,
    ).round(2)

    # NPS score 0–10. Realistically null for non-respondents.
    nps_raw = rng.integers(0, 11, size=n).astype(np.float32)
    response_mask = rng.random(size=n) < p.nps_response_rate
    nps = np.where(response_mask, nps_raw, np.nan)

    # Timestamps spread across the last 30 days (just so it's not all "now").
    offsets_seconds = rng.integers(0, 30 * 86400, size=n)
    timestamps = [
        p.reference_time - timedelta(seconds=int(off))
        for off in offsets_seconds
    ]

    return pd.DataFrame({
        "customer_id": [f"C{i:08d}" for i in range(1, n + 1)],
        "tenure_months": tenure,
        "contract_type": contract,
        "monthly_charges": monthly,
        "total_charges": total,
        "payment_method": payment,
        "internet_service": internet,
        "num_support_calls": support_calls,
        "num_logins_30d": logins,
        "avg_session_minutes": session_minutes,
        "days_since_last_login": days_since,
        "has_paperless_billing": paperless,
        "auto_renew": auto_renew,
        "discount_applied": discount,
        "nps_score": nps,
        "event_timestamp": timestamps,
    })


def _sample_label(
    df: pd.DataFrame,
    p: GenParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute churn label from features using a logistic model.

    Returns a boolean array of length len(df).
    """
    n = len(df)

    # Tenure contribution: more negative for higher tenure.
    log_odds = p.w_tenure * df["tenure_months"].to_numpy()

    # Contract contribution.
    contract = df["contract_type"].to_numpy()
    log_odds = log_odds + np.where(contract == "month-to-month", p.w_contract_m2m, 0.0)
    log_odds = log_odds + np.where(contract == "two-year", p.w_contract_2yr, 0.0)

    # Price.
    log_odds = log_odds + p.w_charges * df["monthly_charges"].to_numpy()

    # Behavior.
    log_odds = log_odds + p.w_support_calls * df["num_support_calls"].to_numpy()
    log_odds = log_odds + p.w_logins * df["num_logins_30d"].to_numpy()

    # Auto-renew: contributes only when OFF.
    log_odds = log_odds + p.w_no_autorenew * (~df["auto_renew"].to_numpy()).astype(float)

    # NPS: contributes per point below 5. Null NPS contributes 0.
    nps = df["nps_score"].to_numpy()
    nps_filled = np.where(np.isnan(nps), 5.0, nps)  # neutral imputation
    log_odds = log_odds + p.w_low_nps * np.maximum(0, 5 - nps_filled)

    # Interaction: new + month-to-month is extra-risky.
    new_m2m = (df["tenure_months"].to_numpy() < 6) & (contract == "month-to-month")
    log_odds = log_odds + p.w_interaction_new_m2m * new_m2m.astype(float)

    # Calibrate intercept to hit base_churn_rate.
    # Find the offset that makes mean(sigmoid(log_odds + offset)) ≈ base rate.
    # Cheap iterative solution: bisection on offset in [-5, 5].
    target = p.base_churn_rate
    lo, hi = -5.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        rate = float((1.0 / (1.0 + np.exp(-(log_odds + mid)))).mean())
        if rate < target:
            lo = mid
        else:
            hi = mid
    intercept = (lo + hi) / 2.0
    log_odds = log_odds + intercept

    # Add gaussian noise.
    log_odds = log_odds + rng.normal(0.0, p.noise_std, size=n)

    # Sample.
    probs = 1.0 / (1.0 + np.exp(-log_odds))
    churned = rng.random(size=n) < probs
    return churned


# ----- Public API -----------------------------------------------------------


def generate(params: GenParams) -> pd.DataFrame:
    """Generate a churn dataset as a pandas DataFrame.

    Deterministic given `params.seed`. Returned DataFrame includes `churned`
    as the label column.
    """
    p = _apply_drift(params)
    rng = np.random.default_rng(p.seed)

    log.info(
        "Generating %d rows (drift_mode=%s, drift_strength=%.2f, seed=%d)",
        p.n_rows, p.drift_mode, p.drift_strength, p.seed,
    )
    df = _sample_features(p, rng)
    df["churned"] = _sample_label(df, p, rng)

    actual_rate = float(df["churned"].mean())
    log.info("Generated %d rows; observed churn rate = %.4f", len(df), actual_rate)
    return df


def write_parquet(df: pd.DataFrame, out_path: Path) -> Path:
    """Write the DataFrame to Parquet, creating parent dirs as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    log.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1024 / 1024)
    return out_path


# ----- CLI ------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic customer churn data.",
    )
    parser.add_argument("--rows", type=int, default=settings.default_n_rows)
    parser.add_argument("--seed", type=int, default=settings.default_seed)
    parser.add_argument(
        "--drift",
        choices=["none", "covariate", "label", "concept"],
        default="none",
    )
    parser.add_argument("--drift-strength", type=float, default=0.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.raw_data_dir / "customers.parquet",
    )
    args = parser.parse_args()

    params = GenParams(
        n_rows=args.rows,
        seed=args.seed,
        drift_mode=args.drift,
        drift_strength=args.drift_strength,
    )
    df = generate(params)
    write_parquet(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
