"""Unit tests for the retrain decision logic and orchestrator.

No MLflow, XGBoost or Spark needed: the backend is a fake and drift reports
are stubbed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from churn_platform.retrain.decision import (
    RetrainConfig,
    auc_score,
    decide_drift,
    decide_promotion,
    holdout_metrics,
    logloss_score,
    paired_bootstrap_improvement,
    pr_auc_score,
)
from churn_platform.retrain.metrics import RetrainMetrics
from churn_platform.retrain.orchestrator import (
    DecisionStore,
    compute_run_id,
    fingerprint,
    run_cycle,
    select_recent_window,
)

NOW = datetime(2026, 1, 31, tzinfo=timezone.utc)


@dataclass
class FakeReport:
    severe_features: list[str]
    moderate_features: list[str]


_RNG = np.random.default_rng(123)
GATE_Y = _RNG.integers(0, 2, 2_000)
_NOISE = _RNG.normal(size=2_000)


def gate_scores(signal: float, noise_seed: int | None = None) -> np.ndarray:
    """Scores whose AUC on GATE_Y grows with `signal`. All models share the same
    noise unless `noise_seed` gives one its own (an equally good, different model)."""
    noise = _NOISE if noise_seed is None else np.random.default_rng(noise_seed).normal(size=2_000)
    return 1 / (1 + np.exp(-(signal * (2 * GATE_Y - 1) + noise)))


class FakeBackend:
    """Champion is version "1", the candidate "2". Models differ by `signal`."""

    def __init__(self, champion="1", champ_signal=0.5, cand_signal=1.5, fail_train=False,
                 cand_noise_seed=None):
        self.champion = champion
        self.signals = {"1": champ_signal, "2": cand_signal}
        self.noise_seeds = {"1": None, "2": cand_noise_seed}
        self.fail_train = fail_train
        self.calls: list[tuple] = []

    def champion_version(self):
        return self.champion

    def train_candidate(self):
        self.calls.append(("train",))
        if self.fail_train:
            raise RuntimeError("boom")
        return "2"

    def holdout_predictions(self, version):
        self.calls.append(("evaluate", version))
        return GATE_Y, gate_scores(self.signals[version], self.noise_seeds[version])

    def promote(self, version):
        self.calls.append(("promote", version))

    def reject(self, version):
        self.calls.append(("reject", version))


def frames(n=1500):
    rng = np.random.default_rng(0)
    return (
        pd.DataFrame({"x": rng.normal(size=n)}),
        pd.DataFrame({"x": rng.normal(size=n) + 1}),
    )


def severe(_ref, _cur):
    return FakeReport(["x"], [])


def calm(_ref, _cur):
    return FakeReport([], ["y"])


# ----- decision.py ----------------------------------------------------------


def test_drift_gate_threshold():
    cfg = RetrainConfig(min_severe_features=2)
    assert not decide_drift(FakeReport(["a"], []), cfg).should_retrain
    assert decide_drift(FakeReport(["a", "b", "a"], []), cfg).should_retrain  # dedupes


def test_drift_gate_force():
    assert decide_drift(FakeReport([], []), RetrainConfig(force=True)).should_retrain


def test_promotion_requires_margin():
    cfg = RetrainConfig(promotion_margin=0.01)
    assert decide_promotion(0.82, 0.80, cfg).promote
    assert decide_promotion(0.81, 0.80, cfg).promote  # exactly at margin (float tolerance)
    assert not decide_promotion(0.805, 0.80, cfg).promote
    assert not decide_promotion(0.70, 0.80, cfg).promote


def test_promotion_without_champion_bootstraps():
    d = decide_promotion(0.7, None, RetrainConfig())
    assert d.promote and d.improvement is None


def test_promotion_logloss_lower_is_better():
    cfg = RetrainConfig(metric="logloss", promotion_margin=0.01)
    assert decide_promotion(0.30, 0.35, cfg).promote
    assert not decide_promotion(0.40, 0.35, cfg).promote


def test_config_validation():
    with pytest.raises(ValueError):
        RetrainConfig(metric="accuracy")
    with pytest.raises(ValueError):
        RetrainConfig(promotion_margin=-1)
    with pytest.raises(ValueError):
        RetrainConfig(n_bootstrap=-1)
    with pytest.raises(ValueError):
        RetrainConfig(ci_level=1.0)


def test_promotion_with_interval_uses_lower_bound():
    cfg = RetrainConfig(promotion_margin=0.002)
    # Point estimate clears the margin, but the interval reaches below it: reject.
    d = decide_promotion(0.81, 0.80, cfg, ci=(-0.001, 0.021))
    assert not d.promote and d.improvement == pytest.approx(0.01)
    assert d.ci_lower == -0.001 and d.ci_level == 0.95
    assert decide_promotion(0.81, 0.80, cfg, ci=(0.003, 0.017)).promote
    assert decide_promotion(0.81, 0.80, cfg, ci=(0.002, 0.017)).promote  # bound at margin


# ----- holdout metrics and paired bootstrap ----------------------------------


def test_numpy_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(np.round(rng.uniform(size=500), 1), 0.05, 0.95)  # coarse values force ties
    assert auc_score(y, p) == pytest.approx(roc_auc_score(y, p))
    assert pr_auc_score(y, p) == pytest.approx(average_precision_score(y, p))
    assert logloss_score(y, p) == pytest.approx(log_loss(y, p))
    assert set(holdout_metrics(y, p)) == {"auc", "pr_auc", "logloss"}


def test_paired_bootstrap_identical_models_is_exactly_zero():
    p = gate_scores(1.0)
    assert paired_bootstrap_improvement(GATE_Y, p, p, "auc", n_boot=200) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_brackets_point_and_detects_clear_gain():
    d, lo, hi = paired_bootstrap_improvement(
        GATE_Y, gate_scores(1.5), gate_scores(0.5), "auc", n_boot=300
    )
    assert lo <= d <= hi and lo > 0.05
    # log loss is lower-is-better: the better model still gets a positive improvement
    d2, lo2, _ = paired_bootstrap_improvement(
        GATE_Y, gate_scores(1.5), gate_scores(0.5), "logloss", n_boot=300
    )
    assert d2 > 0 and lo2 > 0


def test_paired_bootstrap_is_seeded_and_equal_models_straddle_zero():
    # Two different but equally good models: the interval must include zero.
    a, b = gate_scores(1.0), gate_scores(1.0, noise_seed=0)
    r1 = paired_bootstrap_improvement(GATE_Y, b, a, "auc", n_boot=300, seed=1)
    assert r1 == paired_bootstrap_improvement(GATE_Y, b, a, "auc", n_boot=300, seed=1)
    _, lo, hi = r1
    assert lo < 0 < hi


def test_paired_bootstrap_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_improvement(GATE_Y, gate_scores(1.0)[:-1], gate_scores(1.0), "auc")


# ----- orchestrator ---------------------------------------------------------


def test_no_drift_does_not_train(tmp_path):
    ref, cur = frames()
    b = FakeBackend()
    rec = run_cycle(ref, cur, RetrainConfig(), b, DecisionStore(tmp_path), drift_fn=calm, now=NOW)
    assert rec.status == "no_drift"
    assert b.calls == []


def test_promotes_when_candidate_better(tmp_path):
    ref, cur = frames()
    b = FakeBackend(champ_signal=0.5, cand_signal=1.5)
    rec = run_cycle(ref, cur, RetrainConfig(), b, DecisionStore(tmp_path),
                    drift_fn=severe, now=NOW)
    assert rec.status == "promoted"
    assert ("promote", "2") in b.calls
    expected = auc_score(GATE_Y, gate_scores(1.5)) - auc_score(GATE_Y, gate_scores(0.5))
    assert rec.promotion["improvement"] == pytest.approx(expected)
    assert rec.promotion["ci_lower"] >= 0.002 and rec.promotion["n_holdout"] == 2_000
    saved = json.loads(DecisionStore(tmp_path).path(rec.run_id).read_text())
    assert saved["status"] == "promoted" and saved["candidate_version"] == "2"


def test_rejects_when_not_better_by_margin(tmp_path):
    ref, cur = frames()
    b = FakeBackend(champ_signal=1.0, cand_signal=1.01)
    rec = run_cycle(ref, cur, RetrainConfig(promotion_margin=0.01), b,
                    DecisionStore(tmp_path), drift_fn=severe, now=NOW)
    assert rec.status == "rejected"
    assert ("reject", "2") in b.calls and ("promote", "2") not in b.calls


def test_rejects_when_point_gain_clears_margin_but_interval_does_not(tmp_path):
    """The point estimate alone would promote; the lower CI bound blocks it."""
    ref, cur = frames()
    margin = 0.002
    point = auc_score(GATE_Y, gate_scores(1.0, 0)) - auc_score(GATE_Y, gate_scores(1.0))
    assert point >= margin  # a lucky draw: an equally good model looks better on this holdout

    def backend():
        return FakeBackend(champ_signal=1.0, cand_signal=1.0, cand_noise_seed=0)

    loose = run_cycle(ref, cur, RetrainConfig(promotion_margin=margin, n_bootstrap=0),
                      backend(), DecisionStore(tmp_path / "a"), drift_fn=severe, now=NOW)
    assert loose.status == "promoted" and loose.promotion["ci_lower"] is None
    strict = run_cycle(ref, cur, RetrainConfig(promotion_margin=margin), backend(),
                       DecisionStore(tmp_path / "b"), drift_fn=severe, now=NOW)
    assert strict.status == "rejected"
    assert strict.promotion["ci_lower"] < margin <= strict.promotion["improvement"]


def test_holdout_row_mismatch_is_an_error(tmp_path):
    class Mismatch(FakeBackend):
        def holdout_predictions(self, version):
            y, p = super().holdout_predictions(version)
            return (y[::-1], p) if version == "1" else (y, p)

    ref, cur = frames()
    rec = run_cycle(ref, cur, RetrainConfig(), Mismatch(), DecisionStore(tmp_path),
                    drift_fn=severe, now=NOW)
    assert rec.status == "error" and "different holdout rows" in rec.detail


def test_adapter_refuses_final_test_split(tmp_path):
    from churn_platform.retrain.adapters import MlflowBackend

    with pytest.raises(ValueError, match="test split"):
        MlflowBackend(tmp_path, n_trials=1, sample_size=10, holdout_split="test")
    assert MlflowBackend(tmp_path, n_trials=1, sample_size=10).holdout_split == "gate"


def test_no_champion_promotes_without_evaluating_champion(tmp_path):
    ref, cur = frames()
    b = FakeBackend(champion=None)
    rec = run_cycle(ref, cur, RetrainConfig(), b, DecisionStore(tmp_path),
                    drift_fn=severe, now=NOW)
    assert rec.status == "promoted"
    assert ("evaluate", "1") not in b.calls


def test_idempotent_second_run_is_duplicate(tmp_path):
    ref, cur = frames()
    store = DecisionStore(tmp_path)
    b = FakeBackend()
    first = run_cycle(ref, cur, RetrainConfig(), b, store, drift_fn=severe, now=NOW)
    n_calls = len(b.calls)
    second = run_cycle(ref, cur, RetrainConfig(), b, store, drift_fn=severe, now=NOW)
    assert first.status == "promoted"
    assert second.status == "duplicate" and second.run_id == first.run_id
    assert len(b.calls) == n_calls  # nothing retrained again
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_error_is_recorded_and_retried(tmp_path):
    ref, cur = frames()
    store = DecisionStore(tmp_path)
    rec = run_cycle(ref, cur, RetrainConfig(), FakeBackend(fail_train=True), store,
                    drift_fn=severe, now=NOW)
    assert rec.status == "error" and "boom" in rec.detail
    # An error record must not block a retry on identical inputs.
    rec2 = run_cycle(ref, cur, RetrainConfig(), FakeBackend(), store, drift_fn=severe, now=NOW)
    assert rec2.status == "promoted"


def test_insufficient_data(tmp_path):
    ref, cur = frames(n=10)
    b = FakeBackend()
    rec = run_cycle(ref, cur, RetrainConfig(min_rows=100), b, DecisionStore(tmp_path),
                    drift_fn=severe, now=NOW)
    assert rec.status == "insufficient_data" and b.calls == []


def test_dry_run_never_trains(tmp_path):
    ref, cur = frames()
    b = FakeBackend()
    rec = run_cycle(ref, cur, RetrainConfig(dry_run=True), b, DecisionStore(tmp_path),
                    drift_fn=severe, now=NOW)
    assert rec.status == "dry_run" and b.calls == []


def test_real_drift_function_end_to_end(tmp_path):
    """Use the real score_drift on clearly shifted data (no stubs)."""
    ref, cur = frames(n=3000)
    b = FakeBackend()
    rec = run_cycle(ref, cur, RetrainConfig(), b, DecisionStore(tmp_path), now=NOW)
    assert rec.status == "promoted"
    assert rec.drift["severe_features"] == ["x"]


def test_run_id_depends_on_data_and_config_not_champion():
    ref, cur = frames()
    cfg = RetrainConfig()
    a = compute_run_id(fingerprint(ref), fingerprint(cur), cfg)
    assert a == compute_run_id(fingerprint(ref), fingerprint(cur), cfg)
    assert a != compute_run_id(fingerprint(ref), fingerprint(cur.iloc[:-1]), cfg)
    assert a != compute_run_id(fingerprint(ref), fingerprint(cur), RetrainConfig(promotion_margin=0.05))
    assert a != compute_run_id(fingerprint(ref), fingerprint(cur), RetrainConfig(n_bootstrap=0))


def test_rerun_after_promotion_does_not_retrain_again(tmp_path):
    """The first run promotes v2; the champion is now v2, yet identical data is a duplicate."""
    ref, cur = frames()
    store = DecisionStore(tmp_path)
    first = run_cycle(ref, cur, RetrainConfig(), FakeBackend(champion="1"), store,
                      drift_fn=severe, now=NOW)
    b2 = FakeBackend(champion="2")
    second = run_cycle(ref, cur, RetrainConfig(), b2, store, drift_fn=severe, now=NOW)
    assert first.status == "promoted" and second.status == "duplicate"
    assert b2.calls == []


# ----- windowing ------------------------------------------------------------


def test_select_recent_window_filters_and_caps():
    ts = pd.date_range("2026-01-01", periods=60, freq="D")  # naive timestamps
    df = pd.DataFrame({"event_timestamp": ts, "v": range(60)})
    out = select_recent_window(df, days=10, max_rows=None, now=NOW)
    assert out["event_timestamp"].min() >= pd.Timestamp("2026-01-21")
    capped = select_recent_window(df, days=None, max_rows=5, now=NOW)
    assert len(capped) == 5
    pd.testing.assert_frame_equal(
        capped, select_recent_window(df, days=None, max_rows=5, now=NOW)
    )  # seeded => reproducible


def test_select_recent_window_tz_aware():
    ts = pd.date_range("2026-01-01", periods=31, freq="D", tz="UTC")
    df = pd.DataFrame({"event_timestamp": ts})
    assert len(select_recent_window(df, days=5, max_rows=None, now=NOW)) == 6


def test_select_recent_window_without_timestamp_column():
    df = pd.DataFrame({"v": range(10)})
    assert len(select_recent_window(df, days=3, max_rows=None, now=NOW)) == 10


# ----- metrics --------------------------------------------------------------


def test_metrics_observe():
    m = RetrainMetrics()
    m.observe(
        {
            "status": "promoted",
            "drift": {"severe_features": ["a", "b"]},
            "promotion": {"metric": "auc", "candidate_value": 0.9, "champion_value": 0.8,
                          "ci_lower": 0.05},
        },
        now_ts=123.0,
    )
    assert m.registry.get_sample_value("churn_retrain_severe_drift_features") == 2
    assert m.registry.get_sample_value(
        "churn_retrain_last_outcome", {"outcome": "promoted"}
    ) == 1
    assert m.registry.get_sample_value("churn_retrain_candidate_metric", {"metric": "auc"}) == 0.9
    assert m.registry.get_sample_value(
        "churn_retrain_improvement_ci_lower", {"metric": "auc"}
    ) == 0.05
