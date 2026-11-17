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
    decide_drift,
    decide_promotion,
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


class FakeBackend:
    def __init__(self, champion="1", champ_auc=0.80, cand_auc=0.85, fail_train=False):
        self.champion = champion
        self.aucs = {"1": champ_auc, "2": cand_auc}
        self.fail_train = fail_train
        self.calls: list[tuple] = []

    def champion_version(self):
        return self.champion

    def train_candidate(self):
        self.calls.append(("train",))
        if self.fail_train:
            raise RuntimeError("boom")
        return "2"

    def evaluate(self, version):
        self.calls.append(("evaluate", version))
        return {"auc": self.aucs[version], "pr_auc": 0.5, "logloss": 0.4}

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


# ----- orchestrator ---------------------------------------------------------


def test_no_drift_does_not_train(tmp_path):
    ref, cur = frames()
    b = FakeBackend()
    rec = run_cycle(ref, cur, RetrainConfig(), b, DecisionStore(tmp_path), drift_fn=calm, now=NOW)
    assert rec.status == "no_drift"
    assert b.calls == []


def test_promotes_when_candidate_better(tmp_path):
    ref, cur = frames()
    b = FakeBackend(champ_auc=0.80, cand_auc=0.85)
    rec = run_cycle(ref, cur, RetrainConfig(), b, DecisionStore(tmp_path),
                    drift_fn=severe, now=NOW)
    assert rec.status == "promoted"
    assert ("promote", "2") in b.calls
    assert rec.promotion["improvement"] == pytest.approx(0.05)
    saved = json.loads(DecisionStore(tmp_path).path(rec.run_id).read_text())
    assert saved["status"] == "promoted" and saved["candidate_version"] == "2"


def test_rejects_when_not_better_by_margin(tmp_path):
    ref, cur = frames()
    b = FakeBackend(champ_auc=0.80, cand_auc=0.801)
    rec = run_cycle(ref, cur, RetrainConfig(promotion_margin=0.01), b,
                    DecisionStore(tmp_path), drift_fn=severe, now=NOW)
    assert rec.status == "rejected"
    assert ("reject", "2") in b.calls and ("promote", "2") not in b.calls


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
            "promotion": {"metric": "auc", "candidate_value": 0.9, "champion_value": 0.8},
        },
        now_ts=123.0,
    )
    assert m.registry.get_sample_value("churn_retrain_severe_drift_features") == 2
    assert m.registry.get_sample_value(
        "churn_retrain_last_outcome", {"outcome": "promoted"}
    ) == 1
    assert m.registry.get_sample_value("churn_retrain_candidate_metric", {"metric": "auc"}) == 0.9
