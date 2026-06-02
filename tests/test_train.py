"""Tests for the training module.

We don't test the full Optuna+MLflow loop here — that's an integration test
covered by `make train`. These are unit-level tests of the pieces that
benefit from isolation: the search space, data preparation, evaluation.
"""

from __future__ import annotations

import pandas as pd
import pytest
from xgboost import XGBClassifier

from churn_platform.train.search_space import DEFAULT_N_TRIALS, suggest_params
from churn_platform.train.train import (
    CATEGORICAL_COLS,
    NON_FEATURE_COLS,
    Datasets,
    _prepare_features,
    evaluate_on_test,
)


# ----- Search space ---------------------------------------------------------


class TestSearchSpace:
    """The Optuna search space should always return valid XGBoost params."""

    def test_default_n_trials_is_reasonable(self):
        assert 20 <= DEFAULT_N_TRIALS <= 200

    def test_suggest_params_returns_all_required_keys(self):
        import optuna

        study = optuna.create_study(direction="maximize")
        captured = {}

        def _probe(trial):
            captured.update(suggest_params(trial))
            return 0.0

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(_probe, n_trials=1)

        expected_keys = {
            "n_estimators", "learning_rate", "max_depth", "min_child_weight",
            "gamma", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        }
        assert set(captured.keys()) == expected_keys

    def test_suggested_params_pass_xgboost_validation(self):
        """The suggested params must actually be valid XGBoost arguments."""
        import optuna

        study = optuna.create_study(direction="maximize")

        def _probe(trial):
            params = suggest_params(trial)
            # If params are invalid, this constructor call will raise.
            XGBClassifier(**params)
            return 0.0

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(_probe, n_trials=5)


# ----- Feature preparation --------------------------------------------------


class TestPrepareFeatures:
    """_prepare_features should clean, encode, and split (X, y) correctly."""

    @pytest.fixture
    def raw_df(self):
        """A tiny dataframe mirroring the post-Spark feature schema."""
        return pd.DataFrame({
            "customer_id": ["C0001", "C0002", "C0003"],
            "event_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "tenure_months": [3, 36, 12],
            "contract_type": ["month-to-month", "two-year", "one-year"],
            "monthly_charges": [55.0, 80.0, 65.0],
            "total_charges": [None, 2880.0, 780.0],
            "payment_method": ["echeck", "banktx", "creditcard"],
            "internet_service": ["Fiber", "DSL", "Fiber"],
            "num_support_calls": [4, 0, 2],
            "num_logins_30d": [3, 25, 15],
            "avg_session_minutes": [6.0, 40.0, 22.0],
            "days_since_last_login": [20, 1, 4],
            "has_paperless_billing": [True, False, True],
            "auto_renew": [False, True, True],
            "discount_applied": [0.0, 0.15, 0.10],
            "nps_score": [2.0, 9.0, None],
            # derived columns (from Spark features)
            "charge_ratio": [1.0, 1.0, 1.0],
            "tenure_bucket": ["new", "loyal", "mid"],
            "engagement_score": [0.9, 666.7, 82.5],
            "support_calls_per_login": [1.0, 0.0, 0.13],
            "risk_flag_no_renew_m2m": [True, False, False],
            "total_charges_filled": [165.0, 2880.0, 780.0],
            "nps_filled": [2.0, 9.0, 5.0],
            "nps_was_null": [False, False, True],
            "churned": [True, False, False],
        })

    def test_drops_non_feature_columns(self, raw_df):
        X, _ = _prepare_features(raw_df)
        for col in NON_FEATURE_COLS:
            assert col not in X.columns

    def test_returns_correct_y(self, raw_df):
        _, y = _prepare_features(raw_df)
        assert y.tolist() == [1, 0, 0]

    def test_categorical_columns_have_category_dtype(self, raw_df):
        X, _ = _prepare_features(raw_df)
        for col in CATEGORICAL_COLS:
            assert str(X[col].dtype) == "category", (
                f"{col} should be category dtype, got {X[col].dtype}"
            )

    def test_filled_columns_retained(self, raw_df):
        """total_charges_filled and nps_filled should survive — they replace
        the originals."""
        X, _ = _prepare_features(raw_df)
        assert "total_charges_filled" in X.columns
        assert "nps_filled" in X.columns
        assert "nps_was_null" in X.columns


# ----- Evaluation -----------------------------------------------------------


class TestEvaluateOnTest:
    """evaluate_on_test should return AUC, logloss, and PR-AUC."""

    def test_returns_expected_keys(self):
        # Build a trivial model on synthetic data and call evaluate_on_test.
        rng = pd.np.random.default_rng(0) if hasattr(pd, "np") else __import__("numpy").random.default_rng(0)

        X = pd.DataFrame({
            "f1": rng.normal(size=200),
            "f2": rng.normal(size=200),
        })
        y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
        Xt = pd.DataFrame({
            "f1": rng.normal(size=100),
            "f2": rng.normal(size=100),
        })
        yt = pd.Series((Xt["f1"] + Xt["f2"] > 0).astype(int))

        model = XGBClassifier(n_estimators=20, max_depth=3, verbosity=0)
        model.fit(X, y)

        ds = Datasets(X, y, X, y, Xt, yt)
        metrics = evaluate_on_test(model, ds)

        assert set(metrics.keys()) == {"test_auc", "test_logloss", "test_pr_auc"}
        assert 0.0 <= metrics["test_auc"] <= 1.0
        assert metrics["test_logloss"] > 0
        assert 0.0 <= metrics["test_pr_auc"] <= 1.0
