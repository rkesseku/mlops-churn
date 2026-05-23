"""Tests for the synthetic data generator.

These are *statistical* tests, not pure unit tests — they assert properties
of the generated distribution. Tolerances are tight enough to catch real
regressions but loose enough to survive minor random-state changes.

If a tolerance feels too strict in practice, widen it deliberately rather
than silently bumping it every CI failure; the test catching a real drift
is the whole point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_platform.generate.synthesize import (
    GenParams,
    _apply_drift,
    generate,
)

# Default test size. 10k is big enough for stable statistics, small enough
# that pytest doesn't take forever.
TEST_N = 10_000


@pytest.fixture(scope="module")
def baseline_df() -> pd.DataFrame:
    """A single 10k-row baseline dataset, reused across most tests.

    `scope="module"` means it's generated once for the whole file, not per
    test — saves ~3 seconds across the suite.
    """
    return generate(GenParams(n_rows=TEST_N, seed=42))


# ----- Schema and basic shape -----------------------------------------------


class TestSchema:
    """The dataset must always have the expected shape and columns."""

    EXPECTED_COLUMNS = {
        "customer_id", "tenure_months", "contract_type", "monthly_charges",
        "total_charges", "payment_method", "internet_service",
        "num_support_calls", "num_logins_30d", "avg_session_minutes",
        "days_since_last_login", "has_paperless_billing", "auto_renew",
        "discount_applied", "nps_score", "event_timestamp", "churned",
    }

    def test_row_count_matches_request(self, baseline_df):
        assert len(baseline_df) == TEST_N

    def test_all_expected_columns_present(self, baseline_df):
        assert set(baseline_df.columns) == self.EXPECTED_COLUMNS

    def test_no_label_is_null(self, baseline_df):
        assert baseline_df["churned"].notna().all(), (
            "the label column must never be null"
        )

    def test_customer_ids_are_unique(self, baseline_df):
        assert baseline_df["customer_id"].is_unique


# ----- Statistical properties -----------------------------------------------


class TestStatistics:
    """The data must reflect the intended causal structure."""

    def test_churn_rate_near_target(self, baseline_df):
        """Base churn rate should sit close to GenParams.base_churn_rate=0.27.

        With n=10k and bernoulli sampling, the 99% CI for a true rate of 0.27
        is roughly ±0.012. We use ±0.02 to be safe.
        """
        rate = baseline_df["churned"].mean()
        assert 0.25 <= rate <= 0.29, f"churn rate {rate:.4f} outside expected range"

    def test_month_to_month_churns_more_than_two_year(self, baseline_df):
        """Contract length is the second-strongest churn predictor."""
        by_contract = baseline_df.groupby("contract_type")["churned"].mean()
        assert by_contract["month-to-month"] > by_contract["two-year"] + 0.15, (
            f"month-to-month should churn substantially more than two-year; "
            f"got {by_contract.to_dict()}"
        )

    def test_new_customers_churn_more_than_loyal(self, baseline_df):
        """Tenure is the strongest single churn predictor."""
        new = baseline_df.loc[baseline_df["tenure_months"] < 6, "churned"].mean()
        loyal = baseline_df.loc[baseline_df["tenure_months"] >= 24, "churned"].mean()
        assert new > loyal + 0.15, (
            f"new customers ({new:.3f}) should churn far more than loyal ({loyal:.3f})"
        )

    def test_no_autorenew_churns_more(self, baseline_df):
        """Turning off auto-renew is the smoking-gun churn signal."""
        no_renew = baseline_df.loc[~baseline_df["auto_renew"], "churned"].mean()
        renewing = baseline_df.loc[baseline_df["auto_renew"], "churned"].mean()
        assert no_renew > renewing + 0.05, (
            f"no-autorenew ({no_renew:.3f}) should churn more than autorenew ({renewing:.3f})"
        )


# ----- Missingness ----------------------------------------------------------


class TestMissingness:
    """Some columns are intentionally null in specific patterns."""

    def test_total_charges_null_iff_zero_tenure(self, baseline_df):
        """total_charges is null for tenure_months==0 only (Telco-dataset quirk)."""
        null_rows = baseline_df[baseline_df["total_charges"].isna()]
        assert (null_rows["tenure_months"] == 0).all(), (
            "total_charges should be null only when tenure_months == 0"
        )
        non_null = baseline_df[baseline_df["total_charges"].notna()]
        assert (non_null["tenure_months"] > 0).all(), (
            "non-null total_charges rows should have tenure_months > 0"
        )

    def test_nps_response_rate_in_expected_range(self, baseline_df):
        """About 70% of customers respond to NPS surveys (30% null)."""
        response_rate = baseline_df["nps_score"].notna().mean()
        assert 0.65 <= response_rate <= 0.75, (
            f"NPS response rate {response_rate:.3f} outside [0.65, 0.75]"
        )


# ----- Reproducibility ------------------------------------------------------


class TestReproducibility:
    """Same seed must produce identical data."""

    def test_same_seed_same_output(self):
        df1 = generate(GenParams(n_rows=1000, seed=123))
        df2 = generate(GenParams(n_rows=1000, seed=123))
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_output(self):
        df1 = generate(GenParams(n_rows=1000, seed=123))
        df2 = generate(GenParams(n_rows=1000, seed=124))
        assert not df1["churned"].equals(df2["churned"])


# ----- Drift modes ----------------------------------------------------------


class TestDriftModes:
    """Each drift mode must mutate the dataset in the documented way."""

    def test_drift_none_changes_nothing(self):
        """drift_mode='none' should match no-drift baseline exactly."""
        baseline = GenParams(n_rows=2000, seed=99)
        drifted = GenParams(n_rows=2000, seed=99, drift_mode="none", drift_strength=0.8)
        # _apply_drift returns the same params object semantically.
        # Easier to assert: the generated frames should be identical.
        df1 = generate(baseline)
        df2 = generate(drifted)
        pd.testing.assert_frame_equal(df1, df2)

    def test_label_drift_increases_churn(self):
        """drift_mode='label' shifts the base churn rate upward."""
        baseline_rate = generate(GenParams(n_rows=5000, seed=7)).churned.mean()
        drifted_rate = generate(
            GenParams(n_rows=5000, seed=7, drift_mode="label", drift_strength=0.5)
        ).churned.mean()
        assert drifted_rate > baseline_rate + 0.05, (
            f"label drift should raise churn rate; "
            f"baseline={baseline_rate:.3f}, drifted={drifted_rate:.3f}"
        )

    def test_covariate_drift_shifts_tenure(self):
        """drift_mode='covariate' increases mean tenure."""
        baseline_tenure = generate(GenParams(n_rows=5000, seed=7)).tenure_months.mean()
        drifted_tenure = generate(
            GenParams(n_rows=5000, seed=7, drift_mode="covariate", drift_strength=0.5)
        ).tenure_months.mean()
        assert drifted_tenure > baseline_tenure + 2.0, (
            f"covariate drift should raise tenure mean; "
            f"baseline={baseline_tenure:.1f}, drifted={drifted_tenure:.1f}"
        )

    def test_concept_drift_increases_price_sensitivity(self):
        """drift_mode='concept' makes monthly_charges a stronger churn driver.

        We check that the correlation between charges and churn increases.
        """
        baseline = generate(GenParams(n_rows=10000, seed=7))
        drifted = generate(
            GenParams(n_rows=10000, seed=7, drift_mode="concept", drift_strength=0.5)
        )
        baseline_corr = np.corrcoef(
            baseline["monthly_charges"], baseline["churned"].astype(float)
        )[0, 1]
        drifted_corr = np.corrcoef(
            drifted["monthly_charges"], drifted["churned"].astype(float)
        )[0, 1]
        assert drifted_corr > baseline_corr + 0.02, (
            f"concept drift should strengthen price→churn correlation; "
            f"baseline={baseline_corr:.4f}, drifted={drifted_corr:.4f}"
        )


# ----- Drift helper unit test -----------------------------------------------


def test_apply_drift_does_not_mutate_input():
    """The _apply_drift helper must not modify its input GenParams in place."""
    original = GenParams(n_rows=100, seed=1, drift_mode="covariate", drift_strength=0.5)
    original_mean = original.tenure_mean
    _ = _apply_drift(original)
    assert original.tenure_mean == original_mean, (
        "_apply_drift mutated the input params"
    )
