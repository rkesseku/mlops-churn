"""Tests for the Spark feature engineering pipeline.

These tests use a local SparkSession (not the Docker Spark service) so they
run as part of the normal pytest suite. Same code paths as production —
we're testing the actual PySpark transformations, not a pandas approximation.

A local SparkSession starts in ~3 seconds and runs in the test process.
Fast enough for unit tests, deterministic enough for CI.
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from churn_platform.features.pipeline import (
    add_charge_ratio,
    add_engagement_score,
    add_risk_flag_no_renew_m2m,
    add_support_calls_per_login,
    add_tenure_bucket,
    fill_nps,
    fill_total_charges,
    split,
    transform,
)


# ----- Spark session fixture -----------------------------------------------


@pytest.fixture(scope="session")
def spark():
    """A single SparkSession reused across all tests.

    scope="session" because creating a SparkSession takes ~3 seconds; we
    don't want to pay that cost per test.

    `.master("local[2]")` runs Spark in-process with 2 worker threads. Plenty
    for tests; production jobs run in the Docker Spark container.
    """
    spark = (
        SparkSession.builder.appName("churn-features-tests")
        .master("local[2]")
        # Don't drag in S3A driver init for these tests.
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


# ----- Sample data fixture --------------------------------------------------


@pytest.fixture
def sample_df(spark):
    """A small DataFrame mirroring the raw customer schema.

    Hand-crafted so we can write precise assertions on the transforms.
    Covers edge cases: tenure==0, null total_charges, null nps_score,
    auto_renew False on month-to-month, etc.
    """
    rows = [
        # (customer_id, tenure_months, contract_type, monthly_charges,
        #  total_charges, payment_method, internet_service,
        #  num_support_calls, num_logins_30d, avg_session_minutes,
        #  days_since_last_login, has_paperless_billing, auto_renew,
        #  discount_applied, nps_score, churned)
        ("C0001",  0, "month-to-month", 50.00, None,    "echeck",     "DSL",   0,  5, 10.0,  1, True,  False,  0.0,  None, True),
        ("C0002", 12, "one-year",       65.00, 780.0,   "creditcard", "Fiber", 2, 15, 25.0,  3, True,  True,   0.10, 7.0,  False),
        ("C0003", 36, "two-year",       80.00, 2880.0,  "banktx",     "DSL",   1, 20, 30.0,  5, False, True,   0.0,  9.0,  False),
        ("C0004",  3, "month-to-month", 45.00, 135.0,   "echeck",     "Fiber", 5,  2,  5.0, 30, True,  False,  0.0,  2.0,  True),
        ("C0005", 60, "two-year",      100.00, 6000.0,  "creditcard", "DSL",   0, 25, 45.0,  1, False, True,   0.15, 10.0, False),
    ]
    columns = [
        "customer_id", "tenure_months", "contract_type", "monthly_charges",
        "total_charges", "payment_method", "internet_service",
        "num_support_calls", "num_logins_30d", "avg_session_minutes",
        "days_since_last_login", "has_paperless_billing", "auto_renew",
        "discount_applied", "nps_score", "churned",
    ]
    return spark.createDataFrame(rows, columns)


# ----- Individual transform tests -------------------------------------------


class TestChargeRatio:
    def test_adds_charge_ratio_column(self, sample_df):
        result = add_charge_ratio(sample_df).collect()
        assert "charge_ratio" in result[0].asDict()

    def test_stable_pricing_yields_ratio_near_one(self, sample_df):
        # C0002: monthly=65, tenure=12, total=780. Historical avg = 780/12 = 65.
        # Ratio = 65 / 65 = 1.0
        result = {r.customer_id: r.charge_ratio for r in add_charge_ratio(sample_df).collect()}
        assert result["C0002"] == pytest.approx(1.0, rel=0.01)

    def test_null_total_charges_yields_ratio_one(self, sample_df):
        # C0001 has null total_charges (tenure=0); should default to 1.0.
        result = {r.customer_id: r.charge_ratio for r in add_charge_ratio(sample_df).collect()}
        assert result["C0001"] == 1.0


class TestTenureBucket:
    def test_buckets_correctly(self, sample_df):
        result = {r.customer_id: r.tenure_bucket for r in add_tenure_bucket(sample_df).collect()}
        # tenure: 0, 12, 36, 3, 60
        assert result["C0001"] == "new"     # 0 < 6
        assert result["C0002"] == "mid"     # 6 <= 12 < 24
        assert result["C0003"] == "loyal"   # 24 <= 36
        assert result["C0004"] == "new"     # 3 < 6
        assert result["C0005"] == "loyal"   # 24 <= 60


class TestEngagementScore:
    def test_low_engagement_for_disengaged(self, sample_df):
        # C0004: logins=2, session=5min, days_since=30. Score = (2*5)/(30+1) ≈ 0.32
        result = {r.customer_id: r.engagement_score for r in add_engagement_score(sample_df).collect()}
        assert result["C0004"] == pytest.approx(10 / 31, rel=0.01)

    def test_high_engagement_for_active_customer(self, sample_df):
        # C0005: logins=25, session=45, days_since=1. Score = (25*45)/(1+1) = 562.5
        result = {r.customer_id: r.engagement_score for r in add_engagement_score(sample_df).collect()}
        assert result["C0005"] == pytest.approx(562.5, rel=0.01)

    def test_high_engagement_implies_lower_churn_intuition(self, sample_df):
        """Sanity check on the seed data: our highest-engagement row (C0005)
        is also not churned. Not a property of the transform, just a sanity
        check on our hand-built sample."""
        with_score = add_engagement_score(sample_df).collect()
        sorted_by_score = sorted(with_score, key=lambda r: r.engagement_score, reverse=True)
        # Top engagement should not be churned (our seed has C0005 not churned)
        assert sorted_by_score[0].churned is False


class TestRiskFlag:
    def test_flags_no_renew_on_month_to_month(self, sample_df):
        result = {r.customer_id: r.risk_flag_no_renew_m2m
                  for r in add_risk_flag_no_renew_m2m(sample_df).collect()}
        # C0001: auto_renew=False, m2m -> flagged
        # C0004: auto_renew=False, m2m -> flagged
        # C0002: auto_renew=True, one-year -> not flagged
        # C0003: auto_renew=True, two-year -> not flagged
        # C0005: auto_renew=True, two-year -> not flagged
        assert result["C0001"] is True
        assert result["C0004"] is True
        assert result["C0002"] is False
        assert result["C0003"] is False
        assert result["C0005"] is False


class TestSupportCallsPerLogin:
    def test_ratio_calculation(self, sample_df):
        # C0004: support=5, logins=2 -> 5/(2+1) = 1.667
        result = {r.customer_id: r.support_calls_per_login
                  for r in add_support_calls_per_login(sample_df).collect()}
        assert result["C0004"] == pytest.approx(5 / 3, rel=0.01)


class TestFillTotalCharges:
    def test_fills_null_with_monthly_times_tenure(self, sample_df):
        # C0001 has null total_charges, monthly=50, tenure=0 -> filled = 0
        result = {r.customer_id: r.total_charges_filled
                  for r in fill_total_charges(sample_df).collect()}
        assert result["C0001"] == 0.0

    def test_preserves_non_null_total_charges(self, sample_df):
        result = {r.customer_id: r.total_charges_filled
                  for r in fill_total_charges(sample_df).collect()}
        assert result["C0002"] == pytest.approx(780.0)


class TestFillNps:
    def test_fills_null_with_neutral(self, sample_df):
        # C0001 has null nps_score -> filled to 5.0
        result = {(r.customer_id): (r.nps_filled, r.nps_was_null)
                  for r in fill_nps(sample_df).collect()}
        assert result["C0001"] == (5.0, True)

    def test_preserves_non_null_nps(self, sample_df):
        result = {(r.customer_id): (r.nps_filled, r.nps_was_null)
                  for r in fill_nps(sample_df).collect()}
        assert result["C0002"] == (7.0, False)


# ----- End-to-end transform -------------------------------------------------


class TestTransform:
    def test_adds_all_expected_features(self, sample_df):
        result = transform(sample_df)
        new_cols = set(result.columns) - set(sample_df.columns)
        assert new_cols == {
            "charge_ratio",
            "tenure_bucket",
            "engagement_score",
            "support_calls_per_login",
            "risk_flag_no_renew_m2m",
            "total_charges_filled",
            "nps_filled",
            "nps_was_null",
        }

    def test_preserves_row_count(self, sample_df):
        original_count = sample_df.count()
        result_count = transform(sample_df).count()
        assert original_count == result_count

    def test_preserves_original_columns(self, sample_df):
        result = transform(sample_df)
        for col in sample_df.columns:
            assert col in result.columns


# ----- Split ----------------------------------------------------------------


class TestSplit:
    def test_splits_sum_to_total(self, spark):
        """The three splits should together cover every input row exactly once."""
        # Generate a larger frame for stable split sizing.
        df = spark.range(1000).withColumnRenamed("id", "customer_id")
        train, val, test = split(df, fractions=(0.7, 0.15, 0.15), seed=42)
        total = train.count() + val.count() + test.count()
        assert total == 1000

    def test_split_proportions_roughly_correct(self, spark):
        df = spark.range(10000).withColumnRenamed("id", "customer_id")
        train, val, test = split(df, fractions=(0.7, 0.15, 0.15), seed=42)
        # With 10k rows, expect train within ±2% of 7000, val/test within ±1.5% of 1500.
        assert 6800 <= train.count() <= 7200
        assert 1350 <= val.count() <= 1650
        assert 1350 <= test.count() <= 1650

    def test_split_is_deterministic(self, spark):
        df = spark.range(1000).withColumnRenamed("id", "customer_id")
        train1, _, _ = split(df, seed=42)
        train2, _, _ = split(df, seed=42)
        # Same seed should give identical row sets.
        ids1 = {r.customer_id for r in train1.collect()}
        ids2 = {r.customer_id for r in train2.collect()}
        assert ids1 == ids2
