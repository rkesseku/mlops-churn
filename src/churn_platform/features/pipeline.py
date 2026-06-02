"""Spark feature engineering pipeline.

Reads the raw customer Parquet, derives engineered features, splits into
train/val/test, and writes Parquet outputs.

Designed to run two ways:

    Local mode (input is a path on the container's local disk):
        spark-submit /opt/jobs/src/churn_platform/features/pipeline.py \\
            --input /opt/jobs/data/raw/customers.parquet \\
            --output /opt/jobs/data/features

    MinIO mode (input is an S3A URL):
        spark-submit /opt/jobs/src/churn_platform/features/pipeline.py \\
            --input s3a://raw-data/customers.parquet \\
            --output s3a://features

Same code, different paths. We default to local mode for the first run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

log = logging.getLogger("features")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


# ----- Spark session setup --------------------------------------------------


def build_spark(app_name: str = "churn-features") -> SparkSession:
    """Build a SparkSession configured to read from MinIO via S3A.

    The four `spark.hadoop.fs.s3a.*` settings are what wire Spark up to
    MinIO. Without these, the S3A connector fails to find credentials.

    We also disable Spark's verbose log output. Default Spark logging at
    INFO level prints hundreds of lines per query, drowning our own logs.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        # MinIO endpoint (uses the docker-compose service name)
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        # MinIO requires "path-style" S3 URLs (s3a://bucket/key)
        # rather than virtual-host-style (s3a://key.bucket.s3.amazonaws.com).
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        # Trust the default S3A credentials chain (env vars + the settings above)
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        # Accept nanosecond-precision timestamps in Parquet (pandas default).
        # Without this, Spark 3.5 errors on `INT64 (TIMESTAMP(NANOS,true))`.
        .config("spark.sql.parquet.inferTimestampNTZ.enabled", "true")
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        # Quieter Spark logging — only WARN and above for the Spark internals.
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ----- Feature transformations ----------------------------------------------
#
# Each function takes a DataFrame and returns a new DataFrame with one or
# more new columns. They're separate so we can compose them, test them
# independently, and reorder if needed.


def add_charge_ratio(df: DataFrame) -> DataFrame:
    """Ratio of current monthly charge to historical average monthly charge.

    Values > 1 mean the customer is paying more now than they used to —
    a price-hike signal. Values < 1 mean they're paying less (a discount).
    Around 1 means stable pricing.

    Guards against tenure_months == 0 (would divide by zero).
    """
    historical_monthly = F.col("total_charges") / F.greatest(
        F.col("tenure_months"), F.lit(1)
    )
    return df.withColumn(
        "charge_ratio",
        F.when(
            F.col("total_charges").isNull(),
            F.lit(1.0),  # new customers: assume stable pricing
        ).otherwise(F.col("monthly_charges") / historical_monthly),
    )


def add_tenure_bucket(df: DataFrame) -> DataFrame:
    """Discrete tenure category: new / mid / loyal.

    Trees split better on a small discrete feature than on a continuous one
    when the underlying relationship is non-monotonic — but tenure here IS
    monotonic, so we keep tenure_months as well. The bucket is an extra
    feature, not a replacement.
    """
    return df.withColumn(
        "tenure_bucket",
        F.when(F.col("tenure_months") < 6, "new")
        .when(F.col("tenure_months") < 24, "mid")
        .otherwise("loyal")
        .cast(StringType()),
    )


def add_engagement_score(df: DataFrame) -> DataFrame:
    """Composite usage signal.

        engagement = (logins_30d * avg_session_minutes) / (days_since_last_login + 1)

    High score = frequent + long sessions + recent. Low score = checked-out.
    """
    return df.withColumn(
        "engagement_score",
        (F.col("num_logins_30d") * F.col("avg_session_minutes"))
        / (F.col("days_since_last_login") + 1),
    )


def add_support_calls_per_login(df: DataFrame) -> DataFrame:
    """Friction-to-usage ratio.

    Heavy support callers who barely log in are the most at-risk
    behavioral profile. The +1 in the denominator avoids division by zero
    for the (rare) zero-login customer.
    """
    return df.withColumn(
        "support_calls_per_login",
        F.col("num_support_calls") / (F.col("num_logins_30d") + F.lit(1)),
    )


def add_risk_flag_no_renew_m2m(df: DataFrame) -> DataFrame:
    """Binary flag for the highest-risk contract combination."""
    return df.withColumn(
        "risk_flag_no_renew_m2m",
        (~F.col("auto_renew")) & (F.col("contract_type") == "month-to-month"),
    )


def fill_total_charges(df: DataFrame) -> DataFrame:
    """Imputes total_charges where it was null (tenure_months == 0).

    Drop-and-replace pattern: add a filled column rather than mutating the
    original, so the missingness signal is preserved if downstream code
    wants it.
    """
    return df.withColumn(
        "total_charges_filled",
        F.coalesce(
            F.col("total_charges"),
            F.col("monthly_charges") * F.col("tenure_months"),
        ),
    )


def fill_nps(df: DataFrame) -> DataFrame:
    """Imputes nps_score with the neutral value 5, and adds an indicator.

    The indicator nps_was_null lets the model learn that NPS nonresponse
    is itself a signal — often it correlates with disengagement.
    """
    return df.withColumn(
        "nps_was_null", F.col("nps_score").isNull()
    ).withColumn(
        "nps_filled",
        F.coalesce(F.col("nps_score"), F.lit(5.0)),
    )


# Order matters: derived columns must come after the columns they reference.
TRANSFORMS = [
    add_charge_ratio,
    add_tenure_bucket,
    add_engagement_score,
    add_support_calls_per_login,
    add_risk_flag_no_renew_m2m,
    fill_total_charges,
    fill_nps,
]


def transform(df: DataFrame) -> DataFrame:
    """Apply all feature transformations in order."""
    for fn in TRANSFORMS:
        df = fn(df)
    return df


# ----- Train/val/test split -------------------------------------------------


def split(
    df: DataFrame,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Random 70/15/15 split. Uses Spark's randomSplit for parallel sampling."""
    train, val, test = df.randomSplit(list(fractions), seed=seed)
    return train, val, test


# ----- Orchestration --------------------------------------------------------


def run(spark: SparkSession, input_path: str, output_path: str) -> None:
    """Read input, transform, split, write outputs."""
    log.info("Reading %s", input_path)
    raw = spark.read.parquet(input_path)
    n_raw = raw.count()
    log.info("Read %d rows", n_raw)

    features = transform(raw)
    log.info("Applied %d transforms", len(TRANSFORMS))

    train, val, test = split(features)

    # Write as Parquet to <output_path>/{train,val,test}.parquet.
    # Spark writes each as a directory of part-* files; that's normal.
    out = output_path.rstrip("/")
    for name, df_split in [("train", train), ("val", val), ("test", test)]:
        target = f"{out}/{name}.parquet"
        log.info("Writing %s ...", target)
        df_split.write.mode("overwrite").parquet(target)
        log.info("  wrote %d rows", df_split.count())


# ----- CLI ------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build features from raw customer data.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw parquet. Local path (/opt/jobs/...) or s3a:// URL.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will contain train/val/test subdirs.",
    )
    args = parser.parse_args()

    spark = build_spark()
    try:
        run(spark, args.input, args.output)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
