"""Prometheus metrics for retrain decisions.

The retrain job is short-lived (a CronJob), so metrics are pushed to a
Pushgateway when a URL is configured; otherwise they are simply not exported.
Uses a private registry so nothing leaks into the serving app's default one.
"""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

log = logging.getLogger("retrain.metrics")


class RetrainMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        # A gauge (not a Counter): each push from a fresh process would reset a
        # counter to 1. The label carries the outcome of the most recent cycle.
        self.runs = Gauge(
            "churn_retrain_last_outcome", "1 for the outcome of the most recent cycle.",
            ["outcome"], registry=self.registry,
        )
        self.last_run = Gauge(
            "churn_retrain_last_run_timestamp_seconds", "Unix time of last cycle.",
            registry=self.registry,
        )
        self.severe = Gauge(
            "churn_retrain_severe_drift_features", "Features with severe drift in last cycle.",
            registry=self.registry,
        )
        self.candidate = Gauge(
            "churn_retrain_candidate_metric", "Candidate holdout metric.",
            ["metric"], registry=self.registry,
        )
        self.champion = Gauge(
            "churn_retrain_champion_metric", "Champion holdout metric.",
            ["metric"], registry=self.registry,
        )

        self.ci_lower = Gauge(
            "churn_retrain_improvement_ci_lower",
            "Lower CI bound of the candidate-minus-champion improvement (sign-corrected).",
            ["metric"], registry=self.registry,
        )

    def observe(self, record: dict, now_ts: float) -> None:
        self.runs.labels(outcome=record["status"]).set(1)
        self.last_run.set(now_ts)
        if record.get("drift"):
            self.severe.set(len(record["drift"]["severe_features"]))
        promo = record.get("promotion")
        if promo:
            self.candidate.labels(metric=promo["metric"]).set(promo["candidate_value"])
            if promo["champion_value"] is not None:
                self.champion.labels(metric=promo["metric"]).set(promo["champion_value"])
            if promo.get("ci_lower") is not None:
                self.ci_lower.labels(metric=promo["metric"]).set(promo["ci_lower"])

    def push(self, url: str, job: str = "churn-retrain") -> None:
        try:
            push_to_gateway(url, job=job, registry=self.registry)
        except Exception:  # monitoring must never fail the job
            log.exception("Could not push metrics to %s", url)
