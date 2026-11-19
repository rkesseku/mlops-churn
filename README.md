# MLOps Churn Platform

End-to-end machine learning system for predicting customer churn: synthetic
data generation, Spark feature engineering, XGBoost with Optuna-tuned
hyperparameters, MLflow tracking and model registry, FastAPI serving in
Docker, Kubernetes manifests, Prometheus/Grafana monitoring, and drift-triggered
retraining with a champion/challenger promotion gate.

## Why this exists

Most ML projects stop at a notebook with a confusion matrix. Production ML
systems have to answer harder questions: How do we retrain when the data
distribution shifts? How do we know which model is in production? How do we
roll back safely? How do we detect silent degradation before customers do?

This project demonstrates the full MLOps lifecycle on a high-business-value
problem, with every stage wired together and reproducible locally.

## Architecture

```
 synthetic data --> Spark features --> Optuna + XGBoost --> MLflow registry
 (generate/)        (features/)        (train/)             (Postgres + MinIO)
                                                                  | alias: production / staging
                                                                  v
 Grafana <-- Prometheus <-- /metrics <-- FastAPI serving (serve/) +
                 ^
                 | push (Pushgateway)
 drift check (monitor/) --> retrain loop (retrain/) --> candidate --> champion vs challenger
 PSI + KS on recent window     severe drift?                          promote only if better
```

| Stage | Code | Notes |
|-------|------|-------|
| Data | `src/churn_platform/generate/` | Synthetic customers, optional covariate/concept drift injection |
| Features | `src/churn_platform/features/pipeline.py` | PySpark; writes train/val/gate/test parquet (60/15/10/15) |
| Training | `src/churn_platform/train/` | Optuna (TPE) search, XGBoost, every trial logged to MLflow |
| Registry | MLflow + Postgres + MinIO (docker compose) | Models referenced by alias (`staging`, `production`) |
| Serving | `src/churn_platform/serve/` | FastAPI: `/predict`, `/predict_batch`, `/health`, `/metrics`, `/drift/score` |
| Monitoring | `prometheus/`, `grafana/` | Prometheus scrapes serving; Grafana dashboard is provisioned |
| Drift | `src/churn_platform/monitor/drift.py` | PSI + KS per feature; severity none / moderate / severe |
| Retrain | `src/churn_platform/retrain/` | Drift gate, train, champion/challenger, promote, JSON decision record |
| Kubernetes | `k8s/` | Kustomize base + local overlay (see below) |

## Quick start (docker compose)

Prerequisites: Docker, Python 3.10+, `make`.

```bash
pip install -e ".[dev]"      # installs the package and dev tools
touch .env                   # the make targets source ./.env (defaults work)

make up                      # Postgres + MinIO + MLflow  (MLflow: http://localhost:5000)
make data ROWS=200000        # synthetic raw data -> data/raw/customers.parquet
make features                # Spark feature job -> data/features/{train,val,gate,test}.parquet
make train-quick             # 5-trial Optuna run, registers churn_xgboost@staging in MLflow
make serve                   # FastAPI at http://localhost:8000 (docs at /docs)
docker compose up -d prometheus grafana   # Prometheus :9090, Grafana :3000
```

Drift demo: `make demo-drift` generates a drifted dataset and scores it through
the serving API.

### Make targets

Run `make help` for the full list. The main ones:

| Target | What it does |
|--------|--------------|
| `up` / `down` / `clean` | Start / stop / wipe the compose stack |
| `data`, `data-baseline`, `data-drifted` | Generate synthetic datasets |
| `features` | Spark feature pipeline |
| `train`, `train-quick` | Optuna + XGBoost training, MLflow registration |
| `serve` | Build and run the FastAPI container |
| `demo-drift` | Generate drifted data and score it via the API |
| `retrain-dry-run` | Drift gate only; writes a decision record, never trains |
| `retrain-auto` | Full loop: drift gate, train, compare with champion, promote if better |
| `test`, `test-fast`, `lint` | pytest (all / skipping Spark), ruff |
| `k8s-build`, `k8s-load-kind`, `k8s-apply`, `k8s-delete`, `k8s-render`, `k8s-retrain-now` | Kubernetes workflow |

## Automated drift-triggered retraining

`python -m churn_platform.retrain` (or `make retrain-auto`) runs one cycle:

1. **Windowing.** Loads the reference (training-time) data and the current
   scoring data. Rows are limited to the last `--window-days` (when an
   `event_timestamp` column exists) and capped at `--max-rows` with a seeded
   sample.
2. **Drift gate.** Runs `monitor.drift.score_drift`. Retrain only if at least
   `--min-severe-features` distinct features are `severe` (PSI > 0.25, or a
   significant and large KS result). `--force` bypasses the gate.
3. **Train.** Reuses `train.train.run` (Optuna + XGBoost) and registers a new
   model version without touching the `staging` alias that serving loads.
   An optional `--features-cmd` (for example a `spark-submit`) can refresh the
   feature parquet first.
4. **Champion/challenger.** The champion is the version behind alias
   `production` (falling back to `staging`). Candidate and champion are both
   scored on the same dedicated **gate** holdout (`features/gate.parquet`,
   `--holdout-split`). It is separate from `val` (used for Optuna and early
   stopping) and from `test`, which `train.run` scores once per training run and
   which the gate refuses to use, so repeated promotion decisions never reuse the
   final test set.
5. **Promotion gate.** A paired bootstrap over the gate rows (both models scored
   on the same resampled rows, `--n-bootstrap` draws, default 1000) gives a
   `--ci-level` (default 95%) interval for the sign-corrected improvement on
   `--metric` (`auc` default; `pr_auc`, `logloss` supported). Promote only if
   the **lower bound** is at least `--promotion-margin` (default 0.002), so a
   gain that is within holdout noise is rejected; `--n-bootstrap 0` falls back
   to the point estimate. If there is no champion, the candidate is promoted.
   Promotion repoints the `production` and `staging` aliases; rejected versions
   are tagged `retrain_decision=rejected`.
6. **Decision record.** A JSON file per run is written to `--decision-dir`
   with status (`no_drift`, `dry_run`, `promoted`, `rejected`, `error`,
   `insufficient_data`, `duplicate`), drift detail, metrics and config.

**Idempotency.** The run id is a hash of the data window fingerprints and the
decision-relevant config (not the champion, which changes on promotion). Re-running on unchanged inputs returns
`duplicate` without retraining; a previous `error` record is retried.

**Metrics.** If `PUSHGATEWAY_URL` / `--pushgateway-url` is set, the job pushes
`churn_retrain_last_outcome`, `churn_retrain_severe_drift_features`,
`churn_retrain_candidate_metric`, `churn_retrain_champion_metric`,
`churn_retrain_improvement_ci_lower` and
`churn_retrain_last_run_timestamp_seconds`.

Scope notes: serving pods pick up a promoted model when they next start (for
example `kubectl rollout restart deploy/churn-serving`); there is no hot
reload. Feature generation for a new data window is Spark-based and is not
bundled into the CronJob; supply it through `--features-cmd` or an upstream job
that writes `RETRAIN_FEATURES_DIR`.

## Kubernetes

Manifests live in `k8s/` (kustomize):

- `k8s/base`: namespace, ConfigMap, Secret template (placeholders, not
  applied), data PVC, serving Deployment (readiness/liveness/startup probes on
  `/health`, requests/limits, Prometheus scrape annotations), Service,
  HorizontalPodAutoscaler (CPU 70%, needs metrics-server), Pushgateway, and the
  `churn-retrain` CronJob (daily 03:00, `concurrencyPolicy: Forbid`).
- `k8s/components/mlflow`: optional dev-only MLflow (SQLite on a PVC). For a
  real cluster, point `MLFLOW_TRACKING_URI` in the ConfigMap at an external,
  Postgres-backed MLflow instead.
- `k8s/overlays/local`: kind/minikube overlay that adds MLflow, generates dev
  placeholder credentials, uses local images and scales down.

Local deployment with kind:

```bash
kind create cluster
make k8s-build k8s-load-kind     # build churn-serving and churn-retrain, load into kind
make k8s-apply                   # kubectl apply -k k8s/overlays/local
kubectl -n mlops-churn get pods
make k8s-retrain-now             # run the CronJob once
make k8s-delete                  # tear down
```

For a real cluster: push the images to a registry, set `images:` in a new
overlay, create the `churn-secrets` Secret from `k8s/base/secret.template.yaml`
(or your secret manager), edit the ConfigMap, then `kubectl apply -k <overlay>`.
Prometheus must discover pods through the `prometheus.io/*` annotations
(annotation-based scrape config); a Prometheus/Grafana stack is not included in
`k8s/`. The CronJob expects the reference/current raw parquet and the feature
parquet on the `churn-data` PVC.

## Testing and CI

```bash
make test-fast      # everything except Spark tests
make test           # all tests (Spark/xgboost tests are skipped if those libs are missing)
```

Markers: `spark` (needs pyspark and a JVM), `heavy` (needs xgboost/optuna/mlflow
installed). No test needs a running MLflow server. The retrain decision logic
and the Kubernetes manifest checks run with fakes and PyYAML only.

GitHub Actions (`.github/workflows/ci.yml`): ruff + unit tests on Python 3.10
and 3.11, a separate Spark test job, and manifest validation with
`kubectl kustomize | kubeconform`.

## Status

Built incrementally; each milestone has tests and/or a runnable demo.

- [x] Milestone 1: Repo skeleton, Git, CI (GitHub Actions: lint, unit tests, manifest validation)
- [x] Milestone 2: Docker Compose stack: Postgres + MinIO + MLflow
- [x] Milestone 3: Synthetic data generator
- [x] Milestone 4: Spark feature pipeline
- [x] Milestone 5: Training with Optuna + MLflow tracking
- [x] Milestone 6: FastAPI serving from MLflow registry
- [x] Milestone 7: Prometheus + Grafana wiring
- [x] Milestone 8: Drift detection (PSI + KS) and `should_retrain` signal
- [x] Milestone 9: Kubernetes manifests (kustomize, local overlay) and automated drift-triggered retraining with a champion/challenger promotion gate
  - [x] Retrain orchestrator, decision logic unit-tested with fakes
  - [x] Kubernetes manifests structurally tested
  - [ ] Validated on a live cluster (manifests are rendered and schema-checked, not yet applied to a cluster)
  - [ ] Spark feature refresh bundled into the CronJob
  - [ ] Hot model reload in serving after promotion

## License

MIT
