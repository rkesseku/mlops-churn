# MLOps Churn Platform

End-to-end machine learning system for predicting customer churn — synthetic
data generation, feature engineering on Spark, gradient boosting with
Optuna-tuned hyperparameters, MLflow experiment tracking and model registry,
FastAPI serving, and Grafana-based drift monitoring with automated retraining.

## Why this exists

Most ML projects stop at a notebook with a confusion matrix. Production ML
systems have to answer harder questions: How do we retrain when the data
distribution shifts? How do we know which model is in production? How do we
roll back safely? How do we detect silent degradation before customers do?

This project demonstrates the full MLOps lifecycle on a high-business-value
problem — churn prediction directly drives retention spend — with every stage
wired together and reproducible from a single `docker compose up`.

## Status

Built incrementally. Each milestone has an acceptance test that must pass
before the next begins.

- [x] Milestone 1 — Repo skeleton, Git, CI
- [ ] Milestone 2 — Docker Compose stack: Postgres + MinIO + MLflow
- [ ] Milestone 3 — Synthetic data generator
- [ ] Milestone 4 — Spark feature pipeline
- [ ] Milestone 5 — Training with Optuna + MLflow tracking
- [ ] Milestone 6 — FastAPI serving from MLflow registry
- [ ] Milestone 7 — Prometheus + Grafana wiring
- [ ] Milestone 8 — Drift detection + retrain trigger

## License

MIT
