# Makefile for common project commands.
# Run `make` or `make help` to see all available targets.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Tunable defaults — override on the command line, e.g. `make data ROWS=100000`.
PYTHON       ?= python
COMPOSE      ?= docker compose
ROWS         ?= 1000000
SEED         ?= 42
DRIFT        ?= none
DRIFT_STR    ?= 0.0
OUT          ?= data/raw/customers.parquet

.PHONY: help
help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' \
	     $(MAKEFILE_LIST)

# ---------- Infrastructure ----------

.PHONY: up
up:  ## Bring up the docker-compose stack (Postgres, MinIO, MLflow)
	$(COMPOSE) up -d
	@echo
	@echo "MLflow:  http://localhost:5000"
	@echo "MinIO:   http://localhost:9001  (minioadmin / minioadmin)"

.PHONY: down
down:  ## Stop the stack (volumes preserved)
	$(COMPOSE) down

.PHONY: clean
clean:  ## Stop AND delete all docker volumes (wipes MLflow + MinIO data)
	$(COMPOSE) down -v

.PHONY: ps
ps:  ## Show container status
	$(COMPOSE) ps

.PHONY: logs
logs:  ## Tail logs for all services. Override SERVICE=mlflow for one.
	$(COMPOSE) logs -f $(SERVICE)

# ---------- Data ----------

.PHONY: data
data:  ## Generate synthetic data. Override ROWS, SEED, DRIFT, DRIFT_STR, OUT.
	$(PYTHON) -m churn_platform.generate.synthesize \
		--rows $(ROWS) \
		--seed $(SEED) \
		--drift $(DRIFT) \
		--drift-strength $(DRIFT_STR) \
		--out $(OUT)

.PHONY: data-baseline
data-baseline:  ## Generate 1M-row baseline training data (no drift)
	$(MAKE) data ROWS=1000000 OUT=data/raw/customers_baseline.parquet

.PHONY: data-drifted
data-drifted:  ## Generate 100K-row covariate-drifted scoring data for the drift demo
	$(MAKE) data ROWS=100000 DRIFT=covariate DRIFT_STR=0.5 \
	       OUT=data/raw/customers_drifted.parquet

.PHONY: data-sample
data-sample:  ## Generate a tiny 1K-row sample for tests, commit-safe
	$(MAKE) data ROWS=1000 OUT=data/samples/customers_sample.parquet

# ---------- Serving ----------

.PHONY: serve
serve:  ## Bring up the FastAPI serving container (rebuild if model changed)
	docker compose up -d --build serving
	@echo
	@echo "Serving:    http://localhost:8000"
	@echo "API docs:   http://localhost:8000/docs"
	@echo "Health:     http://localhost:8000/health"

.PHONY: serve-logs
serve-logs:  ## Tail the serving container's logs
	docker compose logs -f serving

# ---------- Training ----------

.PHONY: train
train:  ## Train XGBoost with Optuna (50 trials) and register in MLflow
	@set -a; . ./.env; set +a; \
		python -m churn_platform.train.train

.PHONY: train-quick
train-quick:  ## Quick 5-trial training run for smoke-testing the pipeline
	@set -a; . ./.env; set +a; \
		python -m churn_platform.train.train --n-trials 5 --sample-size 50000

# ---------- Features ----------

.PHONY: features
features:  ## Run the Spark feature pipeline (1M rows -> train/val/test parquet)
	docker compose exec spark spark-submit \
		/opt/jobs/src/churn_platform/features/pipeline.py \
		--input /opt/jobs/data/raw/customers.parquet \
		--output /opt/jobs/data/features

.PHONY: features-clean
features-clean:  ## Delete generated feature parquets (uses sudo because Spark writes as a different uid)
	sudo rm -rf data/features

# ---------- Quality ----------

.PHONY: test
test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:  ## Lint with ruff
	ruff check .

.PHONY: format
format:  ## Auto-format with ruff
	ruff format .

# ---------- Coming in later phases ----------
# features      Run Spark feature engineering job
# train         Train an XGBoost model with Optuna and register in MLflow
# serve         Start the FastAPI serving container
# monitor-up    Start Prometheus + Grafana + drift exporter
# demo-drift    End-to-end drift demo
