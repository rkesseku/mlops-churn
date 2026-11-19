"""Run the real pipeline end to end at small scale and dump results to results.json.

All artefacts (data, MLflow store, decision records) go to a temp dir, not the repo.
Features are computed with the repo's Spark transforms; the splits are written with
pandas because Spark's parquet writer needs winutils on Windows.
"""
import json, os, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
TMP = Path(tempfile.mkdtemp(prefix="churn_brief_"))
os.chdir(TMP)  # MLflow artifact root is relative to cwd
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(TMP / 'mlflow.db').as_posix()}"

from churn_platform.generate.synthesize import GenParams, generate  # noqa: E402
from churn_platform.monitor.drift import score_drift  # noqa: E402

N = 30_000
out = {"n_rows": N}
base = generate(GenParams(n_rows=N, seed=42))
base2 = generate(GenParams(n_rows=N, seed=7))  # fresh baseline sample, no drift
out["churn_rate_base"] = float(base.churned.mean())

# ---- drift sweeps (covariate drift at several strengths) ----
sweep = {}
for s in (0.0, 0.3, 0.6, 1.0):
    cur = generate(GenParams(n_rows=N, seed=7, drift_mode="covariate", drift_strength=s)) if s else base2
    rep = score_drift(base, cur)
    sweep[str(s)] = [dict(feature=r.feature, method=r.method, score=r.score, pvalue=r.pvalue, severity=r.severity)
                     for r in rep.results]
out["drift_sweep"] = sweep
for mode in ("label", "concept"):
    cur = generate(GenParams(n_rows=N, seed=7, drift_mode=mode, drift_strength=0.6))
    rep = score_drift(base, cur)
    out[f"drift_{mode}_severe"] = rep.severe_features
    out[f"drift_{mode}_moderate"] = rep.moderate_features
    out[f"churn_rate_{mode}"] = float(cur.churned.mean())

# ---- Spark features (repo transforms) -> pandas parquet ----
raw = TMP / "raw"; raw.mkdir()
base.to_parquet(raw / "base.parquet")
from churn_platform.features.pipeline import SPLIT_NAMES, build_spark, split, transform  # noqa: E402
spark = build_spark(); spark.sparkContext.setLogLevel("ERROR")
sdf = transform(spark.read.parquet(str(raw / "base.parquet")))
parts = split(sdf)
feat = TMP / "features"; feat.mkdir()
for name, p in zip(SPLIT_NAMES, parts, strict=True):
    p.toPandas().to_parquet(feat / f"{name}.parquet")
spark.stop()
split_sizes = {n: len(pd.read_parquet(feat / f"{n}.parquet")) for n in SPLIT_NAMES}
out["split_sizes"] = split_sizes
out["n_raw_cols"] = base.shape[1]
out["n_feature_cols"] = pd.read_parquet(feat / "train.parquet").shape[1]

# ---- training (real train.run) ----
from churn_platform.train import train as T  # noqa: E402
import mlflow  # noqa: E402
t0 = time.time()
v = T.run(n_trials=8, sample_size=15_000, features_dir=feat)
out["champion_train_seconds"] = time.time() - t0
client = mlflow.tracking.MlflowClient()
runs = client.search_runs([client.get_experiment_by_name("churn-xgboost").experiment_id])
trials = sorted([r for r in runs if r.data.tags.get("mlflow.parentRunId")], key=lambda r: r.info.start_time)
out["trial_val_auc"] = [r.data.metrics["val_auc"] for r in trials]
parent = [r for r in runs if not r.data.tags.get("mlflow.parentRunId")][0]
out["parent_metrics"] = parent.data.metrics
out["best_params"] = {k[5:]: v for k, v in parent.data.params.items() if k.startswith("best_")}
# feature importances + holdout curve
import mlflow.xgboost  # noqa: E402
model = mlflow.xgboost.load_model(f"models:/{T.MODEL_NAME}/{v}")
imp = dict(zip(model.get_booster().feature_names, [float(x) for x in model.feature_importances_]))
out["importances"] = imp
Xte, yte = T._prepare_features(pd.read_parquet(feat / "test.parquet"))
Xte = Xte[list(model.get_booster().feature_names)]
np.save(TMP / "y.npy", yte.to_numpy()); np.save(TMP / "p.npy", model.predict_proba(Xte)[:, 1])
out["roc"] = {"y": yte.tolist(), "p": [float(x) for x in model.predict_proba(Xte)[:, 1]]}

# ---- retrain scenarios (real orchestrator + real MLflow backend) ----
from churn_platform.retrain.adapters import MlflowBackend  # noqa: E402
from churn_platform.retrain.decision import RetrainConfig  # noqa: E402
from churn_platform.retrain.orchestrator import DecisionStore, run_cycle  # noqa: E402
store = DecisionStore(TMP / "decisions")
now = datetime.now(timezone.utc)
drifted = generate(GenParams(n_rows=N, seed=7, drift_mode="covariate", drift_strength=0.8))
cfgA = RetrainConfig(min_rows=1000)
strong = MlflowBackend(feat, n_trials=8, sample_size=15_000)
weak = MlflowBackend(feat, n_trials=1, sample_size=1_500)
scen = []
def rec(name, r):
    d = dict(name=name, status=r.status, detail=r.detail, champion=r.champion_version, candidate=r.candidate_version,
             severe=(r.drift or {}).get("severe_features"), promotion=r.promotion)
    scen.append(d); print(name, r.status, r.detail, flush=True)
rec("1. no drift", run_cycle(base, base2, cfgA, strong, store, now=now))
rec("2. drift, dry run", run_cycle(base, drifted, RetrainConfig(dry_run=True), strong, store, now=now))
rec("3. drift, weaker candidate, margin 0.002", run_cycle(base, drifted, cfgA, weak, store, now=now))
cfg0 = RetrainConfig(promotion_margin=0.0)
rec("4. drift, same-recipe candidate, margin 0", run_cycle(base, drifted, cfg0, strong, store, now=now))
rec("5. identical inputs re-run", run_cycle(base, drifted, cfg0, strong, store, now=now))
out["scenarios"] = scen
out["champion_after"] = strong.champion_version()
(HERE / "results.json").write_text(json.dumps(out))
print("done", TMP)
