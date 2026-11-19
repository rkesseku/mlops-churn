"""Build brief figures from results.json (written by run_experiments.py)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, roc_auc_score

H = Path(__file__).parent; F = H / "figures"; F.mkdir(exist_ok=True)
R = json.loads((H / "results.json").read_text())
ACC, GREY, ORG, RED = "#1f4e9e", "#6b7280", "#e69f00", "#c0392b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

def box(ax, x, y, w, h, text, fc="#eef3fb", ec=ACC, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")

def arrow(ax, p, q, color=GREY, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=1.2, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))

# ---- architecture ----
fig, ax = plt.subplots(figsize=(7.2, 4.1)); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
ax.text(0.1, 6.75, "Training path", fontsize=9, weight="bold", color=ACC)
row = [("Synthetic data\ngenerate/", 0.1), ("Spark features\nfeatures/", 2.5), ("Optuna + XGBoost\ntrain/", 4.9), ("MLflow registry\naliases: staging,\nproduction", 7.3)]
for t, x in row: box(ax, x, 5.3, 2.1, 1.2, t)
for i in range(3): arrow(ax, (row[i][1] + 2.1, 5.9), (row[i + 1][1], 5.9))
box(ax, 9.7, 5.3, 2.2, 1.2, "Postgres + MinIO\n(metadata, artifacts)", fc="#f4f4f4", ec=GREY)
arrow(ax, (9.4, 5.9), (9.7, 5.9), style="<|-|>")
ax.text(0.1, 4.45, "Serving and monitoring", fontsize=9, weight="bold", color=ACC)
box(ax, 7.3, 2.95, 2.1, 1.2, "FastAPI serving\n/predict, /health,\n/metrics")
box(ax, 4.5, 2.95, 2.1, 1.2, "Prometheus")
box(ax, 1.7, 2.95, 2.1, 1.2, "Grafana\ndashboard")
arrow(ax, (8.35, 5.3), (8.35, 4.15), color=ACC); ax.text(7.4, 4.7, "load model\nat startup", fontsize=7, color=GREY, va="center")
arrow(ax, (7.3, 3.55), (6.6, 3.55)); arrow(ax, (4.5, 3.55), (3.8, 3.55))
ax.text(0.1, 2.1, "Drift-triggered retraining (Kubernetes CronJob, daily)", fontsize=9, weight="bold", color=ACC)
box(ax, 0.1, 0.35, 2.3, 1.3, "Drift check\nPSI + KS per feature", fc="#fff6e5", ec=ORG)
box(ax, 3.0, 0.35, 2.3, 1.3, "Severe drift?\n(drift gate)", fc="#fff6e5", ec=ORG)
box(ax, 5.9, 0.35, 2.3, 1.3, "Train candidate\n(Optuna + XGBoost)", fc="#fff6e5", ec=ORG)
box(ax, 8.8, 0.35, 3.1, 1.3, "Champion vs candidate\non gate holdout; promote\nif CI lower bound >= margin", fc="#fff6e5", ec=ORG)
for a, b in [(2.4, 3.0), (5.3, 5.9), (8.2, 8.8)]: arrow(ax, (a, 1.0), (b, 1.0))
arrow(ax, (10.35, 1.65), (8.9, 5.3), color=ORG, rad=-0.2); ax.text(10.9, 3.4, "promote:\nrepoint alias", fontsize=7, color=GREY, ha="center")
fig.tight_layout(); fig.savefig(F / "architecture.png", dpi=200); plt.close(fig)

# ---- drift PSI ----
sw = R["drift_sweep"]
feats = sorted({r["feature"] for r in sw["1.0"] if r["method"] == "psi"})
def psi(s): d = {r["feature"]: r["score"] for r in sw[s] if r["method"] == "psi"}; return np.array([d[f] for f in feats])
order = np.argsort(psi("1.0")); feats = [feats[i] for i in order]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]; y = np.arange(len(feats)); w = 0.2
for k, (s, lab, c) in enumerate([("0.0", "no drift (fresh sample)", "#9ca3af"), ("0.3", "strength 0.3", "#8fb0e0"), ("0.6", "strength 0.6", "#4a7cc4"), ("1.0", "strength 1.0", ACC)]):
    ax.barh(y + (k - 1.5) * w, psi(s), height=w, color=c, label=lab)
ax.axvline(0.10, color=ORG, ls="--", lw=1); ax.axvline(0.25, color=RED, ls="--", lw=1)
ax.text(0.105, -0.9, "0.10 moderate", color=ORG, fontsize=7); ax.text(0.255, -0.9, "0.25 severe", color=RED, fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=7.5); ax.set_xlabel("PSI vs. baseline"); ax.set_ylim(-1.2, len(feats))
ax.legend(fontsize=7, loc="lower right", frameon=False); ax.set_title("Per-feature PSI, covariate drift", fontsize=9, loc="left")
ax = axs[1]
sts = ["0.0", "0.3", "0.6", "1.0"]; lab = ["none", "0.3", "0.6", "1.0"]
nsev = [len({r["feature"] for r in sw[s] if r["severity"] == "severe"}) for s in sts]
nmod = [len({r["feature"] for r in sw[s] if r["severity"] == "moderate"} - {r["feature"] for r in sw[s] if r["severity"] == "severe"}) for s in sts]
ax.bar(lab, nsev, color=RED, label="severe"); ax.bar(lab, nmod, bottom=nsev, color=ORG, label="moderate (only)")
ax.set_ylim(0, max(a + b for a, b in zip(nsev, nmod)) + 1.2); ax.set_xlabel("drift strength"); ax.set_ylabel("features flagged"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Features flagged (PSI or KS)", fontsize=9, loc="left")
for i, (a, b) in enumerate(zip(nsev, nmod)): ax.text(i, a + b + 0.1, str(a + b), ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(F / "drift_psi.png", dpi=200); plt.close(fig)

# ---- training ----
y = np.array(R["roc"]["y"]); p = np.array(R["roc"]["p"]); auc = roc_auc_score(y, p)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
t = R["trial_val_auc"]; ax = axs[0]
ax.plot(range(len(t)), t, "o", color=GREY, ms=4, label="trial"); ax.plot(range(len(t)), np.maximum.accumulate(t), "-", color=ACC, label="best so far")
ax.set_xlabel("Optuna trial"); ax.set_ylabel("validation AUC"); ax.legend(fontsize=7, frameon=False); ax.set_title("Search", fontsize=9, loc="left")
ax = axs[1]; fpr, tpr, _ = roc_curve(y, p)
ax.plot(fpr, tpr, color=ACC); ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=0.8)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title(f"Test ROC (AUC {auc:.3f})", fontsize=9, loc="left")
ax = axs[2]; imp = sorted(R["importances"].items(), key=lambda kv: kv[1])[-8:]
ax.barh([k for k, _ in imp], [v for _, v in imp], color=ACC); ax.tick_params(axis="y", labelsize=7); ax.set_xlabel("importance"); ax.set_title("Top features", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(F / "training.png", dpi=200); plt.close(fig)

# ---- retrain decision flow ----
fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.set_xlim(0, 12.2); ax.set_ylim(0, 4.2); ax.axis("off")
box(ax, 0.1, 1.6, 1.9, 1.0, "Hash data windows\n+ config = run id", fs=7.5)
box(ax, 2.5, 1.6, 1.6, 1.0, "Seen before?", fc="#fff6e5", ec=ORG, fs=7.5)
box(ax, 4.6, 1.6, 1.6, 1.0, ">= min rows?", fc="#fff6e5", ec=ORG, fs=7.5)
box(ax, 6.7, 1.6, 1.8, 1.0, "Severe drift\nfeatures >= N?", fc="#fff6e5", ec=ORG, fs=7.5)
box(ax, 9.0, 1.6, 1.4, 1.0, "Train +\nevaluate", fs=7.5)
box(ax, 10.7, 1.6, 1.3, 1.0, "CI lower\n>= margin?", fc="#fff6e5", ec=ORG, fs=7.5)
for a, b in [(2.0, 2.5), (4.1, 4.6), (6.2, 6.7), (8.5, 9.0), (10.4, 10.7)]: arrow(ax, (a, 2.1), (b, 2.1))
for x, txt in [(3.3, "duplicate"), (5.4, "insufficient_data"), (7.6, "no_drift")]:
    arrow(ax, (x, 1.6), (x, 0.95), color=GREY); ax.text(x, 0.65, txt, ha="center", fontsize=7.5, color=GREY)
arrow(ax, (11.35, 1.6), (11.35, 0.95), color=RED); ax.text(11.35, 0.65, "rejected", ha="center", fontsize=7.5, color=RED)
arrow(ax, (11.35, 2.6), (11.35, 3.25), color="#2e7d32"); ax.text(11.35, 3.5, "promoted", ha="center", fontsize=7.5, color="#2e7d32")
ax.text(0.1, 3.8, "Every outcome writes a JSON decision record (plus 'dry_run' and 'error' statuses).", fontsize=7.5, color=GREY)
fig.tight_layout(); fig.savefig(F / "retrain_flow.png", dpi=200); plt.close(fig)
print("ok")
