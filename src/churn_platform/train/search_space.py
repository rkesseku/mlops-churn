"""XGBoost hyperparameter search space for Optuna.

Each parameter has a deliberate range with a comment explaining why.

Sources for these ranges:
  - XGBoost docs' tuning guide (which params matter most)
  - Industry practice from kaggle-grandmaster writeups on tabular problems
  - Our specific data characteristics: ~1M rows, ~20 features, binary
    classification with ~27% positive class

If you tighten or widen any range, leave the old value in a comment with the
date — it's surprisingly useful when you come back in 6 months and wonder
why max_depth doesn't go higher.
"""

from __future__ import annotations

from typing import Any

import optuna


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest one XGBoost parameter set for an Optuna trial.

    Returns a dict that can be passed directly to xgboost.XGBClassifier.
    """
    return {
        # ----- Boosting setup -----------------------------------------------
        # n_estimators capped by Optuna; the actual model uses early stopping
        # at training time, so this is just an upper bound. Wide range lets
        # Optuna trade tree count against learning rate.
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),

        # Learning rate. Lower = more conservative, needs more trees.
        # log scale because the useful range spans 2 orders of magnitude.
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.3, log=True
        ),

        # ----- Tree shape ---------------------------------------------------
        # Max depth: trees deeper than 8 overfit on tabular data with our
        # row count. We allow 4-10 but expect Optuna to converge around 6-8.
        "max_depth": trial.suggest_int("max_depth", 4, 10),

        # min_child_weight: minimum sum of instance weight needed in a leaf.
        # Higher = more conservative (avoids splitting on tiny subgroups).
        # 1 is XGBoost's default; we go up to 20 to test regularization.
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),

        # gamma: minimum loss reduction required to make a split.
        # Higher = more conservative pruning. 0 means no pruning.
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),

        # ----- Sampling (regularization) -----------------------------------
        # subsample: fraction of training rows per tree. <1.0 acts as
        # regularization (each tree sees a different random subset).
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),

        # colsample_bytree: fraction of columns sampled per tree.
        # Strong regularizer; values around 0.7-0.9 are typical for tabular.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        # ----- L1/L2 regularization on weights ------------------------------
        # reg_alpha: L1 regularization. Sparsity-inducing; can drive
        # unimportant features to zero contribution.
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),

        # reg_lambda: L2 regularization. Smoother shrinkage.
        # Default is 1.0; we explore both sides of that.
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

        # ----- Fixed (not searched) ----------------------------------------
        # These don't go in the search space; they're set in train.py:
        #   objective="binary:logistic"
        #   eval_metric="auc"
        #   tree_method="hist"
        #   enable_categorical=True
        #   random_state=42
    }


# Default trial count. Override in train.py via CLI / env var as needed.
# 50 trials usually finds within 0.5% AUC of the convergence point on
# tabular problems this size; 100 is safer if you have the compute budget.
DEFAULT_N_TRIALS = 50
