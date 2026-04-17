"""
hpo_v9.py — Optuna Bayesian HPO for the 48h point model (V9 architecture).

Searches over learning_rate, num_leaves, max_depth, min_child_samples,
colsample_bytree, subsample, reg_alpha, reg_lambda, path_smooth, min_split_gain.

Uses same fat-tail weighting + stratified early stopping as train_v6.py.
Runs 50 trials, saves best params to models_v6/best_optuna_params_v9.json.

Runtime estimate: ~25-40 min (50 trials × 2-4k trees each).
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Install optuna first:  pip install optuna")

from data_fetcher import fetch_full_history
from features_v6 import engineer_features, classify_regime, REGIME_LABELS

MODELS_DIR = Path("models_v6")
OUTPUT_PATH = MODELS_DIR / "best_optuna_params_v9.json"
DATA_PATH   = Path("data/historical.parquet")

N_TRIALS       = 50
HORIZON        = 48
VAL_CUTOFF     = datetime(2025, 1, 1, tzinfo=None)

# ─── Load data ────────────────────────────────────────────────────────────────

print("=" * 65)
print("  V9 Optuna HPO — 48h Point Model")
print("=" * 65)
print(f"  Trials: {N_TRIALS}  |  Horizon: {HORIZON}h")
print(f"  Saving results to: {OUTPUT_PATH}")
print()

if DATA_PATH.exists():
    print("Loading cached data...")
    df = pd.read_parquet(DATA_PATH)
else:
    print("Fetching historical data...")
    df = fetch_full_history()
    df.to_parquet(DATA_PATH)

# ─── Build features once (shared across all trials) ──────────────────────────
print("Building feature matrix...")
X, y = engineer_features(df, horizon_h=HORIZON)
mask = y.notna()
X, y = X[mask], y[mask]

regime = classify_regime(df)
X["regime"] = pd.Categorical(regime.reindex(X.index).fillna(2).astype(int))
cat_features = ["regime"]

# Train/val split
val_cutoff_ts = pd.Timestamp(VAL_CUTOFF).tz_localize("America/Los_Angeles")
train_mask = X.index < val_cutoff_ts

X_train, y_train = X[train_mask].copy(), y[train_mask]
X_val,   y_val   = X[~train_mask].copy(), y[~train_mask]
X_val["regime"]   = pd.Categorical(X_val["regime"])

print(f"  Train: {len(X_train):,} rows  |  Val (2025 holdout): {len(X_val):,} rows")

# Sample weights: temporal + fat-tail
weights = np.ones(len(y_train))
weights[X_train.index.year <= 2022] = 0.5
fat_tail_mask = (X_train["aqi_current"].values > 75) | (np.abs(y_train.values) > 20)
weights[fat_tail_mask] *= 3.0

# Stratified early-stopping split (10%)
es_within = np.zeros(len(X_train), dtype=bool)
_, es_idx = train_test_split(
    np.arange(len(X_train)),
    test_size=0.10,
    random_state=42,
    stratify=X_train.index.month
)
es_within[es_idx] = True

X_fit = X_train[~es_within].copy()
y_fit = y_train[~es_within]
w_fit = weights[~es_within]
X_es  = X_train[es_within].copy()
y_es  = y_train[es_within]
w_es  = weights[es_within]

X_fit["regime"] = pd.Categorical(X_fit["regime"])
X_es["regime"]  = pd.Categorical(X_es["regime"])

print(f"  Fit rows: {len(X_fit):,}  |  ES eval rows: {len(X_es):,}")
print()

# ─── Optuna objective ─────────────────────────────────────────────────────────

trial_results = []

def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators       = 20000,
        learning_rate      = trial.suggest_float("learning_rate", 0.003, 0.015, log=True),
        num_leaves         = trial.suggest_int("num_leaves", 31, 127),
        max_depth          = trial.suggest_int("max_depth", 5, 10),
        min_child_samples  = trial.suggest_int("min_child_samples", 15, 80),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample          = trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq     = 1,
        reg_alpha          = trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        reg_lambda         = trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        path_smooth        = trial.suggest_float("path_smooth", 0.0, 2.0),
        min_split_gain     = trial.suggest_float("min_split_gain", 0.0, 0.5),
        objective          = "huber",
        alpha              = 1.5,      # Huber delta — matches V9 production
        n_jobs             = -1,
        verbosity          = -1,
        random_state       = 42,
    )

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_es, y_es)],
        eval_sample_weight=[w_es],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(period=-1),   # silence per-iteration logs
        ],
        categorical_feature=cat_features,
    )

    # Score on full 2025 holdout (the backtest proxy)
    preds = model.predict(X_val)
    mae   = mean_absolute_error(y_val, preds)
    r2    = r2_score(y_val, preds)

    trial_results.append({
        "trial": trial.number,
        "best_iter": model.best_iteration_,
        "mae": round(mae, 4),
        "r2":  round(r2,  4),
        **{k: trial.params[k] for k in trial.params},
    })

    n = len(trial_results)
    print(f"  Trial {trial.number:3d}/{N_TRIALS} | "
          f"iter={model.best_iteration_:5d} | "
          f"MAE={mae:.3f} | R2={r2:.4f} | "
          f"lr={params['learning_rate']:.4f} | "
          f"leaves={params['num_leaves']} | "
          f"path_smooth={params['path_smooth']:.2f}")

    return mae   # Optuna minimizes; MAE is our primary metric


# ─── Run study ────────────────────────────────────────────────────────────────

sampler = optuna.samplers.TPESampler(seed=42)
study   = optuna.create_study(direction="minimize", sampler=sampler)

print(f"Launching {N_TRIALS} Bayesian trials...")
print("-" * 65)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print("-" * 65)

# ─── Report ───────────────────────────────────────────────────────────────────

best = study.best_trial
print(f"\nBest trial: #{best.number}")
print(f"  MAE  = {best.value:.4f}")
print(f"  R2   = {[t for t in trial_results if t['trial']==best.number][0]['r2']:.4f}")
print(f"  Params:")
for k, v in best.params.items():
    print(f"    {k:25s} = {v}")

# ─── Save to JSON for train_v6.py to consume ─────────────────────────────────

best_r2   = [t["r2"]  for t in trial_results if t["trial"] == best.number][0]
best_iter = [t["best_iter"] for t in trial_results if t["trial"] == best.number][0]

# Load existing v9 params if any
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        existing = json.load(f)
else:
    existing = {}

existing["48h"] = {
    "point": {
        "best_params": best.params,
        "best_mae": round(best.value, 4),
        "best_r2":  round(best_r2, 4),
        "best_iter": best_iter,
        "n_trials": N_TRIALS,
    }
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(existing, f, indent=2)

print(f"\nSaved best params to: {OUTPUT_PATH}")

# Also dump full trial log
trial_df = pd.DataFrame(trial_results).sort_values("mae")
trial_df.to_csv(MODELS_DIR / "hpo_v9_trial_log.csv", index=False)
print(f"Full trial log saved to: {MODELS_DIR / 'hpo_v9_trial_log.csv'}")
print("\nHPO complete. Now update train_v6.py to load from best_optuna_params_v9.json and retrain.")
