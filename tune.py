"""
tune.py — Bayesian hyperparameter optimization for all 12 LightGBM models.

Runs 12 sequential Optuna studies (4 horizons × 3 model types: point, q01, q99).
Each study uses temporal rolling CV with 300-tree fast search, then retrains the
winning params with full tree counts (4000 point / 1500 quantile) and saves to models/.

Non-destructive: does NOT modify train.py, validate.py, features.py, data_fetcher.py,
api.py, or inference.py.

Usage:
    python tune.py
"""

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# ─── Constants ────────────────────────────────────────────────────────────────

HORIZONS = [6, 12, 24, 48]                 # forecast horizons in hours
MODEL_TYPES = ["point", "q01", "q99"]       # 3 models per horizon

# Phase 1 (search): small tree count for fast per-trial evaluation
TUNE_N_ESTIMATORS = 300

# Phase 2 (final retrain): full tree counts matching train.py
FINAL_POINT_N_ESTIMATORS = 4000
FINAL_QUANTILE_N_ESTIMATORS = 1500

# Temporal CV configuration
N_CV_FOLDS = 10          # number of rolling folds
WINDOW_DAYS = 365        # training window width in days
TEST_DAYS = 7            # test window width in days (also the slide stride)

# Optuna configuration
N_TRIALS = 500           # max trials per study
PATIENCE = 30            # early stop after this many stagnant trials
MIN_DELTA = 1e-4         # minimum improvement to reset patience counter

# Hardware: AMD Ryzen 7 9800X3D — 8C/16T, 96MB L3
NUM_THREADS = 16         # saturate all hardware threads inside LightGBM

# Fixed quantile alphas (V4.0: widened from 0.01/0.99 for better coverage)
QUANTILE_ALPHA_LOWER = 0.005   # 0.5th percentile (saved as lgbm_q05_*.pkl — legacy name)
QUANTILE_ALPHA_UPPER = 0.995   # 99.5th percentile (saved as lgbm_q95_*.pkl — legacy name)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
HIST_PATH = DATA_DIR / "historical.parquet"


# ─── Startup checks ──────────────────────────────────────────────────────────

def _check_prerequisites():
    """Verify data file and features module are available before doing anything."""
    if not HIST_PATH.exists():
        print(f"[tune] ERROR: {HIST_PATH} not found. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    try:
        from features import engineer_features  # noqa: F401
    except ImportError as exc:
        print(f"[tune] ERROR: Cannot import features.py: {exc}", file=sys.stderr)
        sys.exit(1)


# ─── Early stopping callback ─────────────────────────────────────────────────

class EarlyStoppingCallback:
    """
    Stops an Optuna study when best_value hasn't improved by more than
    min_delta for `patience` consecutive completed trials.

    Uses study.stop() — NOT TrialPruned — because TrialPruned only prunes
    a single trial and does not halt the study.
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float('inf')
        self._no_improve = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # Only react to completed trials (skip pruned/failed)
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_value < self._best - self.min_delta:
            self._best = study.best_value
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            print(f"[tune] Early stop: no improvement for {self.patience} trials "
                  f"(study={study.study_name}, best={self._best:.4f})")
            study.stop()


# ─── Pinball loss ─────────────────────────────────────────────────────────────

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Mean pinball (quantile) loss.

    For each sample i:
        if y_true[i] >= y_pred[i]:  loss_i = alpha * (y_true[i] - y_pred[i])
        else:                        loss_i = (1 - alpha) * (y_pred[i] - y_true[i])

    Returns the mean loss across all samples.
    """
    diff = y_true - y_pred
    loss = np.where(diff >= 0, alpha * diff, (1 - alpha) * (-diff))
    return float(np.mean(loss))


# ─── Objective factory ────────────────────────────────────────────────────────

def _get_quantile_alpha(model_type: str) -> float:
    """Map model_type string to the fixed quantile alpha."""
    if model_type == "q01":
        return QUANTILE_ALPHA_LOWER
    elif model_type == "q99":
        return QUANTILE_ALPHA_UPPER
    else:
        raise ValueError(f"Not a quantile model type: {model_type}")


def make_objective(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    model_type: str,
    horizon_h: int,
):
    """
    Factory that returns a closure suitable for optuna study.optimize().

    Args:
        X_all:       Full feature matrix (DatetimeIndex, all rows with notna target).
        y_all:       Full residual target series, aligned with X_all.
        fold_splits: Pre-computed list of (train_indices, test_indices) for temporal CV.
        model_type:  One of "point", "q01", "q99".
        horizon_h:   Forecast horizon in hours (used for adaptive search space).

    The closure captures these arguments by reference; no global mutable state is used.
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Single Optuna trial: suggest hyperparameters, run N_CV_FOLDS temporal folds,
        return mean metric (MAE for point, pinball for quantile) in absolute AQI space.
        """
        # ── Suggest hyperparameters (horizon-adaptive search space) ──
        # Long horizons (24h, 48h) use a constrained space because signal-to-noise
        # is low — complex models overfit noise. Short horizons (6h, 12h) have
        # strong local signal and benefit from deeper, wider trees.
        if horizon_h <= 12:
            # Short horizon: wide search space, high capacity allowed
            params = {
                "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
                "max_depth":        trial.suggest_int("max_depth", 4, 12),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq":     1,
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        else:
            # Long horizon: constrained space to prevent overfitting
            params = {
                "num_leaves":       trial.suggest_int("num_leaves", 15, 63),
                "max_depth":        trial.suggest_int("max_depth", 3, 6),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 200),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq":     1,
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

        # Fixed hardware params
        params.update({
            "n_jobs":       -1,
            "num_threads":  NUM_THREADS,
            "verbosity":    -1,
            "n_estimators": TUNE_N_ESTIMATORS,
        })

        # ── Add fixed objective-specific params ──
        if model_type == "point":
            params["objective"] = "huber"
            params["alpha"] = 2.0               # Huber delta, NOT a quantile
        else:
            params["objective"] = "quantile"
            params["alpha"] = _get_quantile_alpha(model_type)

        # ── Cross-validation loop ──
        fold_scores = []
        fold_coverages = []  # track one-sided coverage for quantile models

        for train_idx, test_idx in fold_splits:
            X_train_fold = X_all.iloc[train_idx]
            y_train_fold = y_all.iloc[train_idx]
            X_test_fold = X_all.iloc[test_idx]
            y_test_fold = y_all.iloc[test_idx]

            if len(X_train_fold) < 200 or len(X_test_fold) == 0:
                continue

            # Refit imputer on this fold's training data only (no leakage)
            imp = SimpleImputer(strategy="median")
            X_tr_imp = pd.DataFrame(
                imp.fit_transform(X_train_fold),
                columns=X_train_fold.columns,
            )
            X_te_imp = pd.DataFrame(
                imp.transform(X_test_fold),
                columns=X_test_fold.columns,
            )

            # Train a fresh model with the trial's hyperparameters
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr_imp, y_train_fold)

            # Predict residuals and INVERT to absolute AQI space (FACT 1)
            pred_residual = model.predict(X_te_imp)
            base_aqi = X_test_fold["aqi_current"].values
            pred_abs = pred_residual + base_aqi
            y_abs = y_test_fold.values + base_aqi

            # Compute metric in absolute AQI space
            if model_type == "point":
                score = mean_absolute_error(y_abs, pred_abs)
            else:
                alpha = _get_quantile_alpha(model_type)
                score = pinball_loss(y_abs, pred_abs, alpha)

                # ── Coverage-aware penalty (V4.0) ──
                # Compute one-sided coverage for this quantile:
                #   q01 (lower bound): fraction of actuals ABOVE the prediction
                #   q99 (upper bound): fraction of actuals BELOW the prediction
                # Target coverage per side: 1 - alpha (e.g., 0.995 for α=0.005)
                if model_type == "q01":
                    # Lower bound: actual should be >= predicted
                    one_sided_cov = float(np.mean(y_abs >= pred_abs))
                else:  # q99
                    # Upper bound: actual should be <= predicted
                    one_sided_cov = float(np.mean(y_abs <= pred_abs))
                fold_coverages.append(one_sided_cov)

            fold_scores.append(score)

        if not fold_scores:
            # No valid folds — return a large penalty so Optuna avoids this region
            return 1e6

        mean_score = float(np.mean(fold_scores))

        # ── Add coverage penalty for quantile models ──
        # If actual coverage drops below 90% (combined two-sided equivalent),
        # penalize quadratically. This steers the sampler toward intervals
        # wide enough to meet the coverage target.
        if model_type != "point" and fold_coverages:
            COVERAGE_TARGET = 0.90
            mean_coverage = float(np.mean(fold_coverages))
            coverage_shortfall = max(0.0, COVERAGE_TARGET - mean_coverage)
            penalty = 100.0 * coverage_shortfall ** 2
            mean_score += penalty

        return mean_score

    return objective


# ─── CV fold construction ────────────────────────────────────────────────────

def build_temporal_folds(
    X_all: pd.DataFrame, n_folds: int = N_CV_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build N_CV_FOLDS rolling temporal CV splits.

    Each fold:
      - Train: WINDOW_DAYS of data ending at fold_end
      - Test:  TEST_DAYS immediately after the training window
      - Both windows slide forward by TEST_DAYS (7 days) per fold

    The last fold's test window ends at the dataset's final timestamp.
    Folds are anchored from the END of the dataset backwards to guarantee
    the most recent data is always included in at least one test fold.

    Returns list of (train_positional_indices, test_positional_indices).
    """
    index = X_all.index
    # FIX (Hyperparameter Leakage): Enforce hard cutoff at 2026-01-25.
    # The Optuna CV must not see any data that validate.py uses for evaluation.
    hard_cutoff = pd.Timestamp("2026-01-25").tz_localize(index.tz)
    data_end = min(index.max(), hard_cutoff)

    # Walk backwards from data_end to anchor the folds
    folds = []
    for i in range(n_folds):
        # How many fold-widths back from the end this fold sits
        offset = (n_folds - 1 - i) * TEST_DAYS
        test_end = data_end - timedelta(days=offset)
        test_start = test_end - timedelta(days=TEST_DAYS)
        train_end = test_start
        train_start = train_end - timedelta(days=WINDOW_DAYS)

        train_mask = (index >= train_start) & (index < train_end)
        test_mask = (index >= test_start) & (index < test_end)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))

    print(f"[tune] Built {len(folds)} temporal CV folds "
          f"(window={WINDOW_DAYS}d train, {TEST_DAYS}d test, {TEST_DAYS}d stride)")
    if folds:
        first_train_start = index[folds[0][0][0]]
        last_test_end = index[folds[-1][1][-1]]
        print(f"[tune] Fold range: {first_train_start} → {last_test_end}")

    return folds


# ─── Baseline computation ────────────────────────────────────────────────────

def _default_params(model_type: str, horizon_h: int) -> dict:
    """
    Return train.py's default hyperparameters for a given model type and horizon.
    These serve as the baseline for the comparison table.
    """
    if model_type == "point":
        # Replicate _point_params from train.py
        params = {
            "objective": "huber",
            "alpha": 2.0,
            "n_estimators": TUNE_N_ESTIMATORS,  # Use fast tree count for fair comparison
            "learning_rate": 0.01,
            "n_jobs": -1,
            "num_threads": NUM_THREADS,
            "verbosity": -1,
        }
        if horizon_h <= 12:
            params.update(
                num_leaves=63, max_depth=8, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
            )
        else:
            params.update(
                num_leaves=63, max_depth=7, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0,
            )
    else:
        # Replicate _quantile_params from train.py
        alpha = _get_quantile_alpha(model_type)
        params = {
            "objective": "quantile",
            "alpha": alpha,
            "n_estimators": TUNE_N_ESTIMATORS,
            "learning_rate": 0.01,
            "n_jobs": -1,
            "num_threads": NUM_THREADS,
            "verbosity": -1,
        }
        if horizon_h <= 12:
            params.update(num_leaves=31, max_depth=6)
        else:
            params.update(num_leaves=15, max_depth=4)

    return params


def compute_baseline(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    model_type: str,
    horizon_h: int,
) -> float:
    """
    Compute baseline metric using train.py's default hyperparameters on the
    same temporal CV folds. This gives a fair apples-to-apples comparison
    against the Optuna-tuned params.
    """
    params = _default_params(model_type, horizon_h)
    fold_scores = []

    for train_idx, test_idx in fold_splits:
        X_train_fold = X_all.iloc[train_idx]
        y_train_fold = y_all.iloc[train_idx]
        X_test_fold = X_all.iloc[test_idx]
        y_test_fold = y_all.iloc[test_idx]

        if len(X_train_fold) < 200 or len(X_test_fold) == 0:
            continue

        imp = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(
            imp.fit_transform(X_train_fold), columns=X_train_fold.columns
        )
        X_te_imp = pd.DataFrame(
            imp.transform(X_test_fold), columns=X_test_fold.columns
        )

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr_imp, y_train_fold)

        pred_residual = model.predict(X_te_imp)
        base_aqi = X_test_fold["aqi_current"].values
        pred_abs = pred_residual + base_aqi
        y_abs = y_test_fold.values + base_aqi

        if model_type == "point":
            score = mean_absolute_error(y_abs, pred_abs)
        else:
            alpha = _get_quantile_alpha(model_type)
            score = pinball_loss(y_abs, pred_abs, alpha)

        fold_scores.append(score)

    return float(np.mean(fold_scores)) if fold_scores else float("inf")


# ─── Final retrain ────────────────────────────────────────────────────────────

def retrain_final_model(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    best_params: dict,
    model_type: str,
    horizon_h: int,
):
    """
    Phase 2: retrain ONE final model on ALL available data using the best
    hyperparameters from Optuna, but with full tree counts (4000 point / 1500 quantile).

    Saves the model and imputer to models/ using train.py's filename convention.
    """
    # Determine final tree count
    if model_type == "point":
        n_estimators = FINAL_POINT_N_ESTIMATORS
    else:
        n_estimators = FINAL_QUANTILE_N_ESTIMATORS

    # Build final params: start from Optuna best, override tree count and fixed params
    final_params = dict(best_params)
    final_params["n_estimators"] = n_estimators
    final_params["n_jobs"] = -1
    final_params["num_threads"] = NUM_THREADS
    final_params["verbosity"] = -1
    final_params["bagging_freq"] = 1

    # Set the correct objective + alpha (these are NOT tuned)
    if model_type == "point":
        final_params["objective"] = "huber"
        final_params["alpha"] = 2.0
    else:
        final_params["objective"] = "quantile"
        final_params["alpha"] = _get_quantile_alpha(model_type)

    # Fit imputer on all data
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X_all), columns=X_all.columns)

    # Train on everything
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_imp, y_all, callbacks=[lgb.log_evaluation(500)])

    # Save model and imputer using train.py's naming convention
    # Point: lgbm_point_{h}h.pkl   Quantile: lgbm_q05_{h}h.pkl / lgbm_q95_{h}h.pkl
    if model_type == "point":
        model_filename = f"lgbm_point_{horizon_h}h.pkl"
    elif model_type == "q01":
        model_filename = f"lgbm_q05_{horizon_h}h.pkl"     # legacy filename
    else:
        model_filename = f"lgbm_q95_{horizon_h}h.pkl"     # legacy filename

    model_path = MODELS_DIR / model_filename
    joblib.dump(model, model_path)
    print(f"[tune] Final model saved → {model_path}")

    imputer_path = MODELS_DIR / f"imputer_{horizon_h}h.pkl"
    joblib.dump(imp, imputer_path)
    print(f"[tune] Imputer saved → {imputer_path}")


# ─── Main orchestrator ────────────────────────────────────────────────────────

def run_study(
    df: pd.DataFrame,
    horizon_h: int,
    model_type: str,
) -> dict:
    """
    Run one Optuna study for a single (horizon, model_type) combination.

    Returns a dict with: best hyperparameters, best score, baseline score,
    number of completed trials, and whether early stopping fired.
    """
    from features import engineer_features

    study_name = f"aqi_{horizon_h}h_{model_type}"
    print(f"\n{'=' * 60}")
    print(f"  Study: {study_name}")
    print(f"{'=' * 60}")

    # ── Build features once for this horizon ──
    X, y = engineer_features(df, horizon_h)
    mask = y.notna()
    X, y = X[mask], y[mask]
    print(f"[tune] Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} cols")

    # ── Build temporal CV folds (shared across baseline + all trials) ──
    fold_splits = build_temporal_folds(X)

    # ── Compute baseline metric (train.py defaults, same CV, same tree count) ──
    print(f"[tune] Computing baseline ({model_type}, {horizon_h}h)...")
    t0 = time.time()
    baseline_score = compute_baseline(X, y, fold_splits, model_type, horizon_h)
    print(f"[tune] Baseline score: {baseline_score:.4f}  ({time.time() - t0:.1f}s)")

    # ── Create Optuna study ──
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",       # both MAE and pinball loss are minimized
        sampler=sampler,
    )

    objective_fn = make_objective(X, y, fold_splits, model_type, horizon_h)
    early_stop_cb = EarlyStoppingCallback(patience=PATIENCE, min_delta=MIN_DELTA)

    # ── Run optimization ──
    t0 = time.time()
    study.optimize(
        objective_fn,
        n_trials=N_TRIALS,
        callbacks=[early_stop_cb],
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    # Determine if early stopping fired (fewer trials completed than max)
    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    early_stopped = n_completed < N_TRIALS

    best_score = study.best_value
    best_params = study.best_params
    improvement_pct = ((baseline_score - best_score) / baseline_score * 100
                       if baseline_score > 0 else 0.0)

    print(f"[tune] Best score: {best_score:.4f}  "
          f"(improvement: {improvement_pct:+.1f}%)  "
          f"({n_completed} trials in {elapsed:.0f}s)")

    # ── Phase 2: retrain final model with full tree count ──
    print(f"[tune] Retraining final model with full trees...")
    retrain_final_model(X, y, best_params, model_type, horizon_h)

    return {
        "best_params": best_params,
        "best_cv_score": round(best_score, 4),
        "baseline_cv_score": round(baseline_score, 4),
        "improvement_pct": round(improvement_pct, 1),
        "n_trials_completed": n_completed,
        "early_stopped": early_stopped,
    }


def main():
    """Entry point: run all 12 studies sequentially and print summary."""
    _check_prerequisites()

    # Suppress Optuna's verbose default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 60)
    print("  Folsom AQI — Optuna Hyperparameter Tuning")
    print("=" * 60)

    # Load historical data from disk (train.py must have been run already)
    print(f"\nLoading data from {HIST_PATH}...")
    df = pd.read_parquet(HIST_PATH)
    print(f"  {len(df):,} rows  |  {df.index.min()} → {df.index.max()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run all 12 studies sequentially ──
    # Sequential because LightGBM already saturates all threads via n_jobs=-1.
    # Nested parallelism would cause thread oversubscription on the 9800X3D.
    all_results: dict[str, dict] = {}
    summary_rows = []
    total_start = time.time()

    for horizon_h in HORIZONS:
        horizon_key = f"{horizon_h}h"
        all_results[horizon_key] = {}

        for model_type in MODEL_TYPES:
            result = run_study(df, horizon_h, model_type)
            all_results[horizon_key][model_type] = result

            summary_rows.append({
                "horizon": horizon_key,
                "type": model_type,
                "baseline": result["baseline_cv_score"],
                "best": result["best_cv_score"],
                "improvement": result["improvement_pct"],
                "trials": result["n_trials_completed"],
                "early_stop": "yes" if result["early_stopped"] else "no",
            })

    total_elapsed = time.time() - total_start

    # ── Save best params JSON ──
    params_path = MODELS_DIR / "best_optuna_params.json"
    with open(params_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[tune] Best params saved → {params_path}")

    # ── Print comparison table ──
    print(f"\n{'=' * 90}")
    print("  OPTUNA TUNING RESULTS — BASELINE vs TUNED")
    print(f"{'=' * 90}")
    header = (f"  {'Horizon':<10} {'Type':<8} {'Baseline':<12} {'Tuned':<12} "
              f"{'Improve%':<12} {'Trials':<10} {'EarlyStop'}")
    print(header)
    print(f"  {'-' * 85}")

    for row in summary_rows:
        improvement_str = f"{row['improvement']:+.1f}%"
        status = "✓" if row["improvement"] > 0 else "—"
        print(f"  {status} {row['horizon']:<9} {row['type']:<8} "
              f"{row['baseline']:<12.4f} {row['best']:<12.4f} "
              f"{improvement_str:<12} {row['trials']:<10} {row['early_stop']}")

    print(f"\n  Total wall time: {total_elapsed / 60:.1f} minutes")
    print(f"\n✓ Tuning complete. All 12 models retrained with best params.")
    print(f"  Next: run python validate.py to verify accuracy.")


if __name__ == "__main__":
    main()
