"""
train_v6.py — V6 Hybrid Physics-Informed Training Pipeline.
Identical to train.py but uses features_v6 (95 features + regime categorical).

Key differences from train.py:
  1. Imports from features_v6 instead of features.
  2. Calls classify_regime() and injects 'regime' as a pd.Categorical column.
  3. Saves models to models_v6/ to avoid overwriting production models.
  4. Uses existing best_optuna_params.json from V4 for hyperparameters.

Expected runtime: 10–20 minutes on a modern laptop.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:
    InconsistentVersionWarning = UserWarning

from data_fetcher import fetch_full_history
from features_v6 import engineer_features, get_feature_names, classify_regime, REGIME_LABELS

# ─── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models_v6")
DATA_DIR   = Path("data")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Read Optuna params from the V4 production directory
V4_MODELS_DIR = Path("models")

HORIZONS = [6, 12, 24, 48]

# ─── Optuna parameter ingestion ───────────────────────────────────────────────

OPTUNA_PARAMS_PATH = V4_MODELS_DIR / "best_optuna_params.json"


def _load_optuna_params() -> dict | None:
    if not OPTUNA_PARAMS_PATH.exists():
        return None
    try:
        with open(OPTUNA_PARAMS_PATH) as f:
            data = json.load(f)
        print(f"[train_v6] ✓ Loaded Optuna-tuned params from {OPTUNA_PARAMS_PATH}")
        return data
    except Exception as exc:
        print(f"[train_v6] WARNING: Could not load {OPTUNA_PARAMS_PATH}: {exc}",
              file=sys.stderr)
        return None


def _get_optuna_best(horizon_h: int, model_type: str) -> dict | None:
    if _OPTUNA_PARAMS is None:
        return None
    horizon_key = f"{horizon_h}h"
    entry = _OPTUNA_PARAMS.get(horizon_key, {}).get(model_type, {})
    best = entry.get("best_params")
    if best:
        print(f"[train_v6] Using Optuna-tuned params for {model_type} {horizon_key}")
    return best


_OPTUNA_PARAMS = _load_optuna_params()

# ─── LightGBM hyperparameters ─────────────────────────────────────────────────


def _point_params(horizon_h: int) -> dict:
    tuned = _get_optuna_best(horizon_h, "point")
    if tuned is not None:
        # V8: For 48h, use MAE (Pinball α=0.5) instead of Huber to give
        # equal gradient weight to fat-tail smoke events (no outlier dampening)
        if horizon_h >= 48:
            obj, alpha = 'mae', None
        else:
            obj, alpha = 'huber', 2.0
        params = {
            'objective':    obj,
            'n_estimators': 10000,
            'n_jobs':       -1,
            'verbosity':    -1,
            'random_state': 42,
            'bagging_freq': 1,
            **tuned,
        }
        if alpha is not None:
            params['alpha'] = alpha
        return params
    base = dict(
        n_estimators=10000,
        learning_rate=0.01, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1,
    )
    if horizon_h <= 12:
        base.update(objective='huber', alpha=2.0,
                    num_leaves=63, max_depth=8, min_child_samples=20,
                    reg_alpha=0.1, reg_lambda=1.0)
    elif horizon_h <= 24:
        base.update(objective='huber', alpha=2.0,
                    num_leaves=63, max_depth=7, min_child_samples=40,
                    reg_alpha=1.0, reg_lambda=2.0)
    else:
        # V8: MAE for 48h + deeper trees to exploit momentum features
        base.update(objective='mae',
                    num_leaves=63, max_depth=7, min_child_samples=40,
                    reg_alpha=1.0, reg_lambda=2.0)
    return base


def _quantile_params(alpha: float, horizon_h: int) -> dict:
    model_type = "q01" if alpha < 0.5 else "q99"
    tuned = _get_optuna_best(horizon_h, model_type)
    if tuned is not None:
        return {
            'objective':    'quantile',
            'alpha':        alpha,
            'n_estimators': 10000,
            'n_jobs':       -1,
            'verbosity':    -1,
            'random_state': 42,
            'bagging_freq': 1,
            **tuned,
        }
    return dict(
        objective='quantile', alpha=alpha, n_estimators=10000,
        learning_rate=0.01, num_leaves=63, max_depth=7, min_child_samples=40, verbosity=-1,
    )


# ─── Training ─────────────────────────────────────────────────────────────────

def train_horizon(df: pd.DataFrame, horizon_h: int, val_cutoff: datetime) -> dict:
    """
    Train point + quantile models for one horizon using V6 physics features.
    The 'regime' column is injected as a pd.Categorical so LightGBM treats
    it as a native categorical feature (optimal splits, no one-hot encoding).
    """
    print(f"\n{'='*60}")
    print(f"  Horizon: {horizon_h}h  (V6 Physics-Informed)")
    print(f"{'='*60}")

    # 1. Build V6 feature matrix (95 features)
    X, y = engineer_features(df, horizon_h)
    mask = y.notna()
    X, y = X[mask], y[mask]

    # 2. Inject regime as a categorical feature
    regime = classify_regime(df)
    X['regime'] = pd.Categorical(regime.reindex(X.index).fillna(2).astype(int))

    # Print regime distribution in this dataset
    regime_counts = X['regime'].value_counts().sort_index()
    print(f"  Features: {len(X.columns)} cols  (95 continuous + 1 categorical)")
    for r_val, r_count in regime_counts.items():
        pct = r_count / len(X) * 100
        print(f"    Regime {r_val} ({REGIME_LABELS.get(r_val, '?'):<25s}): {r_count:>6,} ({pct:.1f}%)")

    # 3. Temporal split — strictly chronological
    train_mask = X.index < val_cutoff
    val_mask   = X.index >= val_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"  Train rows: {len(X_train):,}  |  Val rows: {len(X_val):,}")

    if len(X_train) < 500:
        raise RuntimeError(f"Too few training rows ({len(X_train)}). Check data fetch.")

    # Tell LightGBM which columns are categorical
    cat_features = ['regime']

    # 4. Temporal Sample Weighting
    # 2021-2022 data is valuable for extreme wildfire memory, but less representative 
    # of current atmospheric dynamics. De-weight it (0.5) to prioritize recent patterns.
    weights = np.ones(len(y_train))
    weights[X_train.index.year <= 2022] = 0.5

    # ── Leakage-free early stopping ──────────────────────────────────────
    # Using a 10% stratified random sample (by month) so the evaluation set
    # sees all seasons, not just the chronologically last 30 days (winter).
    from sklearn.model_selection import train_test_split
    es_within = np.zeros(len(X_train), dtype=bool)
    _, es_idx = train_test_split(
        np.arange(len(X_train)), 
        test_size=0.10, 
        random_state=42, 
        stratify=X_train.index.month
    )
    es_within[es_idx] = True

    X_fit_df = X_train[~es_within]
    y_fit    = y_train[~es_within]
    w_fit    = weights[~es_within]
    
    X_es_df  = X_train[es_within]
    y_es     = y_train[es_within]
    w_es     = weights[es_within]

    print(f"  Fit rows: {len(X_fit_df):,}  |  ES eval rows: {len(X_es_df):,}")

    # 5a. Point model with early stopping
    print("  Training point model...")
    point_model = lgb.LGBMRegressor(**_point_params(horizon_h))
    point_model.fit(
        X_fit_df, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_es_df, y_es)],
        eval_sample_weight=[w_es],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100, verbose=True)],
        categorical_feature=cat_features,
    )
    point_path = MODELS_DIR / f"lgbm_point_{horizon_h}h.pkl"
    joblib.dump(point_model, point_path)
    print(f"  Point model saved → {point_path}  (best iter: {point_model.best_iteration_})")

    # 5b. Lower quantile (with early stopping)
    print("  Training lower quantile model (q005)...")
    q05_model = lgb.LGBMRegressor(**_quantile_params(0.005, horizon_h))
    q05_model.fit(
        X_fit_df, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_es_df, y_es)],
        eval_sample_weight=[w_es],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100, verbose=False)],
        categorical_feature=cat_features,
    )
    q05_path = MODELS_DIR / f"lgbm_q05_{horizon_h}h.pkl"
    joblib.dump(q05_model, q05_path)
    print(f"  Q005 model saved → {q05_path}")

    # 5c. Upper quantile (with early stopping)
    print("  Training upper quantile model (q995)...")
    q95_model = lgb.LGBMRegressor(**_quantile_params(0.995, horizon_h))
    q95_model.fit(
        X_fit_df, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_es_df, y_es)],
        eval_sample_weight=[w_es],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100, verbose=False)],
        categorical_feature=cat_features,
    )
    q95_path = MODELS_DIR / f"lgbm_q95_{horizon_h}h.pkl"
    joblib.dump(q95_model, q95_path)
    print(f"  Q995 model saved → {q95_path}")

    # 6. Validation metrics
    if len(X_val) > 0:
        val_point_res = point_model.predict(X_val)
        val_q05_res   = q05_model.predict(X_val)
        val_q95_res   = q95_model.predict(X_val)

        base_aqi  = X_val['aqi_current'].values
        val_point = val_point_res + base_aqi
        val_q05   = val_q05_res + base_aqi
        val_q95   = val_q95_res + base_aqi
        y_val_abs = y_val.values + base_aqi

        val_point = np.clip(val_point, 0, 500)
        val_q05   = np.clip(val_q05,   0, 500)
        val_q95   = np.clip(val_q95,   0, 500)

        mae      = mean_absolute_error(y_val_abs, val_point)
        r2       = r2_score(y_val_abs, val_point)
        covered  = np.mean((y_val_abs >= val_q05) & (y_val_abs <= val_q95))
        avg_width = np.mean(val_q95 - val_q05)

        print(f"\n  Val MAE:       {mae:.2f} AQI")
        print(f"  Val R²:        {r2:.3f}")
        print(f"  Val Coverage:  {covered*100:.1f}%  (target ≥ 90%)")
        print(f"  Avg CI Width:  {avg_width:.1f} AQI")

        return {
            "horizon_h":    horizon_h,
            "val_mae":      round(mae, 2),
            "val_r2":       round(r2, 3),
            "val_coverage": round(covered * 100, 1),
            "avg_width":    round(avg_width, 1),
        }
    else:
        print("  [WARN] No validation data available.")
        return {"horizon_h": horizon_h, "val_mae": None, "val_r2": None,
                "val_coverage": None, "avg_width": None}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Folsom AQI Forecast — V6 Physics-Informed Training")
    print("=" * 60)

    # Step 1: Load historical data (reuse cached data if possible)
    hist_path = DATA_DIR / "historical.parquet"
    if hist_path.exists():
        print(f"\nLoading cached historical data from {hist_path}...")
        df = pd.read_parquet(hist_path)
    else:
        print("\nStep 1: Fetching historical data (2020-01-01 → today)...")
        df = fetch_full_history()
        df.to_parquet(hist_path)

    print(f"  {len(df):,} rows  |  {df.index.min()} → {df.index.max()}")

    # Validation cutoff: 60 days before today
    val_cutoff = datetime.now(tz=df.index.tz) - timedelta(days=60)
    print(f"  Train/Val cutoff: {val_cutoff.strftime('%Y-%m-%d')}")

    # Step 2: Train V6 models for each horizon
    print("\nStep 2: Training V6 physics-informed models...")
    results = []
    for h in HORIZONS:
        metrics = train_horizon(df, h, val_cutoff)
        results.append(metrics)

    # Step 3: Print summary table
    print("\n" + "=" * 70)
    print("  V6 TRAINING SUMMARY (Physics-Informed + Regime Conditioning)")
    print("=" * 70)
    print(f"  {'Horizon':<10} {'Val MAE':<12} {'Val R²':<10} {'Coverage':<12} {'Avg Width'}")
    print(f"  {'-'*60}")
    for r in results:
        mae = f"{r['val_mae']:.1f}" if r['val_mae'] is not None else "N/A"
        r2  = f"{r['val_r2']:.3f}" if r['val_r2'] is not None else "N/A"
        cov = f"{r['val_coverage']:.1f}%" if r['val_coverage'] is not None else "N/A"
        wid = f"{r['avg_width']:.1f}" if r['avg_width'] is not None else "N/A"
        print(f"  {str(r['horizon_h'])+'h':<10} {mae:<12} {r2:<10} {cov:<12} {wid}")

    # Save feature names
    feature_names = get_feature_names(6)
    feature_names.append('regime')  # Add the categorical column
    fn_path = MODELS_DIR / "feature_names_v6.json"
    with open(fn_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"\n  Feature names saved → {fn_path}  ({len(feature_names)} features)")

    # Save training metrics
    metrics_path = MODELS_DIR / "training_metrics_v6.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "architecture": "V6 Physics-Informed + Regime Soft-Routing",
            "total_features": len(feature_names),
            "categorical_features": ["regime"],
            "horizons": results,
        }, f, indent=2)
    print(f"  Training metrics saved → {metrics_path}")

    print("\n✓ V6 Training complete. Models saved to models_v6/")
    print("  Compare these metrics against V4 (models/) to evaluate improvement.")


if __name__ == "__main__":
    # Suppress sklearn/pandas noise for a clean production output
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    
    main()
