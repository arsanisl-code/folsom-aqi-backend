"""
train_v5.py — V5 Hybrid Physics-Informed Training Pipeline.
Identical to train.py but uses features_v5 (95 features + regime categorical).

Key differences from train.py:
  1. Imports from features_v5 instead of features.
  2. Calls classify_regime() and injects 'regime' as a pd.Categorical column.
  3. Saves models to models_v5/ to avoid overwriting production models.
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
from features_v5 import engineer_features, get_feature_names, classify_regime, REGIME_LABELS

# ─── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models_v5")
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
        print(f"[train_v5] ✓ Loaded Optuna-tuned params from {OPTUNA_PARAMS_PATH}")
        return data
    except Exception as exc:
        print(f"[train_v5] WARNING: Could not load {OPTUNA_PARAMS_PATH}: {exc}",
              file=sys.stderr)
        return None


def _get_optuna_best(horizon_h: int, model_type: str) -> dict | None:
    if _OPTUNA_PARAMS is None:
        return None
    horizon_key = f"{horizon_h}h"
    entry = _OPTUNA_PARAMS.get(horizon_key, {}).get(model_type, {})
    best = entry.get("best_params")
    if best:
        print(f"[train_v5] Using Optuna-tuned params for {model_type} {horizon_key}")
    return best


_OPTUNA_PARAMS = _load_optuna_params()

# ─── LightGBM hyperparameters ─────────────────────────────────────────────────


def _point_params(horizon_h: int) -> dict:
    tuned = _get_optuna_best(horizon_h, "point")
    if tuned is not None:
        return {
            'objective':    'huber',
            'alpha':        2.0,
            'n_estimators': 4000,
            'n_jobs':       -1,
            'verbosity':    -1,
            'random_state': 42,
            'bagging_freq': 1,
            **tuned,
        }
    base = dict(
        objective='huber', alpha=2.0, n_estimators=4000,
        learning_rate=0.01, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1,
    )
    if horizon_h <= 12:
        base.update(num_leaves=63, max_depth=8, min_child_samples=20,
                    reg_alpha=0.1, reg_lambda=1.0)
    elif horizon_h <= 24:
        base.update(num_leaves=63, max_depth=7, min_child_samples=40,
                    reg_alpha=1.0, reg_lambda=2.0)
    else:
        base.update(num_leaves=31, max_depth=5, min_child_samples=60,
                    reg_alpha=2.0, reg_lambda=5.0)
    return base


def _quantile_params(alpha: float, horizon_h: int) -> dict:
    model_type = "q01" if alpha < 0.5 else "q99"
    tuned = _get_optuna_best(horizon_h, model_type)
    if tuned is not None:
        return {
            'objective':    'quantile',
            'alpha':        alpha,
            'n_estimators': 1500,
            'n_jobs':       -1,
            'verbosity':    -1,
            'random_state': 42,
            'bagging_freq': 1,
            **tuned,
        }
    base = dict(
        objective='quantile', alpha=alpha, n_estimators=1500,
        learning_rate=0.01, random_state=42, n_jobs=-1, verbosity=-1,
    )
    if horizon_h <= 12:
        base.update(num_leaves=31, max_depth=6)
    elif horizon_h <= 24:
        base.update(num_leaves=15, max_depth=4)
    else:
        base.update(num_leaves=15, max_depth=3)
    return base


# ─── Training ─────────────────────────────────────────────────────────────────

def train_horizon(df: pd.DataFrame, horizon_h: int, val_cutoff: datetime) -> dict:
    """
    Train point + quantile models for one horizon using V5 physics features.
    The 'regime' column is injected as a pd.Categorical so LightGBM treats
    it as a native categorical feature (optimal splits, no one-hot encoding).
    """
    print(f"\n{'='*60}")
    print(f"  Horizon: {horizon_h}h  (V5 Physics-Informed)")
    print(f"{'='*60}")

    # 1. Build V5 feature matrix (95 features)
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

    # 4. Impute continuous columns only — preserve the categorical 'regime'
    #    Strategy: separate regime, impute the rest, then reattach.
    regime_train = X_train['regime']
    regime_val   = X_val['regime']
    X_train_cont = X_train.drop(columns=['regime'])
    X_val_cont   = X_val.drop(columns=['regime'])

    imputer = SimpleImputer(strategy='median')
    X_train_imp_np = imputer.fit_transform(X_train_cont)
    
    # Robust column reconstruction — handle case where imputer drops all-NaN cols
    if X_train_imp_np.shape[1] != X_train_cont.shape[1]:
        print(f"  [WARNING] Imputer dropped {X_train_cont.shape[1] - X_train_imp_np.shape[1]} columns. Reconstructing...")
        try:
            # Reconstruct with 0s for dropped columns to maintain 104-feature shape
            kept_cols = X_train_cont.columns[imputer.get_support()]
            X_train_imp = pd.DataFrame(X_train_imp_np, columns=kept_cols, index=X_train.index)
            for col in set(X_train_cont.columns) - set(kept_cols):
                X_train_imp[col] = 0.0
            X_train_imp = X_train_imp[X_train_cont.columns] # Reorder
        except:
            X_train_imp = pd.DataFrame(X_train_imp_np, columns=X_train_cont.columns[:X_train_imp_np.shape[1]], index=X_train.index)
    else:
        X_train_imp = pd.DataFrame(X_train_imp_np, columns=X_train_cont.columns, index=X_train.index)

    X_val_imp_np = imputer.transform(X_val_cont)
    if X_val_imp_np.shape[1] != X_val_cont.shape[1]:
        try:
            kept_cols = X_val_cont.columns[imputer.get_support()]
            X_val_imp = pd.DataFrame(X_val_imp_np, columns=kept_cols, index=X_val.index)
            for col in set(X_val_cont.columns) - set(kept_cols):
                X_val_imp[col] = 0.0
            X_val_imp = X_val_imp[X_val_cont.columns]
        except:
            X_val_imp = pd.DataFrame(X_val_imp_np, columns=X_val_cont.columns[:X_val_imp_np.shape[1]], index=X_val.index)
    else:
        X_val_imp = pd.DataFrame(X_val_imp_np, columns=X_val_cont.columns, index=X_val.index)

    # Reattach regime as categorical
    X_train_imp['regime'] = regime_train.values
    X_train_imp['regime'] = pd.Categorical(X_train_imp['regime'])
    X_val_imp['regime'] = regime_val.values
    X_val_imp['regime'] = pd.Categorical(X_val_imp['regime'])

    imputer_path = MODELS_DIR / f"imputer_{horizon_h}h.pkl"
    joblib.dump(imputer, imputer_path)
    print(f"  Imputer saved → {imputer_path}")

    # Tell LightGBM which columns are categorical
    cat_features = ['regime']

    # ── Leakage-free early stopping ──────────────────────────────────────
    ES_DAYS = 30
    es_cutoff = val_cutoff - timedelta(days=ES_DAYS)
    es_within = X_train_imp.index >= es_cutoff

    X_fit_df = X_train_imp[~es_within]
    y_fit    = y_train[~es_within]
    X_es_df  = X_train_imp[es_within]
    y_es     = y_train[es_within]

    print(f"  Fit rows: {len(X_fit_df):,}  |  ES eval rows: {len(X_es_df):,}")

    # 5a. Point model with early stopping
    print("  Training point model...")
    point_model = lgb.LGBMRegressor(**_point_params(horizon_h))
    point_model.fit(
        X_fit_df, y_fit,
        eval_set=[(X_es_df, y_es)],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(50, verbose=True)],
        categorical_feature=cat_features,
    )
    point_path = MODELS_DIR / f"lgbm_point_{horizon_h}h.pkl"
    joblib.dump(point_model, point_path)
    print(f"  Point model saved → {point_path}  (best iter: {point_model.best_iteration_})")

    # 5b. Lower quantile (no early stopping)
    print("  Training lower quantile model (q005)...")
    q05_model = lgb.LGBMRegressor(**_quantile_params(0.005, horizon_h))
    q05_model.fit(
        X_train_imp, y_train,
        callbacks=[lgb.log_evaluation(200)],
        categorical_feature=cat_features,
    )
    q05_path = MODELS_DIR / f"lgbm_q05_{horizon_h}h.pkl"
    joblib.dump(q05_model, q05_path)
    print(f"  Q005 model saved → {q05_path}")

    # 5c. Upper quantile (no early stopping)
    print("  Training upper quantile model (q995)...")
    q95_model = lgb.LGBMRegressor(**_quantile_params(0.995, horizon_h))
    q95_model.fit(
        X_train_imp, y_train,
        callbacks=[lgb.log_evaluation(200)],
        categorical_feature=cat_features,
    )
    q95_path = MODELS_DIR / f"lgbm_q95_{horizon_h}h.pkl"
    joblib.dump(q95_model, q95_path)
    print(f"  Q995 model saved → {q95_path}")

    # 6. Validation metrics
    if len(X_val_imp) > 0:
        val_point_res = point_model.predict(X_val_imp)
        val_q05_res   = q05_model.predict(X_val_imp)
        val_q95_res   = q95_model.predict(X_val_imp)

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
    print("  Folsom AQI Forecast — V5 Physics-Informed Training")
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

    # Step 2: Train V5 models for each horizon
    print("\nStep 2: Training V5 physics-informed models...")
    results = []
    for h in HORIZONS:
        metrics = train_horizon(df, h, val_cutoff)
        results.append(metrics)

    # Step 3: Print summary table
    print("\n" + "=" * 70)
    print("  V5 TRAINING SUMMARY (Physics-Informed + Regime Conditioning)")
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
    fn_path = MODELS_DIR / "feature_names_v5.json"
    with open(fn_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"\n  Feature names saved → {fn_path}  ({len(feature_names)} features)")

    # Save training metrics
    metrics_path = MODELS_DIR / "training_metrics_v5.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "architecture": "V5 Physics-Informed + Regime Soft-Routing",
            "total_features": len(feature_names),
            "categorical_features": ["regime"],
            "horizons": results,
        }, f, indent=2)
    print(f"  Training metrics saved → {metrics_path}")

    print("\n✓ V5 Training complete. Models saved to models_v5/")
    print("  Compare these metrics against V4 (models/) to evaluate improvement.")


if __name__ == "__main__":
    # Suppress sklearn/pandas noise for a clean production output
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    
    main()
