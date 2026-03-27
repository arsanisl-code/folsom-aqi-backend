"""
train.py — Train 12 LightGBM models (point + quantile bounds) for 4 horizons.
Run once locally on Windows; copies models/ to DigitalOcean via deploy.sh.

Expected runtime: 10–20 minutes on a modern laptop.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from data_fetcher import fetch_full_history
from features import engineer_features, get_feature_names

# ─── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
DATA_DIR   = Path("data")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [6, 12, 24, 48]

# ─── Optuna parameter ingestion ───────────────────────────────────────────────

OPTUNA_PARAMS_PATH = MODELS_DIR / "best_optuna_params.json"


def _load_optuna_params() -> dict | None:
    """
    Load Optuna best hyperparameters from JSON if available.
    Returns the parsed dict, or None if the file doesn't exist or is invalid.
    """
    if not OPTUNA_PARAMS_PATH.exists():
        return None
    try:
        with open(OPTUNA_PARAMS_PATH) as f:
            data = json.load(f)
        print(f"[train] ✓ Loaded Optuna-tuned params from {OPTUNA_PARAMS_PATH}")
        return data
    except Exception as exc:
        print(f"[train] WARNING: Could not load {OPTUNA_PARAMS_PATH}: {exc}",
              file=sys.stderr)
        return None


def _get_optuna_best(horizon_h: int, model_type: str) -> dict | None:
    """
    Extract the best_params dict for a specific horizon + model type.
    Returns None if Optuna params are unavailable for this combination.

    model_type: "point", "q01", or "q99"
    """
    if _OPTUNA_PARAMS is None:
        return None
    horizon_key = f"{horizon_h}h"
    entry = _OPTUNA_PARAMS.get(horizon_key, {}).get(model_type, {})
    best = entry.get("best_params")
    if best:
        print(f"[train] Using Optuna-tuned params for {model_type} {horizon_key}")
    return best


# Load once at module level — None if file doesn't exist yet (first run)
_OPTUNA_PARAMS = _load_optuna_params()

# ─── LightGBM hyperparameters ─────────────────────────────────────────────────

def _point_params(horizon_h: int) -> dict:
    """
    Return point model (Huber) hyperparameters for the given horizon.
    If Optuna-tuned params exist, use them; otherwise fall back to defaults.
    Fixed params (objective, alpha, n_estimators, n_jobs) are never overridden.
    """
    # ── Check for Optuna-tuned params first ──
    tuned = _get_optuna_best(horizon_h, "point")
    if tuned is not None:
        return {
            # Fixed params — never tuned
            'objective':    'huber',
            'alpha':        2.0,          # Huber delta, NOT a quantile
            'n_estimators': 4000,
            'n_jobs':       -1,
            'verbosity':    -1,
            'random_state': 42,
            'bagging_freq': 1,            # required when bagging_fraction < 1.0
            # Tuned params from Optuna (num_leaves, max_depth, learning_rate,
            # min_data_in_leaf, feature_fraction, bagging_fraction, reg_alpha,
            # reg_lambda)
            **tuned,
        }

    # ── Fallback: hardcoded defaults (original train.py logic) ──
    base = dict(
        objective='huber',
        alpha=2.0,
        n_estimators=4000,
        learning_rate=0.01,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if horizon_h <= 12:
        base.update(
            num_leaves=63, max_depth=8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0,
        )
    elif horizon_h <= 24:
        base.update(
            num_leaves=63, max_depth=7, min_child_samples=40,
            reg_alpha=1.0, reg_lambda=2.0,
        )
    else:  # 48h — constrained: low signal-to-noise, simpler trees generalize better
        base.update(
            num_leaves=31, max_depth=5, min_child_samples=60,
            reg_alpha=2.0, reg_lambda=5.0,
        )
    return base


def _quantile_params(alpha: float, horizon_h: int) -> dict:
    """
    Return quantile model hyperparameters for the given alpha and horizon.
    If Optuna-tuned params exist, use them; otherwise fall back to defaults.
    Fixed params (objective, alpha, n_estimators, n_jobs) are never overridden.
    """
    # Map alpha to model_type key used in best_optuna_params.json
    model_type = "q01" if alpha < 0.5 else "q99"

    # ── Check for Optuna-tuned params first ──
    tuned = _get_optuna_best(horizon_h, model_type)
    if tuned is not None:
        return {
            # Fixed params — never tuned
            'objective':    'quantile',
            'alpha':        alpha,        # 0.01 or 0.99, passed by caller
            'n_estimators': 1500,
            'n_jobs':       -1,
            'verbosity':    -1,
            'random_state': 42,
            'bagging_freq': 1,
            # Tuned params from Optuna
            **tuned,
        }

    # ── Fallback: hardcoded defaults (original train.py logic) ──
    base = dict(
        objective='quantile',
        alpha=alpha,
        n_estimators=1500,
        learning_rate=0.01,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if horizon_h <= 12:
        base.update(num_leaves=31, max_depth=6)
    elif horizon_h <= 24:
        base.update(num_leaves=15, max_depth=4)
    else:  # 48h — maximally constrained for low-signal horizon
        base.update(num_leaves=15, max_depth=3)
    return base


# ─── Training ─────────────────────────────────────────────────────────────────

def train_horizon(df: pd.DataFrame, horizon_h: int, val_cutoff: datetime) -> dict:
    """
    Train point + quantile models for one horizon.
    Returns dict with val MAE, coverage, and interval width.
    """
    print(f"\n{'='*60}")
    print(f"  Horizon: {horizon_h}h")
    print(f"{'='*60}")

    # 1. Build features
    X, y = engineer_features(df, horizon_h)
    mask = y.notna()
    X, y = X[mask], y[mask]

    # 2. Temporal split — strictly chronological, no shuffle
    train_mask = X.index < val_cutoff
    val_mask   = X.index >= val_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"  Train rows: {len(X_train):,}  |  Val rows: {len(X_val):,}")

    if len(X_train) < 500:
        raise RuntimeError(f"Too few training rows ({len(X_train)}). Check data fetch.")

    # 3. Impute — fit on train only, apply to both
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    imputer_path = MODELS_DIR / f"imputer_{horizon_h}h.pkl"
    joblib.dump(imputer, imputer_path)
    print(f"  Imputer saved → {imputer_path}")

    # Wrap back as DataFrames WITH DatetimeIndex (needed for early stopping split)
    X_train_df = pd.DataFrame(X_train_imp, columns=X_train.columns, index=X_train.index)
    X_val_df   = pd.DataFrame(X_val_imp,   columns=X_val.columns, index=X_val.index)

    # ── Leakage-free early stopping ──────────────────────────────────────
    # CRITICAL: We must NOT use the final 60-day val set (which contains
    # validate.py's 30-day holdout) as eval_set. Instead, carve out the
    # last ES_DAYS of the TRAINING window as the early stopping eval set.
    # Timeline:
    #   [training fit data] [ES eval: 30d] [val_cutoff] [val set: 60d (includes validate.py 30d)]
    # The model trains on data before (val_cutoff - 30d).
    # Early stopping monitors on (val_cutoff - 30d) to val_cutoff.
    # validate.py's 30-day window is completely unseen.
    ES_DAYS = 30
    es_cutoff = val_cutoff - timedelta(days=ES_DAYS)
    es_within = X_train_df.index >= es_cutoff

    X_fit_df = X_train_df[~es_within]
    y_fit    = y_train[~es_within]
    X_es_df  = X_train_df[es_within]
    y_es     = y_train[es_within]

    print(f"  Fit rows: {len(X_fit_df):,}  |  ES eval rows: {len(X_es_df):,}")

    # 4a. Point forecast model with early stopping
    print("  Training point model...")
    point_model = lgb.LGBMRegressor(**_point_params(horizon_h))
    point_model.fit(
        X_fit_df, y_fit,
        eval_set=[(X_es_df, y_es)],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(50, verbose=True)],
    )
    point_path = MODELS_DIR / f"lgbm_point_{horizon_h}h.pkl"
    joblib.dump(point_model, point_path)
    print(f"  Point model saved → {point_path}  (best iter: {point_model.best_iteration_})")

    # 4b. Lower quantile (0.5th percentile)
    # NOTE: No early stopping for quantile models. At 1500 trees with lr=0.01,
    # overfitting risk is minimal and quantile loss on the small ES eval set
    # is too noisy, causing catastrophic early stops (e.g., iter=1).
    print("  Training lower quantile model (q005)...")
    q05_model = lgb.LGBMRegressor(**_quantile_params(0.005, horizon_h))
    q05_model.fit(X_train_df, y_train, callbacks=[lgb.log_evaluation(200)])
    q05_path = MODELS_DIR / f"lgbm_q05_{horizon_h}h.pkl"
    joblib.dump(q05_model, q05_path)
    print(f"  Q005 model saved → {q05_path}")

    # 4c. Upper quantile (99.5th percentile)
    print("  Training upper quantile model (q995)...")
    q95_model = lgb.LGBMRegressor(**_quantile_params(0.995, horizon_h))
    q95_model.fit(X_train_df, y_train, callbacks=[lgb.log_evaluation(200)])
    q95_path = MODELS_DIR / f"lgbm_q95_{horizon_h}h.pkl"
    joblib.dump(q95_model, q95_path)
    print(f"  Q995 model saved → {q95_path}")

    # 5. Validation metrics
    if len(X_val_df) > 0:
        val_point_res = point_model.predict(X_val_df)
        val_q05_res   = q05_model.predict(X_val_df)
        val_q95_res   = q95_model.predict(X_val_df)

        # Invert the target transformation: Predicted AQI = Residual + Current AQI
        base_aqi  = X_val['aqi_current'].values
        val_point = val_point_res + base_aqi
        val_q05   = val_q05_res + base_aqi
        val_q95   = val_q95_res + base_aqi

        # True y values must also be inverted back to absolute AQI rather than the residual delta
        y_val_abs = y_val.values + base_aqi

        # Clip
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
        return {"horizon_h": horizon_h, "val_mae": None, "val_r2": None, "val_coverage": None, "avg_width": None}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Folsom AQI Forecast — Training Pipeline")
    print("=" * 60)

    # Step 1: Fetch historical data
    print("\nStep 1: Fetching historical data (2022-01-01 → today)...")
    df = fetch_full_history()
    hist_path = DATA_DIR / "historical.parquet"
    df.to_parquet(hist_path)
    print(f"  Saved merged history: {len(df):,} rows → {hist_path}")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")

    # Check for required columns
    required = ['us_aqi', 'pm2_5', 'boundary_layer_height', 'wind_speed_10m',
                'surface_pressure', 'relative_humidity_2m', 'temperature_2m',
                'precipitation', 'cloud_cover', 'wind_direction_10m',
                'direct_radiation', 'soil_temperature_0_to_7cm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n[ERROR] Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Validation cutoff: 60 days before today
    val_cutoff = datetime.now(tz=df.index.tz) - timedelta(days=60)
    print(f"\n  Train/Val cutoff: {val_cutoff.strftime('%Y-%m-%d')}")

    # Step 2: Train models for each horizon
    print("\nStep 2: Training models for each horizon...")
    results = []
    for h in HORIZONS:
        metrics = train_horizon(df, h, val_cutoff)
        results.append(metrics)

    # Step 3: Print summary table
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Horizon':<10} {'Val MAE':<12} {'Val R²':<10} {'Coverage':<12} {'Avg Width'}")
    print(f"  {'-'*60}")
    for r in results:
        mae = f"{r['val_mae']:.1f}" if r['val_mae'] is not None else "N/A"
        r2  = f"{r['val_r2']:.3f}" if r['val_r2'] is not None else "N/A"
        cov = f"{r['val_coverage']:.1f}%" if r['val_coverage'] is not None else "N/A"
        wid = f"{r['avg_width']:.1f}" if r['avg_width'] is not None else "N/A"
        print(f"  {str(r['horizon_h'])+'h':<10} {mae:<12} {r2:<10} {cov:<12} {wid}")

    # Step 4: Save feature names for inference alignment check
    feature_names = get_feature_names(6)
    fn_path = MODELS_DIR / "feature_names.json"
    with open(fn_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"\n  Feature names saved → {fn_path}  ({len(feature_names)} features)")

    # Save training metrics
    metrics_path = MODELS_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "horizons":   results,
        }, f, indent=2)
    print(f"  Training metrics saved → {metrics_path}")

    print("\n✓ Training complete. Check 6h MAE — target is ≤ 8.0 AQI.")
    print("  Next: run python validate.py")


if __name__ == "__main__":
    main()
