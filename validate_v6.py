"""
validate_v6.py — Walk-forward validation for V6 Physics-Informed models.
Mirrors validate.py but uses features_v6.
Points to models_v6/ directory.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:
    InconsistentVersionWarning = UserWarning

from features_v6 import engineer_features, classify_regime

MODELS_DIR = Path("models_v6")
DATA_DIR   = Path("data")
HORIZONS   = [6, 12, 24, 48]
N_FOLDS    = 30


def load_models(horizon_h: int):
    point   = joblib.load(MODELS_DIR / f"lgbm_point_{horizon_h}h.pkl")
    q05     = joblib.load(MODELS_DIR / f"lgbm_q05_{horizon_h}h.pkl")
    q95     = joblib.load(MODELS_DIR / f"lgbm_q95_{horizon_h}h.pkl")
    imputer = joblib.load(MODELS_DIR / f"imputer_{horizon_h}h.pkl")
    return point, q05, q95, imputer


def walk_forward_validate(df: pd.DataFrame, horizon_h: int) -> dict:
    print(f"\n  Validating {horizon_h}h horizon (V6)...")

    point_model, q05_model, q95_model, _ = load_models(horizon_h)

    errors, pers_errors, covered, worst_days = [], [], [], []
    all_y_true, all_y_pred, all_rows = [], [], []

    now = df.index.max()
    window_start = now - timedelta(days=N_FOLDS)

    # Build feature matrix once
    X_all, y_all = engineer_features(df, horizon_h)
    valid_mask = y_all.notna()
    X_all, y_all = X_all[valid_mask], y_all[valid_mask]

    # Inject regime as categorical
    regime = classify_regime(df)
    X_all['regime'] = pd.Categorical(regime.reindex(X_all.index).fillna(2).astype(int))

    for fold_idx in range(N_FOLDS):
        fold_date = (window_start + timedelta(days=fold_idx)).normalize()
        fold_end  = fold_date + timedelta(days=1)

        train_mask = X_all.index < fold_date
        test_mask  = (X_all.index >= fold_date) & (X_all.index < fold_end)

        if train_mask.sum() < 200 or test_mask.sum() == 0:
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        # Impute continuous columns only; preserve regime categorical
        from sklearn.impute import SimpleImputer
        regime_train = X_train['regime']
        regime_test  = X_test['regime']
        X_train_cont = X_train.drop(columns=['regime'])
        X_test_cont  = X_test.drop(columns=['regime'])

        imp = SimpleImputer(strategy='median')
        X_train_imp = pd.DataFrame(
            imp.fit_transform(X_train_cont),
            columns=X_train_cont.columns,
        )
        X_test_imp = pd.DataFrame(
            imp.transform(X_test_cont),
            columns=X_test_cont.columns,
        )

        # Reattach regime
        X_train_imp['regime'] = regime_train.values
        X_train_imp['regime'] = pd.Categorical(X_train_imp['regime'])
        X_test_imp['regime'] = regime_test.values
        X_test_imp['regime'] = pd.Categorical(X_test_imp['regime'])

        # Predict
        pred_point_res = point_model.predict(X_test_imp)
        pred_q05_res   = q05_model.predict(X_test_imp)
        pred_q95_res   = q95_model.predict(X_test_imp)

        # Invert residuals to absolute AQI
        base_aqi   = X_test['aqi_current'].values
        pred_point = pred_point_res + base_aqi
        pred_q05   = pred_q05_res + base_aqi
        pred_q95   = pred_q95_res + base_aqi
        y_test_abs = y_test.values + base_aqi

        # Quantize
        pred_point = np.round(np.clip(pred_point, 0, 500)).astype(int)
        pred_q05   = np.round(np.clip(pred_q05,   0, 500)).astype(int)
        pred_q95   = np.round(np.clip(pred_q95,   0, 500)).astype(int)
        y_test_abs = np.round(np.clip(y_test_abs, 0, 500)).astype(int)

        for ts, tru, prd in zip(y_test.index, y_test_abs, pred_point):
            all_rows.append({
                "timestamp": ts,
                "actual_aqi": int(tru),
                "predicted_aqi": int(prd),
                "horizon_h": horizon_h,
            })

        all_y_true.extend(y_test_abs)
        all_y_pred.extend(pred_point)

        fold_mae = mean_absolute_error(y_test_abs, pred_point)
        fold_pers_mae = mean_absolute_error(y_test_abs, base_aqi)
        fold_cov = np.mean((y_test_abs >= pred_q05) & (y_test_abs <= pred_q95))

        errors.append(fold_mae)
        pers_errors.append(fold_pers_mae)
        covered.append(fold_cov)
        worst_days.append({
            "date": fold_date.strftime("%Y-%m-%d"),
            "mae":  round(fold_mae, 2),
        })

    worst_days.sort(key=lambda x: x["mae"], reverse=True)

    global_r2 = r2_score(all_y_true, all_y_pred) if all_y_true else None

    mean_mae = float(np.mean(errors)) if errors else None
    mean_pers_mae = float(np.mean(pers_errors)) if pers_errors else None
    skill_score = (1 - (mean_mae / mean_pers_mae)) if (mean_mae and mean_pers_mae and mean_pers_mae > 0) else None

    return {
        "horizon_h":     horizon_h,
        "n_folds":       len(errors),
        "mean_mae":      round(mean_mae, 2) if mean_mae else None,
        "baseline_mae":  round(mean_pers_mae, 2) if mean_pers_mae else None,
        "skill_score":   round(skill_score, 3) if skill_score else None,
        "median_mae":    round(float(np.median(errors)), 2) if errors else None,
        "r2_score":      round(float(global_r2), 3) if global_r2 is not None else None,
        "mean_coverage": round(float(np.mean(covered)) * 100, 1) if covered else None,
        "worst_5_days":  worst_days[:5],
        "hourly_data":   all_rows,
    }


def main():
    print("=" * 60)
    print("  Folsom AQI — V6 Walk-Forward Validation (last 30 days)")
    print("=" * 60)

    hist_path = DATA_DIR / "historical.parquet"
    if not hist_path.exists():
        print("[ERROR] data/historical.parquet not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading historical data from {hist_path}...")
    df = pd.read_parquet(hist_path)
    print(f"  {len(df):,} rows  |  {df.index.min()} → {df.index.max()}")

    # Check V6 models exist
    missing = []
    for h in HORIZONS:
        for suffix in ["point", "q05", "q95"]:
            p = MODELS_DIR / f"lgbm_{suffix}_{h}h.pkl"
            if not p.exists():
                missing.append(str(p))
    if missing:
        print(f"\n[ERROR] Missing V6 model files: {missing}", file=sys.stderr)
        sys.exit(1)

    results = []
    for h in HORIZONS:
        r = walk_forward_validate(df, h)
        results.append(r)

    # Print summary
    print("\n" + "=" * 80)
    print("  V6 WALK-FORWARD VALIDATION RESULTS (30-day window)")
    print("=" * 80)
    print(f"  {'Horizon':<10} {'Model MAE':<12} {'Base MAE':<12} {'Skill Score':<12} {'R² Score':<10} {'Coverage'}")
    print(f"  {'-'*80}")
    for r in results:
        mae   = f"{r['mean_mae']:.2f}" if r['mean_mae'] is not None else "N/A"
        base  = f"{r['baseline_mae']:.2f}" if r.get('baseline_mae') is not None else "N/A"
        skill = f"{r['skill_score']:.3f}" if r.get('skill_score') is not None else "N/A"
        r2    = f"{r['r2_score']:.3f}" if r['r2_score'] is not None else "N/A"
        cov   = f"{r['mean_coverage']:.1f}%" if r['mean_coverage'] is not None else "N/A"
        status = "✓" if r['mean_mae'] is not None and r['mean_mae'] <= 8 else "⚠"
        print(f"  {status} {str(r['horizon_h'])+'h':<9} {mae:<12} {base:<12} {skill:<12} {r2:<10} {cov}")

    # Worst days
    print("\n  Worst 5 days per horizon (highest MAE):")
    for r in results:
        print(f"\n  {r['horizon_h']}h horizon:")
        for day in r['worst_5_days']:
            print(f"    {day['date']}  MAE={day['mae']:.2f}")

    # Save results
    with open(MODELS_DIR / "validation_results_v6.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "hourly_data"} for r in results], f, indent=2)

    print(f"\n  Results saved → {MODELS_DIR / 'validation_results_v6.json'}")
    print("\n✓ V6 Validation complete.")


if __name__ == "__main__":
    # Suppress sklearn/pandas noise for a clean production output
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    main()
