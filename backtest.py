"""
backtest.py — 2025 Holdout Study (Performance across all 12 months).
Trains on 2020–2024, evaluates on every month of 2025.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from features import engineer_features

# ─── Configuration ────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
HORIZONS = [6, 12, 24, 48]
MODEL_TYPES = ["point", "q05", "q95"]

# Backtest Overrides
TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-12-31"
BACKTEST_YEAR = 2025

BACKTEST_POINT_N_ESTIMATORS = 1000
BACKTEST_QUANTILE_N_ESTIMATORS = 500

# Metadata for labels
ALPHAS = {
    "q01": 0.005,
    "q99": 0.995
}

def load_best_params():
    params_path = MODELS_DIR / "best_optuna_params.json"
    if not params_path.exists():
        print(f"[backtest] ERROR: {params_path} not found. Run tune.py first.")
        sys.exit(1)
    with open(params_path, "r") as f:
        return json.load(f)

def get_params(all_params, horizon_h, model_type):
    h_key = f"{horizon_h}h"
    section = all_params.get(h_key, {}).get(model_type, {})
    best = section.get("best_params", {})
    
    # Base setup
    params = {
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        **best
    }
    
    # Overrides
    if model_type == "point":
        params["objective"] = "huber"
        params["alpha"] = 2.0
        params["n_estimators"] = BACKTEST_POINT_N_ESTIMATORS
    else:
        params["objective"] = "quantile"
        params["alpha"] = ALPHAS[model_type]
        params["n_estimators"] = BACKTEST_QUANTILE_N_ESTIMATORS
        
    return params

# ─── Main Execution ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  Folsom AQI — 2025 Holdout Backtest Study")
    print("=" * 60)

    # 1. Load Data
    hist_path = DATA_DIR / "historical.parquet"
    if not hist_path.exists():
        print(f"[backtest] ERROR: {hist_path} not found.")
        sys.exit(1)
    
    df = pd.read_parquet(hist_path)
    print(f"Loaded {len(df):,} rows from {hist_path}")

    all_params = load_best_params()
    monthly_results = []

    # Process each horizon
    for h in HORIZONS:
        print(f"\n>>> Processing {h}h Horizon...")
        
        # 2. Build Features
        X, y = engineer_features(df, h)
        mask = y.notna()
        X, y = X[mask], y[mask]

        # 3. Filter Training Set (2020–2024)
        train_mask = (X.index >= TRAIN_START) & (X.index <= TRAIN_END)
        X_train, y_train = X[train_mask], y[train_mask]
        
        if len(X_train) == 0:
            print(f"  [ERROR] No training data found for {h}h in range {TRAIN_START} to {TRAIN_END}")
            continue

        # 4. Impute
        imp = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)

        # 5. Train Models
        trained_models = {}
        for m_type in MODEL_TYPES:
            print(f"  Training {m_type} model ({X_train_imp.shape[0]} rows)...")
            params = get_params(all_params, h, m_type)
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_imp, y_train)
            trained_models[m_type] = model

        # 6. Evaluate by Month (2025)
        for month in range(1, 13):
            month_start = pd.Timestamp(year=BACKTEST_YEAR, month=month, day=1, tz=X.index.tz)
            if month == 12:
                month_end = pd.Timestamp(year=BACKTEST_YEAR + 1, month=1, day=1, tz=X.index.tz)
            else:
                month_end = pd.Timestamp(year=BACKTEST_YEAR, month=month + 1, day=1, tz=X.index.tz)
            
            test_mask = (X.index >= month_start) & (X.index < month_end)
            if test_mask.sum() == 0:
                continue
                
            X_test, y_test = X[test_mask], y[test_mask]
            X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)
            
            # Predict
            pred_point = trained_models["point"].predict(X_test_imp)
            pred_q05   = trained_models["q01"].predict(X_test_imp)
            pred_q95   = trained_models["q99"].predict(X_test_imp)
            
            # Invert residuals
            curr_aqi = X_test["aqi_current"].values
            y_abs = y_test.values + curr_aqi
            p_abs = pred_point + curr_aqi
            q05_abs = pred_q05 + curr_aqi
            q95_abs = pred_q95 + curr_aqi
            
            # Metrics
            mae = mean_absolute_error(y_abs, p_abs)
            r2  = r2_score(y_abs, p_abs)
            
            # Baseline (Persistence)
            base_mae = mean_absolute_error(y_abs, curr_aqi)
            skill = 1 - (mae / base_mae) if base_mae > 0 else 0
            
            # Coverage
            cov = np.mean((y_abs >= q05_abs) & (y_abs <= q95_abs)) * 100
            
            monthly_results.append({
                "Horizon": f"{h}h",
                "Year": BACKTEST_YEAR,
                "Month": month,
                "Month_Name": month_start.strftime("%b"),
                "MAE": round(mae, 2),
                "R2": round(r2, 3),
                "Skill": round(skill, 3),
                "Coverage": round(cov, 1)
            })

    # 7. Print Summary & Export
    if not monthly_results:
        print("\n[ERROR] No backtest results generated.")
        return

    results_df = pd.DataFrame(monthly_results)
    
    print("\n" + "=" * 85)
    print(f"  2025 SEASONAL BACKTEST PROFILE (Trained on 2020-2024)")
    print("=" * 85)
    header = f"  {'Horizon':<10} {'Month':<10} {'MAE':<10} {'R2':<10} {'Skill':<10} {'Coverage'}"
    print(header)
    print("-" * 85)
    
    for _, row in results_df.iterrows():
        print(f"  {row['Horizon']:<10} {row['Month_Name']:<10} {row['MAE']:<10.2f} {row['R2']:<10.3f} {row['Skill']:<10.3f} {row['Coverage']}%")
        
    print("-" * 85)
    
    # 8. Annual Averages
    annual = results_df.groupby("Horizon")[["MAE", "R2", "Skill", "Coverage"]].mean().reset_index()
    print(f"  ANNUAL AVERAGES (2025 Full Year)")
    for _, row in annual.iterrows():
        print(f"  ✓ {row['Horizon']:<8} MAE={row['MAE']:<8.2f} R2={row['R2']:<8.3f} Skill={row['Skill']:<8.3f} Cov={row['Coverage']:.1f}%")
        
    # 9. Export
    report_path = "backtest_2025_report.csv"
    results_df.to_csv(report_path, index=False)
    print(f"\n✓ Backtest complete. Report saved → {report_path}")

if __name__ == "__main__":
    main()
