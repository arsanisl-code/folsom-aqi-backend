"""
backtest_v5.py — V5 Physics-Informed 2025 Holdout Study.
Trains on 2020–2024 with:
  - 98 continuous features (including 850hPa inversion delta)
  - 1 categorical 'regime' feature (soft routing)
Evaluates on every month of 2025.
"""

import json
import sys
import warnings
from datetime import datetime
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

from features_v5 import engineer_features, classify_regime, REGIME_LABELS
from data_fetcher import fetch_full_history

# ─── Configuration ────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")       # Read Optuna params from V4 production
DATA_DIR   = Path("data")
HORIZONS   = [6, 12, 24, 48]
MODEL_TYPES = ["point", "q01", "q99"]

TRAIN_START    = "2021-01-01"
TRAIN_END      = "2024-12-31"
BACKTEST_YEAR  = 2025

BACKTEST_POINT_N_ESTIMATORS    = 1000
BACKTEST_QUANTILE_N_ESTIMATORS = 500

ALPHAS = {"q01": 0.005, "q99": 0.995}


def load_best_params():
    params_path = MODELS_DIR / "best_optuna_params.json"
    if not params_path.exists():
        print(f"[backtest_v5] ERROR: {params_path} not found.")
        sys.exit(1)
    with open(params_path, "r") as f:
        return json.load(f)


def get_params(all_params, horizon_h, model_type):
    h_key = f"{horizon_h}h"
    section = all_params.get(h_key, {}).get(model_type, {})
    best = section.get("best_params", {})

    params = {"random_state": 42, "n_jobs": -1, "verbosity": -1, **best}

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
    print("  Folsom AQI — V5 Physics-Informed 2025 Holdout Backtest")
    print("=" * 60)

    # 1. Load or fetch data
    hist_path = DATA_DIR / "historical.parquet"
    if hist_path.exists():
        df = pd.read_parquet(hist_path)
        # Check if 850hPa column exists; if not, re-fetch
        if 'temperature_850hPa' not in df.columns:
            print("[backtest_v5] historical.parquet missing temperature_850hPa — re-fetching...")
            df = fetch_full_history()
            df.to_parquet(hist_path)
    else:
        print("[backtest_v5] Fetching historical data...")
        df = fetch_full_history()
        df.to_parquet(hist_path)

    print(f"Loaded {len(df):,} rows from {hist_path}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Has temperature_850hPa: {'temperature_850hPa' in df.columns}")
    if 'temperature_700hPa' not in df.columns:
        print("  [WARNING] 'temperature_700hPa' missing. Skipping Group 4 700hPa Inversion Depth features gracefully.")


    all_params = load_best_params()
    monthly_results = []

    for h in HORIZONS:
        print(f"\n>>> Processing {h}h Horizon...")

        # 2. Build V5 features (98 continuous)
        X, y = engineer_features(df, h)
        mask = y.notna()
        X, y = X[mask], y[mask]

        # 3. Inject regime as categorical
        regime = classify_regime(df)
        X['regime'] = pd.Categorical(regime.reindex(X.index).fillna(2).astype(int))

        # 4. Filter training set (2020–2024)
        train_mask = (X.index >= TRAIN_START) & (X.index <= TRAIN_END)
        X_train, y_train = X[train_mask], y[train_mask]

        if len(X_train) == 0:
            print(f"  [ERROR] No training data for {h}h")
            continue

        # 5. Impute continuous columns, preserve regime
        regime_train = X_train['regime']
        X_train_cont = X_train.drop(columns=['regime'])

        print(f"  Training feature set: {len(X_train_cont.columns)} continuous + 1 categorical")

        imp = SimpleImputer(strategy="median")
        X_train_imp_np = imp.fit_transform(X_train_cont)
        
        # Robust column reconstruction if imputer drops all-NaN columns
        if X_train_imp_np.shape[1] != X_train_cont.shape[1]:
            print(f"  [WARNING] Imputer dropped {X_train_cont.shape[1] - X_train_imp_np.shape[1]} columns. Reconstructing...")
            # Use get_support if available (scikit-learn 1.0+)
            try:
                kept_cols = X_train_cont.columns[imp.get_support()]
                X_train_imp = pd.DataFrame(X_train_imp_np, columns=kept_cols)
                # Re-add dropped columns as 0 to keep constant shape
                for col in set(X_train_cont.columns) - set(kept_cols):
                    X_train_imp[col] = 0.0
                X_train_imp = X_train_imp[X_train_cont.columns] # Reorder to original
            except:
                X_train_imp = pd.DataFrame(X_train_imp_np, columns=X_train_cont.columns[:X_train_imp_np.shape[1]])
        else:
            X_train_imp = pd.DataFrame(X_train_imp_np, columns=X_train_cont.columns)

        X_train_imp['regime'] = regime_train.values
        X_train_imp['regime'] = pd.Categorical(X_train_imp['regime'])

        cat_features = ['regime']

        # 6. Train models
        trained_models = {}
        for m_type in MODEL_TYPES:
            print(f"  Training {m_type} model ({len(X_train_imp)} rows)...")
            params = get_params(all_params, h, m_type)
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_imp, y_train, categorical_feature=cat_features)
            trained_models[m_type] = model

        # 7. Evaluate by month (2025)
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

            # Impute test set (continuous only)
            regime_test = X_test['regime']
            X_test_cont = X_test.drop(columns=['regime'])
            
            # Use same logic for test set imputation to ensure shape consistency
            X_test_imp_np = imp.transform(X_test_cont)
            if X_test_imp_np.shape[1] != X_test_cont.shape[1]:
                # Handled already by re-adding dropped columns to the imputer result
                try:
                    kept_cols = X_test_cont.columns[imp.get_support()]
                    X_test_imp = pd.DataFrame(X_test_imp_np, columns=kept_cols)
                    for col in set(X_test_cont.columns) - set(kept_cols):
                        X_test_imp[col] = 0.0
                    X_test_imp = X_test_imp[X_test_cont.columns]
                except:
                    X_test_imp = pd.DataFrame(X_test_imp_np, columns=X_test_cont.columns[:X_test_imp_np.shape[1]])
            else:
                X_test_imp = pd.DataFrame(X_test_imp_np, columns=X_test_cont.columns)

            X_test_imp['regime'] = regime_test.values
            X_test_imp['regime'] = pd.Categorical(X_test_imp['regime'])

            # Predict
            pred_point = trained_models["point"].predict(X_test_imp)
            pred_q05   = trained_models["q01"].predict(X_test_imp)
            pred_q95   = trained_models["q99"].predict(X_test_imp)

            # Invert residuals
            curr_aqi = X_test["aqi_current"].values
            y_abs    = y_test.values + curr_aqi
            p_abs    = pred_point + curr_aqi
            q05_abs  = pred_q05 + curr_aqi
            q95_abs  = pred_q95 + curr_aqi

            mae   = mean_absolute_error(y_abs, p_abs)
            r2    = r2_score(y_abs, p_abs)
            base  = mean_absolute_error(y_abs, curr_aqi)
            skill = 1 - (mae / base) if base > 0 else 0
            cov   = np.mean((y_abs >= q05_abs) & (y_abs <= q95_abs)) * 100

            monthly_results.append({
                "Horizon": f"{h}h",
                "Month": month,
                "Month_Name": month_start.strftime("%b"),
                "MAE": round(mae, 2),
                "R2": round(r2, 3),
                "Skill": round(skill, 3),
                "Coverage": round(cov, 1),
            })

    # 8. Print Summary
    if not monthly_results:
        print("\n[ERROR] No backtest results generated.")
        return

    results_df = pd.DataFrame(monthly_results)

    print("\n" + "=" * 85)
    print("  V5 PHYSICS-INFORMED 2025 BACKTEST (Trained on 2020-2024)")
    print("=" * 85)
    header = f"  {'Horizon':<10} {'Month':<10} {'MAE':<10} {'R2':<10} {'Skill':<10} {'Coverage'}"
    print(header)
    print("-" * 85)

    for _, row in results_df.iterrows():
        print(f"  {row['Horizon']:<10} {row['Month_Name']:<10} "
              f"{row['MAE']:<10.2f} {row['R2']:<10.3f} "
              f"{row['Skill']:<10.3f} {row['Coverage']}%")

    print("-" * 85)

    annual = results_df.groupby("Horizon")[["MAE", "R2", "Skill", "Coverage"]].mean().reset_index()
    print("  ANNUAL AVERAGES (2025 Full Year)")
    for _, row in annual.iterrows():
        print(f"  ✓ {row['Horizon']:<8} MAE={row['MAE']:<8.2f} "
              f"R2={row['R2']:<8.3f} Skill={row['Skill']:<8.3f} "
              f"Cov={row['Coverage']:.1f}%")

    report_path = "backtest_v5_2025_report.csv"
    results_df.to_csv(report_path, index=False)
    print(f"\n✓ V5 Backtest complete. Report saved → {report_path}")


if __name__ == "__main__":
    # Suppress sklearn/pandas noise for a clean production output
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    main()
