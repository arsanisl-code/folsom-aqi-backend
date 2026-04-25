import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from features import engineer_features
from logger import get_logger
from train import _NNLSMeta

log = get_logger(__name__)

MODELS_DIR = Path("models")
DATA_CACHE = Path("data/cache/recent_combined_ph168_fd5.parquet")
HOLDOUT_YEAR = 2025

# Targets from V12 report
TARGETS = {
    6: {"mae": 2.81, "r2": 0.936},
    12: {"mae": 5.29, "r2": 0.767},
    24: {"mae": 7.12, "r2": 0.629},
    48: {"mae": 8.45, "r2": 0.515}
}

def run_verification():
    print("="*70)
    print("  ENSEMBLE VERIFICATION: NEW MODELS VS V12 BENCHMARKS")
    print("="*70)
    
    # Load data
    from data_fetcher import fetch_full_history
    df = fetch_full_history()
    df_ho = df[df.index.year == HOLDOUT_YEAR]
    
    if len(df_ho) == 0:
        print("Error: No 2025 holdout data found.")
        return

    results = []

    for h in [6, 12, 24, 48]:
        print(f"\nChecking {h}h horizon...")
        
        # Load models
        try:
            m_full = joblib.load(MODELS_DIR / f"lgbm_full_{h}h.pkl")
            m_xgb = joblib.load(MODELS_DIR / f"xgb_{h}h.pkl")
            m_phys = joblib.load(MODELS_DIR / f"lgbm_physics_{h}h.pkl")
            m_meta = joblib.load(MODELS_DIR / f"meta_learner_{h}h.pkl")
            imputer = joblib.load(MODELS_DIR / f"imputer_{h}h.pkl")
            with open(MODELS_DIR / f"physics_cols_{h}h.json", "r") as f:
                phys_cols = json.load(f)
            with open(MODELS_DIR / f"feature_names_{h}h.json", "r") as f:
                feature_names = json.load(f)
        except Exception as e:
            print(f"  Missing or broken models for {h}h: {e}")
            continue

        # Engineer features
        X_ho_raw, y_ho_raw = engineer_features(df, h)
        mask = y_ho_raw.notna() & (X_ho_raw.index.year == HOLDOUT_YEAR)
        X_ho = X_ho_raw[mask]
        y_ho = y_ho_raw[mask]
        
        # Align features
        X_ho = X_ho.reindex(columns=feature_names).fillna(0)
        
        # Impute
        X_ho_imp = imputer.transform(X_ho)
        X_ho_df = pd.DataFrame(X_ho_imp, columns=feature_names, index=X_ho.index)
        
        # Base AQI for residual inversion
        base_aqi = X_ho_df["aqi_current"].values
        
        # Predict
        p_full = m_full.predict(X_ho_df)
        p_xgb = m_xgb.predict(X_ho_df)
        p_phys = m_phys.predict(X_ho_df[phys_cols])
        
        # Meta-blend
        meta_in = np.column_stack([p_full, p_xgb, p_phys])
        p_res = m_meta.predict(meta_in)
        
        # Final absolute AQI
        p_abs = np.clip(p_res + base_aqi, 0, 500)
        t_abs = y_ho.values + base_aqi
        
        # Metrics
        mae = mean_absolute_error(t_abs, p_abs)
        r2 = r2_score(t_abs, p_abs)
        
        target = TARGETS[h]
        mae_diff = mae - target["mae"]
        r2_diff = r2 - target["r2"]
        
        status = "PASS" if mae <= target["mae"] + 0.5 and r2 >= target["r2"] - 0.05 else "REVIEW"
        
        print(f"  RESULT: MAE={mae:.2f} (Target {target['mae']:.2f}, diff={mae_diff:+.2f})")
        print(f"          R2 ={r2:.3f} (Target {target['r2']:.3f}, diff={r2_diff:+.3f})")
        print(f"  STATUS: {status}")
        
        results.append({"h": h, "mae": mae, "r2": r2, "status": status})

    print("\n" + "="*70)
    print("  FINAL VERIFICATION SUMMARY")
    print("="*70)
    if not results:
        print("  NO RESULTS GENERATED. Check model paths.")
        return
        
    all_pass = all(r["status"] == "PASS" for r in results)
    if all_pass:
        print("  ALL HORIZONS PASSED. Ready for deployment.")
    else:
        print("  SOME HORIZONS REQUIRE REVIEW.")

if __name__ == "__main__":
    run_verification()
