import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from features_v6 import engineer_features

# 1. Feature Importance Extraction
model_path = Path("models_v6/lgbm_point_48h.pkl")
hist_path = Path("data/historical.parquet")

def run_diagnostics():
    print("============================================================")
    print("  V6 VETO DIAGNOSTICS: Feature Importance & Variance Check")
    print("============================================================\n")

    # Load Model
    model = joblib.load(model_path)
    
    # Get feature names from features_v6
    # LightGBM stores internal names, but we can extract importance arrays
    try:
        feature_names = model.feature_name_
    except AttributeError:
        # Fallback if feature names weren't saved
        feature_names = [f"Feature_{i}" for i in range(model.feature_importances_.shape[0])]

    # Extract Gain (How much each feature contributed to lowering loss)
    importance_gain = model.booster_.feature_importance(importance_type='gain')
    importance_split = model.booster_.feature_importance(importance_type='split')

    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Gain': importance_gain,
        'Splits': importance_split
    }).sort_values(by='Gain', ascending=False)

    print("--- TOP 25 FEATURES BY GAIN (48h MODEL) ---")
    print(df_importance.head(25).to_string(index=False))

    # Check specifically for the fire features
    print("\n--- GROUP 8 WILDFIRE FEATURES RANKING ---")
    fire_features = df_importance[df_importance['Feature'].str.contains('fire', case=False, na=False)]
    if fire_features.empty:
        print("WARNING: No 'fire' features found in the model!")
    else:
        print(fire_features.to_string(index=False))


    # 2. Variance Check for September Holdout
    print("\n============================================================")
    print("  SEPTEMBER VARIANCE CHECK")
    print("============================================================")
    
    df = pd.read_parquet(hist_path)
    # The holdout backtest evaluated data from Jan 2024/2025 onwards.
    # Let's specifically look at September. We will get all Septembers in the dataset and specifically the last one handled by backtest (2024 or 2025).
    
    # Let's isolate the last September in the dataset
    end_year = df.index.max().year
    if df.index.max().month < 9:
        sep_year = end_year - 1
    else:
        sep_year = end_year

    df_sep = df[(df.index.year == sep_year) & (df.index.month == 9)]
    
    if df_sep.empty:
        print(f"No data found for September {sep_year}.")
    else:
        aqi = df_sep['us_aqi'].dropna()
        mean_aqi = aqi.mean()
        std_aqi = aqi.std()
        max_aqi = aqi.max()
        min_aqi = aqi.min()
        
        print(f"Ground Truth AQI Stats for September {sep_year}:")
        print(f"  Count: {len(aqi)} hours")
        print(f"  Mean AQI: {mean_aqi:.2f}")
        print(f"  Std Dev : {std_aqi:.2f}  <-- If this is low, R² will naturally be near zero.")
        print(f"  Range   : {min_aqi:.0f} to {max_aqi:.0f} AQI")
        
        # How many days exceeded Moderate (AQI > 50)?
        days_over_50 = (aqi > 50).sum()
        days_over_100 = (aqi > 100).sum()
        print(f"  Hours > 50 (Moderate) : {days_over_50}")
        print(f"  Hours > 100 (Unhealthy): {days_over_100}")

if __name__ == "__main__":
    run_diagnostics()
