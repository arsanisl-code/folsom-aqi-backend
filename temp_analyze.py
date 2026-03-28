import sys
import traceback
import os
import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import fetch_recent_combined
from features_v6 import engineer_features, classify_regime
from inference import load_all_models

def analyze():
    print("=== Folsom V6 Failure Analysis ===")
    
    # 1. Check Env
    key = os.environ.get("FIRMS_MAP_KEY")
    print(f"FIRMS_MAP_KEY set: {bool(key)}")
    
    # 2. Fetch data (Recent)
    try:
        df = fetch_recent_combined(past_hours=72)
        print(f"Fetched DF columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Data Fetch Failure: {e}")
        return

    # 3. Engineer Features
    try:
        horizon = 48
        X_all, y_all = engineer_features(df, horizon_h=horizon)
        regime = classify_regime(df)
        X_all['regime'] = pd.Categorical(regime.reindex(X_all.index).fillna(2).astype(int))
        print(f"Engineered features: {len(X_all.columns)}")
    except Exception as e:
        print(f"Feature Engineering Failure: {e}")
        traceback.print_exc()
        return

    # 4. Load Models and Compare
    try:
        models = load_all_models()
        m = models[horizon]['point']
        expected = m.feature_name_
        actual = X_all.columns.tolist()
        
        missing = [f for f in expected if f not in actual]
        extra = [f for f in actual if f not in expected]
        
        print(f"Model expects {len(expected)} features.")
        print(f"Actual features {len(actual)}")
        
        if missing:
            print(f"CRITICAL: Missing features: {missing}")
        if extra:
            print(f"Extra features (non-critical): {extra[:5]}...")
            
    except Exception as e:
        print(f"Model Loading/Comparison Failure: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    analyze()
