import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import engineer_features


def main():
    print("Loading data...")
    df = pd.read_parquet("data/historical.parquet")
    val_cutoff = datetime.now(tz=df.index.tz) - timedelta(days=60)

    conformal_margins = {}

    for h in [6, 12, 24, 48]:
        print(f"Processing {h}h...")
        try:
            X_full, y_full = engineer_features(df, h)
            mask = y_full.notna()
            X_full, y_full = X_full[mask], y_full[mask]

            val_mask = X_full.index >= val_cutoff
            X_val, y_val = X_full[val_mask], y_full[val_mask]

            if len(X_val) == 0:
                print(f"No validation data for {h}h.")
                continue

            imputer = joblib.load(f"models/imputer_{h}h.pkl")
            X_val_imp = imputer.transform(X_val)
            X_val_df = pd.DataFrame(X_val_imp, columns=X_val.columns, index=X_val.index)

            physics_cols = json.loads(Path(f"models/physics_cols_{h}h.json").read_text())

            lgbm_full = joblib.load(f"models/lgbm_full_{h}h.pkl")
            xgb = joblib.load(f"models/xgb_{h}h.pkl")
            lgbm_physics = joblib.load(f"models/lgbm_physics_{h}h.pkl")
            meta = joblib.load(f"models/meta_learner_{h}h.pkl")

            base_aqi_val = X_val["aqi_current"].values

            pred_lgbm_full = np.clip(lgbm_full.predict(X_val_df) + base_aqi_val, 0, 500)
            pred_xgb = np.clip(xgb.predict(X_val_df) + base_aqi_val, 0, 500)
            pred_lgbm_physics = np.clip(lgbm_physics.predict(X_val_df[physics_cols]) + base_aqi_val, 0, 500)

            val_oof_stack = np.column_stack([
                pred_lgbm_full - base_aqi_val,
                pred_xgb - base_aqi_val,
                pred_lgbm_physics - base_aqi_val,
            ])

            pred_meta = np.clip(meta.predict_normalized(val_oof_stack) + base_aqi_val, 0, 500)
            y_val_abs = y_val.values + base_aqi_val

            abs_errors = np.abs(y_val_abs - pred_meta)
            margin_90 = np.percentile(abs_errors, 90)

            print(f"  {h}h margin: {margin_90:.2f}")
            conformal_margins[str(h)] = round(float(margin_90), 2)

        except Exception as e:
            print(f"  Error processing {h}h: {e}")

    if conformal_margins:
        Path("models/conformal_margins.json").write_text(json.dumps(conformal_margins, indent=2))
        print("Saved models/conformal_margins.json")

if __name__ == "__main__":
    main()
