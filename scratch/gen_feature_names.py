import json
from pathlib import Path

import pandas as pd

from features import engineer_features

# Load a small slice of data to get feature names
df = pd.read_parquet("data/cache/recent_combined_ph168_fd5.parquet").head(100)

MODELS_DIR = Path("models")

for h in [6, 12, 24, 48]:
    print(f"Generating feature names for {h}h...")
    X, _ = engineer_features(df, horizon_h=h)
    feature_names = X.columns.tolist()
    (MODELS_DIR / f"feature_names_{h}h.json").write_text(json.dumps(feature_names, indent=2))
    print(f"  Saved {len(feature_names)} features.")

print("Done.")
