import joblib
import pandas as pd

model = joblib.load('models/lgbm_q05_6h.pkl')
if hasattr(model, 'feature_name_'):
    names = model.feature_name_
    print(f"MODEL EXPECTS: {len(names)} features")
    print(names)
else:
    # If using lgb.Booster directly
    names = model.feature_name()
    print(f"BOOSTER EXPECTS: {len(names)} features")
    print(names)
