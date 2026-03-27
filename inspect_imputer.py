import joblib

imp = joblib.load('models/imputer_6h.pkl')
print(f"IMPUTER EXPECTS: {imp.n_features_in_} features")
