import pandas as pd
import numpy as np
from features import engineer_features
import json

with open('models/feature_names.json', 'r') as f:
    expected = json.load(f)

# Mock a dataframe with all possible columns
idx = pd.date_range('2023-01-01', periods=200, freq='h', tz='America/Los_Angeles')
df = pd.DataFrame(index=idx)
for col in ['us_aqi', 'pm2_5', 'boundary_layer_height', 'wind_speed_10m', 
            'surface_pressure', 'relative_humidity_2m', 'temperature_2m', 
            'precipitation', 'cloud_cover', 'wind_direction_10m', 
            'direct_radiation', 'soil_temperature_0_to_7cm', 'aerosol_optical_depth']:
    df[col] = np.random.rand(200)

X, _ = engineer_features(df, 6)
current = list(X.columns)

print(f"EXPECTED COUNT: {len(expected)}")
print(f"CURRENT COUNT: {len(current)}")

missing = [c for c in expected if c not in current]
extra = [c for c in current if c not in expected]

print(f"MISSING FROM CURRENT: {missing}")
print(f"EXTRA IN CURRENT: {extra}")

# Check order
if current == expected:
    print("ORDER MATCHES")
else:
    print("ORDER MISMATCH")
    for i in range(min(len(expected), len(current))):
        if expected[i] != current[i]:
            print(f"Mismatch at index {i}: Expected '{expected[i]}', Got '{current[i]}'")
            break
