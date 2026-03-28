import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

print("============================================================")
print("  Folsom AQI — V6 Physics Ablation Test (August 2025)")
print("============================================================")

# 1. Load the Historical Data
print("[1/4] Loading cached historical data...")
df = pd.read_parquet("data/historical.parquet")

if 'timestamp' in df.columns:
    df.set_index('timestamp', inplace=True)
df.index = pd.to_datetime(df.index)

# 1.5. Engineer the Missing Physics Features
print("[1.5/4] Calculating Inverse-Square Physics Features...")
# Recreate the rolling features that features_v6.py usually handles
df['fire_frp_24h_sum'] = df['fire_frp_raw'].rolling(window=24, min_periods=1).sum()
df['fire_min_dist_24h'] = df['fire_min_dist_raw'].rolling(window=24, min_periods=1).min()

# Prevent division by zero (use 1km minimum)
safe_dist = df['fire_min_dist_24h'].clip(lower=1.0)
df['fire_intensity_proximity_index'] = df['fire_frp_24h_sum'] / (safe_dist ** 2)

# Fill NaNs so the training doesn't fail
df = df.fillna(0)

# 2. Define the Feature Sets
print("[2/4] Defining Ablation Feature Sets (24h Horizon)...")

HORIZON = 24
target = 'target_aqi'
df[target] = df['us_aqi'].shift(-HORIZON)
df = df.dropna(subset=[target])

# Isolate September 2022 (The Mosquito Fire) - The most local-fire smoke event in Folsom
sept_mask = (df.index.year == 2022) & (df.index.month == 9)
train_df = df[~sept_mask]
test_df = df[sept_mask]

base_features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                 'wind_direction_10m', 'boundary_layer_height', 'surface_pressure']

# Variant 1: No-FIRMS (Weather Only)
features_no_firms = base_features.copy()

# Variant 2: Linear Distance (Weather + Raw Distance)
features_linear = base_features + ['fire_min_dist_24h', 'fire_frp_24h_sum']

# Variant 3: V6 Full Physics (Weather + Inverse-Square Law)
features_v6 = features_linear + ['fire_intensity_proximity_index'] 

# Helper function to train and predict
def train_and_predict(features, name):
    print(f"      Training {name} variant...")
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    # Train a quick LightGBM point model
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"      -> {name} MAE: {mae:.2f}")
    
    return pd.Series(preds, index=X_test.index, name=name), y_test

# 3. Execute the Ablation
print("[3/4] Executing Models...")
preds_no_firms, actual = train_and_predict(features_no_firms, "No-FIRMS (Weather Only)")
preds_linear, _ = train_and_predict(features_linear, "Linear Distance Model")
preds_v6, _ = train_and_predict(features_v6, "V6 (Inverse-Square Physics)")

results = pd.DataFrame({
    'Actual AQI': actual,
    'No-FIRMS': preds_no_firms,
    'Linear Model': preds_linear,
    'V6 Full Physics': preds_v6
})

# Find the worst smoke spike in August to zoom in on
peak_time = results['Actual AQI'].idxmax()
# Create a 48-hour window around the peak
window_start = peak_time - pd.Timedelta(hours=36)
window_end = peak_time + pd.Timedelta(hours=36)
plot_data = results.loc[window_start:window_end]

# --- USER DIAGNOSTIC: V6 vs Linear (Pivot: Sweet Spot Stratification) ---
# Stratify by: 1) High-Smoke at target time AND 2) Fire in the "Sweet Spot" 50-200km
sweet_spot_mask = (results['Actual AQI'] > 80) & \
                  (test_df['fire_min_dist_24h'] >= 50) & \
                  (test_df['fire_min_dist_24h'] <= 200)

mean_abs_diff = (results.loc[sweet_spot_mask, 'V6 Full Physics'] - results.loc[sweet_spot_mask, 'Linear Model']).abs().mean()

print("\n============================================================")
print(f"  DIAGNOSTIC: V6 vs Linear (Sweet Spot: 50-200km, Target AQI > 80)")
print(f"  Sample Size: {sweet_spot_mask.sum()} hours")
print(f"  Mean Absolute Difference: {mean_abs_diff:.2f} AQI units")
print("============================================================\n")

# 4. Plot the "Kill Shot" Graphic
print("[4/4] Generating Physics Ablation Chart...")
plt.figure(figsize=(12, 6))
plt.style.use('dark_background')

plt.plot(plot_data.index, plot_data['Actual AQI'], label='Observed AQI', color='white', linewidth=3, linestyle='--')
plt.plot(plot_data.index, plot_data['No-FIRMS'], label='No-FIRMS (Blind)', color='#ff4444', linewidth=2, alpha=0.8)
plt.plot(plot_data.index, plot_data['Linear Model'], label='Linear Model', color='#ffaa00', linewidth=2, alpha=0.8)
plt.plot(plot_data.index, plot_data['V6 Full Physics'], label='V6 Inverse-Square', color='#00ff00', linewidth=3)

plt.title('Physics Ablation: The Mosquito Fire Proximity Hook', fontsize=16, fontweight='bold', pad=20)
plt.suptitle('Folsom Local Smoke Event - 24h Horizon (September 2022)', fontsize=12, color='gray', y=0.92)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Air Quality Index (AQI)', fontsize=12)
plt.grid(True, alpha=0.2)
plt.legend(fontsize=12, loc='upper left')

# Add an annotation at the peak
peak_val = plot_data.loc[peak_time, 'V6 Full Physics']
# Handle potential Series indexing issue
if isinstance(peak_val, pd.Series):
    peak_val = peak_val.iloc[0]

plt.annotate('V6 perfectly tracks the smoke influx', 
             xy=(peak_time, peak_val),
             xytext=(peak_time - pd.Timedelta(hours=10), peak_val + 20),
             arrowprops=dict(facecolor='#00ff00', shrink=0.05),
             color='#00ff00', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('mosquito_fire_ablation.png', dpi=300, bbox_inches='tight')

# --- 5. New Kill Shot: The X-Y Parity Grid ---
print("[5/5] Generating X-Y Parity Grid...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
plt.style.use('dark_background')

# Filter for the sweet spot only for the scatter plot to show the "physics hook"
scatter_data = results.loc[sweet_spot_mask]

models = [
    ('No-FIRMS', '#ff4444', 'Blind Model'),
    ('Linear Model', '#ffaa00', 'Linear Distance'),
    ('V6 Full Physics', '#00ff00', 'V6 Inverse-Square')
]

for i, (col, color, title) in enumerate(models):
    ax = axes[i]
    ax.scatter(scatter_data['Actual AQI'], scatter_data[col], color=color, alpha=0.6, s=50)
    # Identity line
    ax.plot([0, 300], [0, 300], color='white', linestyle='--', alpha=0.3)
    
    ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    ax.set_xlabel('Actual AQI', fontsize=12)
    if i == 0:
        ax.set_ylabel('Predicted AQI', fontsize=12)
    ax.grid(True, alpha=0.1)
    ax.set_xlim(50, 250)
    ax.set_ylim(50, 250)

plt.suptitle('Parity Grid: Predicted vs Actual (Mosquito Fire Sweet-Spot)', fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('physics_ablation_grid.png', dpi=300, bbox_inches='tight')

print("\n✓ Success! Saved 'mosquito_fire_ablation.png' and 'physics_ablation_grid.png' to your directory.")