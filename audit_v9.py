"""V9 Deep-State Forensic Audit Script — Research Only"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SECTION 3: RESIDUAL ERROR CLUSTERING
# ============================================================
print("=" * 70)
print("  SECTION 3: RESIDUAL ERROR CLUSTERING (2025 BACKTEST)")
print("=" * 70)

from features_v6 import engineer_features

df = pd.read_parquet("data/historical.parquet")
d25 = df[df.index.year == 2025].copy()

for h in [24, 48]:
    X, y = engineer_features(d25, horizon_h=h)
    mask = y.notna() & ~X.isna().any(axis=1)
    X_clean = X[mask].copy()
    y_clean = y[mask]

    model = joblib.load(f"models_v6/lgbm_point_{h}h.pkl")
    expected = list(model.feature_name_)
    for col in expected:
        if col not in X_clean.columns:
            X_clean[col] = 0.0
    X_clean = X_clean[expected].copy()
    X_clean["regime"] = pd.Categorical(X_clean["regime"])

    preds = model.predict(X_clean)
    residuals = y_clean.values - preds
    abs_errors = np.abs(residuals)

    print(f"\n--- {h}h Residual Statistics ---")
    print(f"  Mean Abs Error: {abs_errors.mean():.3f}")
    print(f"  Median Abs Error: {np.median(abs_errors):.3f}")
    print(f"  90th Percentile: {np.percentile(abs_errors, 90):.3f}")
    print(f"  95th Percentile: {np.percentile(abs_errors, 95):.3f}")
    print(f"  99th Percentile: {np.percentile(abs_errors, 99):.3f}")
    print(f"  Max Abs Error:   {abs_errors.max():.3f}")

    # Wind speed clustering
    curr_wind = d25.loc[X_clean.index, "wind_speed_10m"]
    hi_wind = curr_wind > curr_wind.quantile(0.8)
    lo_wind = curr_wind < curr_wind.quantile(0.2)
    print(f"\n  Error by Wind Speed:")
    print(f"    High wind (>80th pctl={curr_wind.quantile(0.8):.1f}): MAE={abs_errors[hi_wind].mean():.2f}  Bias={residuals[hi_wind].mean():+.2f}  n={hi_wind.sum()}")
    print(f"    Low wind  (<20th pctl={curr_wind.quantile(0.2):.1f}): MAE={abs_errors[lo_wind].mean():.2f}  Bias={residuals[lo_wind].mean():+.2f}  n={lo_wind.sum()}")

    # Pressure change
    curr_pres_diff = d25.loc[X_clean.index, "surface_pressure"].diff(48)
    pres_drop = curr_pres_diff < curr_pres_diff.quantile(0.1)
    pres_rise = curr_pres_diff > curr_pres_diff.quantile(0.9)
    pres_stable = (~pres_drop) & (~pres_rise)
    print(f"\n  Error by 48h Pressure Delta:")
    print(f"    Rapid drop (<10th pctl): MAE={abs_errors[pres_drop].mean():.2f}  Bias={residuals[pres_drop].mean():+.2f}  n={pres_drop.sum()}")
    print(f"    Stable (+/-):            MAE={abs_errors[pres_stable].mean():.2f}  Bias={residuals[pres_stable].mean():+.2f}  n={pres_stable.sum()}")
    print(f"    Rapid rise (>90th pctl): MAE={abs_errors[pres_rise].mean():.2f}  Bias={residuals[pres_rise].mean():+.2f}  n={pres_rise.sum()}")

    # AQI level at prediction time
    curr_aqi = d25.loc[X_clean.index, "us_aqi"]
    hi_aqi = curr_aqi > 75
    mid_aqi = (curr_aqi >= 35) & (curr_aqi <= 75)
    lo_aqi = curr_aqi < 35
    print(f"\n  Error by Current AQI Level:")
    print(f"    High AQI (>75  AQI): MAE={abs_errors[hi_aqi].mean():.2f}  Bias={residuals[hi_aqi].mean():+.2f}  n={hi_aqi.sum()}")
    print(f"    Mid  AQI (35-75   ): MAE={abs_errors[mid_aqi].mean():.2f}  Bias={residuals[mid_aqi].mean():+.2f}  n={mid_aqi.sum()}")
    print(f"    Low  AQI (<35  AQI): MAE={abs_errors[lo_aqi].mean():.2f}  Bias={residuals[lo_aqi].mean():+.2f}  n={lo_aqi.sum()}")

    # Seasonal clustering
    months = X_clean.index.month
    print(f"\n  Error by Season:")
    for season, mos in [("Winter", [12,1,2]), ("Spring", [3,4,5]), ("Summer", [6,7,8]), ("Fall", [9,10,11])]:
        mask_s = months.isin(mos)
        if mask_s.sum() > 0:
            print(f"    {season:8s}: MAE={abs_errors[mask_s].mean():.2f}  Bias={residuals[mask_s].mean():+.2f}  n={mask_s.sum()}")

    # Diurnal pattern of errors
    hrs = X_clean.index.hour
    print(f"\n  Error by Hour of Day (Diurnal):")
    for hr_bin, label in [((0,6),"Night (0-6am)"), ((6,12),"Morning (6am-12pm)"), ((12,18),"Afternoon (12-6pm)"), ((18,24),"Evening (6-12pm)")]:
        mask_h = (hrs >= hr_bin[0]) & (hrs < hr_bin[1])
        if mask_h.sum() > 0:
            print(f"    {label:22s}: MAE={abs_errors[mask_h].mean():.2f}  Bias={residuals[mask_h].mean():+.2f}")

    # Fat-tail events (AQI spikes)
    fat_tail_mask = y_clean.abs() > 20
    normal_mask = y_clean.abs() <= 20
    print(f"\n  Error by Target Magnitude:")
    print(f"    Fat-tail events (|target| > 20): MAE={abs_errors[fat_tail_mask].mean():.2f}  n={fat_tail_mask.sum()}")
    print(f"    Normal events   (|target| <= 20): MAE={abs_errors[normal_mask].mean():.2f}  n={normal_mask.sum()}")

# ============================================================
# SECTION 4: ACCUMULATION WINDOW TEST
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 4: TEMPORAL INTEGRITY — ACCUMULATION WINDOW")
print("=" * 70)

# Gaussian window via numpy
def gauss_fn(M, std):
    n = np.arange(0, M) - (M - 1.0) / 2.0
    return np.exp(-n**2 / (2 * std**2))
ghi = pd.to_numeric(df.get("shortwave_radiation", 0), errors="coerce").fillna(0)
temp = pd.to_numeric(df["temperature_2m"], errors="coerce")
rh = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
photochem = (ghi / 1000.0) * (rh / 100.0).clip(0.3, 1.0) * np.exp(0.069 * (temp - 25.0))

target_48 = df["us_aqi"].shift(-48) - df["us_aqi"]

print("\n  Correlation of accumulation windows vs 48h AQI residual target:")
for window in [3, 6, 9, 12, 18, 24]:
    simple = photochem.rolling(window, min_periods=1).sum().shift(-48)
    mask_v = ~(target_48.isna() | simple.isna())
    corr = np.corrcoef(simple[mask_v].values, target_48[mask_v].values)[0, 1]
    print(f"    Simple sum  w={window:2d}h: Corr={corr:.5f}")

# Gaussian-weighted window comparison
for window in [12, 18, 24]:
    wts = gauss_fn(window, std=window/4)
    wts = wts / wts.sum()
    def gwt(x):
        w = wts[-len(x):]
        w = w / w.sum()
        return np.dot(x, w)
    gauss_acc = photochem.rolling(window, min_periods=1).apply(gwt, raw=True).shift(-48)
    mask_v = ~(target_48.isna() | gauss_acc.isna())
    corr = np.corrcoef(gauss_acc[mask_v].values, target_48[mask_v].values)[0, 1]
    print(f"    Gaussian wt w={window:2d}h: Corr={corr:.5f}")

# ============================================================
# SECTION 5: ZERO-INFLATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 5: ZERO-INFLATION & TARGET DISTRIBUTION")
print("=" * 70)

aqi = df[df.index.year == 2025]["us_aqi"]
print(f"\n  2025 AQI Distribution:")
print(f"    Mean={aqi.mean():.1f}  Median={aqi.median():.1f}  Std={aqi.std():.1f}")
for threshold, label in [(30,"<30"), (40,"<40"), (50,"<50"), (75,">75"), (100,">100"), (150,">150")]:
    if "<" in label:
        pct = (aqi < threshold).mean() * 100
    else:
        pct = (aqi > threshold).mean() * 100
    print(f"    AQI {label}: {pct:.1f}%")

target_48_2025 = d25["us_aqi"].shift(-48) - d25["us_aqi"]
valid = target_48_2025.dropna()
print(f"\n  48h Residual Target Distribution:")
print(f"    Mean={valid.mean():.2f}  Std={valid.std():.2f}")
print(f"    Within +/-5:  {(valid.abs() < 5).mean()*100:.1f}%")
print(f"    Within +/-10: {(valid.abs() < 10).mean()*100:.1f}%")
print(f"    Within +/-20: {(valid.abs() < 20).mean()*100:.1f}%")
print(f"    Beyond +/-20: {(valid.abs() > 20).mean()*100:.1f}%")
print(f"    Positive skew: {valid[valid > 0].mean():.2f} (avg spike)")
print(f"    Negative skew: {valid[valid < 0].mean():.2f} (avg drop)")

# Month-level bias check
print(f"\n  Month-level systematic bias (48h residual target mean):")
for m in range(1, 13):
    sub = valid[valid.index.month == m]
    if len(sub) > 0:
        print(f"    Month {m:2d}: mean={sub.mean():+.2f}  std={sub.std():.2f}  n={len(sub)}")

print("\nAudit complete.")
