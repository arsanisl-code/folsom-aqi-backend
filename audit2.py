import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

df = pd.read_parquet("data/historical.parquet")

def gauss_fn(M, std):
    n = np.arange(0, M) - (M - 1.0) / 2.0
    return np.exp(-n**2 / (2 * std**2))

ghi = pd.to_numeric(df.get("shortwave_radiation", 0), errors="coerce").fillna(0)
temp = pd.to_numeric(df["temperature_2m"], errors="coerce")
rh = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
photochem = (ghi / 1000.0) * (rh / 100.0).clip(0.3, 1.0) * np.exp(0.069 * (temp - 25.0))
target_48 = df["us_aqi"].shift(-48) - df["us_aqi"]

print("=== SECTION 4: ACCUMULATION WINDOW CORRELATION TEST ===")
for window in [3, 6, 9, 12, 18, 24]:
    simple = photochem.rolling(window, min_periods=1).sum().shift(-48)
    mask_v = ~(target_48.isna() | simple.isna())
    corr = np.corrcoef(simple[mask_v].values, target_48[mask_v].values)[0, 1]
    print(f"  Simple sum  w={window:2d}h shifted-48: Corr={corr:.5f}")

print()
for window in [12, 18, 24]:
    wts = gauss_fn(window, window/4)
    wts = wts / wts.sum()
    def gwt(x, w=wts):
        w2 = w[-len(x):]
        return np.dot(x, w2 / w2.sum())
    gauss_acc = photochem.rolling(window, min_periods=1).apply(gwt, raw=True).shift(-48)
    mask_v = ~(target_48.isna() | gauss_acc.isna())
    corr = np.corrcoef(gauss_acc[mask_v].values, target_48[mask_v].values)[0, 1]
    print(f"  Gaussian wt w={window:2d}h shifted-48: Corr={corr:.5f}")

print()
print("=== SECTION 5: TARGET DISTRIBUTION ===")
aqi = df[df.index.year == 2025]["us_aqi"]
print(f"2025 AQI: Mean={aqi.mean():.1f} Median={aqi.median():.1f} Std={aqi.std():.1f}")
for thr, label in [(30,"<30"),(40,"<40"),(50,"<50"),(75,">75"),(100,">100"),(150,">150")]:
    pct = ((aqi < thr) if label[0] == "<" else (aqi > thr)).mean()*100
    print(f"  AQI {label}: {pct:.1f}%")

d25 = df[df.index.year == 2025].copy()
valid = (d25["us_aqi"].shift(-48) - d25["us_aqi"]).dropna()
print(f"48h residual: Mean={valid.mean():.2f} Std={valid.std():.2f}")
print(f"  Within +/-5:  {(valid.abs()<5).mean()*100:.1f}%")
print(f"  Within +/-10: {(valid.abs()<10).mean()*100:.1f}%")
print(f"  Beyond +/-20: {(valid.abs()>20).mean()*100:.1f}%")
print(f"  Avg spike (+): {valid[valid>0].mean():.2f}")
print(f"  Avg drop  (-): {valid[valid<0].mean():.2f}")

print()
print("Month-level systematic BIAS in 48h residual target (2025):")
for m in range(1,13):
    sub = valid[valid.index.month == m]
    if len(sub) > 0:
        print(f"  Month {m:2d}: bias={sub.mean():+.2f} std={sub.std():.2f} n={len(sub)}")
