"""Quick diagnostic: compute regime distribution across the full 2020-2026 dataset."""

import pandas as pd
from features_v5 import classify_regime, REGIME_LABELS

df = pd.read_parquet("data/historical.parquet")
regime = classify_regime(df)

print("=" * 70)
print("  ATMOSPHERIC REGIME DISTRIBUTION (2020-2026)")
print("=" * 70)

total = len(regime)
for r in sorted(regime.unique()):
    count = int((regime == r).sum())
    pct = count / total * 100
    label = REGIME_LABELS.get(r, "Unknown")
    print(f"  Regime {r} ({label:<25s}): {count:>6,} rows  ({pct:>5.1f}%)")

print(f"\n  Total rows: {total:,}")

# Seasonal breakdown
print("\n" + "=" * 70)
print("  REGIME DISTRIBUTION BY SEASON")
print("=" * 70)
df["regime"] = regime.values
df["month"] = df.index.month

seasons = {
    "Winter (Dec-Feb)": [12, 1, 2],
    "Spring (Mar-May)": [3, 4, 5],
    "Summer (Jun-Aug)": [6, 7, 8],
    "Fall (Sep-Nov)":   [9, 10, 11],
}

for season_name, months in seasons.items():
    subset = df[df["month"].isin(months)]
    r_counts = subset["regime"].value_counts().sort_index()
    n = len(subset)
    parts = []
    for r in [0, 1, 2]:
        c = r_counts.get(r, 0)
        parts.append(f"R{r}={c/n*100:.1f}%")
    joined = "  ".join(parts)
    print(f"  {season_name:<22s}: {joined}")

# Mean AQI by regime
print("\n" + "=" * 70)
print("  MEAN AQI BY REGIME")
print("=" * 70)
for r in sorted(regime.unique()):
    mask = df["regime"] == r
    mean_aqi = df.loc[mask, "us_aqi"].mean()
    std_aqi = df.loc[mask, "us_aqi"].std()
    label = REGIME_LABELS.get(r, "Unknown")
    print(f"  Regime {r} ({label:<25s}): Mean AQI = {mean_aqi:.1f} +/- {std_aqi:.1f}")
