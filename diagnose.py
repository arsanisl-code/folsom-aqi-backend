"""
diagnose.py — Deep-dive diagnostic on 48h R² and 12h coverage issues.

Produces:
  1. Feature importance analysis (48h point model)
  2. Residual distribution analysis (12h model — fat-tail detection)
  3. Correlation matrix (48h target_residual vs weather features)
  4. Console summary with action recommendations

Usage:
    python diagnose.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from features import engineer_features

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
HIST_PATH = DATA_DIR / "historical.parquet"

# ─── Feature importance analysis ─────────────────────────────────────────────

def analyze_feature_importance(horizon_h: int = 48, top_n: int = 20):
    """
    Load the point model for the given horizon and rank features by
    LightGBM's built-in 'gain' importance. Identifies whether lag-0
    features dominate at long horizons (a sign the model is anchoring
    to current AQI, which has no real predictive power 2 days out).
    """
    print(f"\n{'=' * 70}")
    print(f"  FEATURE IMPORTANCE — {horizon_h}h POINT MODEL (Top {top_n})")
    print(f"{'=' * 70}")

    model = joblib.load(MODELS_DIR / f"lgbm_point_{horizon_h}h.pkl")

    # LightGBM stores feature importances by gain (total split gain)
    importances = model.feature_importances_
    feature_names = model.feature_name_

    # Sort descending by importance
    idx = np.argsort(importances)[::-1]

    total_gain = importances.sum()
    cumulative = 0.0

    print(f"\n  {'Rank':<6} {'Feature':<35} {'Gain':>12} {'% Total':>10} {'Cumul%':>10}")
    print(f"  {'-' * 75}")

    lag0_features = []
    lag0_gain = 0.0

    for rank, i in enumerate(idx[:top_n], 1):
        name = feature_names[i]
        gain = importances[i]
        pct = gain / total_gain * 100
        cumulative += pct

        # Flag lag-0 / current features
        is_lag0 = any(tag in name for tag in [
            'aqi_current', 'pm25_current', 'aqi_ewma',
            'pm25_ewma', 'aqi_diff_1h', 'aqi_x_wind',
            'aqi_x_rad', 'aod_current'
        ])
        marker = " ◄ LAG-0" if is_lag0 else ""

        if is_lag0:
            lag0_features.append(name)
            lag0_gain += gain

        print(f"  {rank:<6} {name:<35} {gain:>12.0f} {pct:>9.1f}% {cumulative:>9.1f}%{marker}")

    lag0_pct = lag0_gain / total_gain * 100
    print(f"\n  ── Lag-0 / Current-state features ──")
    print(f"  Total lag-0 gain share: {lag0_pct:.1f}%")
    print(f"  Features: {', '.join(lag0_features)}")

    if lag0_pct > 40:
        print(f"  ⚠ OVER-RELIANCE: {lag0_pct:.1f}% of gain comes from current-state "
              f"features that have NO predictive power 48h out.")
        print(f"    The model is essentially predicting 'AQI stays roughly the same'")
        print(f"    which is the persistence baseline. This explains the negative R².")
    elif lag0_pct > 20:
        print(f"  ⚠ MODERATE: {lag0_pct:.1f}% of gain from lag-0 features.")
    else:
        print(f"  ✓ OK: Only {lag0_pct:.1f}% of gain from lag-0 features.")

    return {
        "horizon_h": horizon_h,
        "lag0_gain_pct": round(lag0_pct, 1),
        "lag0_features": lag0_features,
        "top_features": [
            {"name": feature_names[i], "gain_pct": round(importances[i] / total_gain * 100, 2)}
            for i in idx[:top_n]
        ],
    }


# ─── Residual analysis ───────────────────────────────────────────────────────

def analyze_residuals(df: pd.DataFrame, horizon_h: int = 12):
    """
    Analyze the error distribution for the given horizon's point model.
    Checks for fat tails (kurtosis), skewness, and outlier frequency.
    Fat tails explain why 90% confidence intervals are too narrow:
    the quantile models underestimate extreme-event probability.
    """
    print(f"\n{'=' * 70}")
    print(f"  RESIDUAL ANALYSIS — {horizon_h}h POINT MODEL")
    print(f"{'=' * 70}")

    model = joblib.load(MODELS_DIR / f"lgbm_point_{horizon_h}h.pkl")
    imputer = joblib.load(MODELS_DIR / f"imputer_{horizon_h}h.pkl")
    q05_model = joblib.load(MODELS_DIR / f"lgbm_q05_{horizon_h}h.pkl")
    q95_model = joblib.load(MODELS_DIR / f"lgbm_q95_{horizon_h}h.pkl")

    # Build features
    X, y = engineer_features(df, horizon_h)
    mask = y.notna()
    X, y = X[mask], y[mask]

    # Use last 60 days as the analysis window (same as val set)
    cutoff = X.index.max() - pd.Timedelta(days=60)
    val_mask = X.index >= cutoff
    X_val, y_val = X[val_mask], y[val_mask]

    if len(X_val) == 0:
        print("  [ERROR] No validation data available.")
        return {}

    # Impute and predict
    X_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    pred_res = model.predict(X_imp)
    pred_q05_res = q05_model.predict(X_imp)
    pred_q95_res = q95_model.predict(X_imp)

    # Invert residuals to absolute AQI
    base_aqi = X_val['aqi_current'].values
    pred_abs = pred_res + base_aqi
    y_abs = y_val.values + base_aqi
    pred_q05_abs = pred_q05_res + base_aqi
    pred_q95_abs = pred_q95_res + base_aqi

    # Compute errors
    errors = y_abs - pred_abs
    abs_errors = np.abs(errors)

    # Coverage
    covered = (y_abs >= pred_q05_abs) & (y_abs <= pred_q95_abs)
    coverage = np.mean(covered) * 100

    # Error distribution statistics
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    skewness = stats.skew(errors)
    kurtosis = stats.kurtosis(errors)  # excess kurtosis (0 = normal)
    mae = np.mean(abs_errors)

    # Percentile analysis of absolute errors
    p50 = np.percentile(abs_errors, 50)
    p90 = np.percentile(abs_errors, 90)
    p95 = np.percentile(abs_errors, 95)
    p99 = np.percentile(abs_errors, 99)

    # Count outliers (errors > 2σ and > 3σ)
    n_2sigma = np.sum(abs_errors > 2 * std_err)
    n_3sigma = np.sum(abs_errors > 3 * std_err)
    pct_2sigma = n_2sigma / len(errors) * 100
    pct_3sigma = n_3sigma / len(errors) * 100

    # For a normal distribution: ~5% beyond 2σ, ~0.3% beyond 3σ
    expected_2sigma_pct = 4.55
    expected_3sigma_pct = 0.27

    # Analyze CI width vs actual error spread
    ci_widths = pred_q95_abs - pred_q05_abs
    mean_ci_width = np.mean(ci_widths)

    # Where does coverage fail?
    uncovered = ~covered
    uncovered_errors = errors[uncovered]
    below_lower = y_abs < pred_q05_abs
    above_upper = y_abs > pred_q95_abs

    print(f"\n  Error Distribution:")
    print(f"    Mean error:     {mean_err:>8.2f} AQI  (bias)")
    print(f"    Std error:      {std_err:>8.2f} AQI")
    print(f"    Skewness:       {skewness:>8.3f}  (0=symmetric)")
    print(f"    Excess kurtosis:{kurtosis:>8.3f}  (0=normal, >0=fat tails)")
    print(f"    MAE:            {mae:>8.2f} AQI")

    print(f"\n  Absolute Error Percentiles:")
    print(f"    P50 (median):   {p50:>8.2f} AQI")
    print(f"    P90:            {p90:>8.2f} AQI")
    print(f"    P95:            {p95:>8.2f} AQI")
    print(f"    P99:            {p99:>8.2f} AQI")
    print(f"    P99/P50 ratio:  {p99/p50:>8.1f}×  (normal≈3.8×, fat-tail>5×)")

    print(f"\n  Outlier Frequency:")
    print(f"    >2σ: {n_2sigma} ({pct_2sigma:.1f}%)  expected for normal: {expected_2sigma_pct:.1f}%")
    print(f"    >3σ: {n_3sigma} ({pct_3sigma:.1f}%)  expected for normal: {expected_3sigma_pct:.1f}%")

    fat_tail = kurtosis > 1.0 or pct_3sigma > 1.0 or (p99 / max(p50, 0.01)) > 5.0
    if fat_tail:
        print(f"\n  ⚠ FAT TAILS DETECTED:")
        print(f"    Excess kurtosis = {kurtosis:.2f} (normal=0) → errors have heavy tails")
        print(f"    The q05/q95 quantile models are fitting the BULK of the distribution")
        print(f"    but miss the EXTREME tails, producing intervals that are too narrow.")
    else:
        print(f"\n  ✓ Tail distribution appears near-normal.")

    print(f"\n  Coverage Analysis:")
    print(f"    Overall coverage: {coverage:.1f}%  (target: ≥90%)")
    print(f"    Mean CI width:    {mean_ci_width:.1f} AQI")
    print(f"    Failures below q01: {np.sum(below_lower)} ({np.sum(below_lower)/len(y_abs)*100:.1f}%)")
    print(f"    Failures above q99: {np.sum(above_upper)} ({np.sum(above_upper)/len(y_abs)*100:.1f}%)")

    if np.sum(above_upper) > np.sum(below_lower) * 2:
        print(f"    → Asymmetric failures: model under-predicts extreme HIGH AQI events")
    elif np.sum(below_lower) > np.sum(above_upper) * 2:
        print(f"    → Asymmetric failures: model over-predicts (actual AQI often lower)")

    return {
        "horizon_h": horizon_h,
        "mae": round(mae, 2),
        "bias": round(mean_err, 2),
        "std": round(std_err, 2),
        "skewness": round(skewness, 3),
        "kurtosis": round(kurtosis, 3),
        "coverage": round(coverage, 1),
        "ci_width": round(mean_ci_width, 1),
        "fat_tails": bool(fat_tail),
        "p99_p50_ratio": round(p99 / max(p50, 0.01), 1),
        "pct_above_3sigma": round(pct_3sigma, 1),
        "failures_below": int(np.sum(below_lower)),
        "failures_above": int(np.sum(above_upper)),
    }


# ─── Correlation analysis ────────────────────────────────────────────────────

def analyze_correlations(df: pd.DataFrame, horizon_h: int = 48):
    """
    Compute Pearson and Spearman correlations between the 48h target_residual
    and weather features. Identifies which features actually carry signal
    at a 2-day horizon vs which are just noise.
    """
    print(f"\n{'=' * 70}")
    print(f"  CORRELATION ANALYSIS — {horizon_h}h TARGET vs FEATURES")
    print(f"{'=' * 70}")

    X, y = engineer_features(df, horizon_h)
    mask = y.notna()
    X, y = X[mask], y[mask]

    # Also compute correlations with absolute target for comparison
    y_abs = y.values + X['aqi_current'].values

    # Weather features to check
    weather_features = [
        'boundary_layer_height', 'wind_speed_10m', 'surface_pressure',
        'relative_humidity_2m', 'temperature_2m', 'precipitation',
        'cloud_cover', 'direct_radiation', 'soil_temp',
        'blh_x_wind_speed', 'wind_dir_sin', 'wind_dir_cos',
        'wildfire_hdwi', 'precip_30d_sum', 'flag_extreme_heat_dry',
    ]

    # AQI lag features
    aqi_features = [
        'aqi_current', 'aqi_lag_1h', 'aqi_lag_6h', 'aqi_lag_12h',
        'aqi_lag_24h', 'aqi_lag_48h',
        'aqi_diff_1h', 'aqi_diff_24h',
        'aqi_roll_24h_mean', 'aqi_roll_168h_mean',
        'aqi_ewma_6h', 'aqi_ewma_24h',
        'pm25_current', 'pm25_roll_24h_mean',
    ]

    # Pressure differencing features
    pressure_features = [
        'pressure_diff_3h', 'pressure_diff_6h',
        'pressure_diff_12h', 'pressure_diff_24h', 'temp_diff_24h',
    ]

    # AOD features
    aod_features = [f for f in ['aod_current', 'aod_diff_3h', 'aod_diff_6h']
                    if f in X.columns]

    # Temporal features
    temporal_features = [
        'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos',
        'future_hour_sin', 'future_hour_cos',
    ]

    all_features = weather_features + aqi_features + pressure_features + aod_features + temporal_features
    available = [f for f in all_features if f in X.columns]

    results = []
    for feat in available:
        col = X[feat].values
        valid = ~(np.isnan(col) | np.isnan(y.values))
        if valid.sum() < 100:
            continue

        pearson_r, pearson_p = stats.pearsonr(col[valid], y.values[valid])
        spearman_r, spearman_p = stats.spearmanr(col[valid], y.values[valid])

        results.append({
            "feature": feat,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "abs_pearson": abs(pearson_r),
        })

    # Sort by absolute Pearson correlation
    results.sort(key=lambda x: x["abs_pearson"], reverse=True)

    # Print grouped by category
    def _print_group(title, feature_list):
        group = [r for r in results if r["feature"] in feature_list]
        if not group:
            return
        group.sort(key=lambda x: x["abs_pearson"], reverse=True)
        print(f"\n  {title}:")
        print(f"    {'Feature':<30} {'Pearson r':>10} {'p-value':>12} {'Spearman r':>12} {'Signal?'}")
        print(f"    {'-' * 70}")
        for r in group:
            sig = "✓ YES" if r["abs_pearson"] > 0.05 and r["pearson_p"] < 0.01 else "✗ no"
            strength = ""
            if r["abs_pearson"] > 0.3:
                strength = " (STRONG)"
            elif r["abs_pearson"] > 0.1:
                strength = " (moderate)"
            elif r["abs_pearson"] > 0.05:
                strength = " (weak)"
            print(f"    {r['feature']:<30} {r['pearson_r']:>+10.4f} {r['pearson_p']:>12.2e} "
                  f"{r['spearman_r']:>+12.4f} {sig}{strength}")

    _print_group("Weather Features (current values at T)", weather_features)
    _print_group("AQI / PM2.5 Features", aqi_features)
    _print_group("Pressure & Temp Differencing", pressure_features)
    _print_group("Aerosol Optical Depth", aod_features)
    _print_group("Temporal Encodings", temporal_features)

    # Summary: how many features have |r| > 0.1?
    strong = [r for r in results if r["abs_pearson"] > 0.1]
    moderate = [r for r in results if 0.05 < r["abs_pearson"] <= 0.1]
    weak = [r for r in results if r["abs_pearson"] <= 0.05]

    print(f"\n  ── Signal Summary ──")
    print(f"    Strong (|r|>0.1):   {len(strong)} features")
    print(f"    Moderate (|r|>0.05):{len(moderate)} features")
    print(f"    Noise (|r|≤0.05):   {len(weak)} features")

    if len(strong) < 5:
        print(f"\n  ⚠ VERY FEW STRONG SIGNALS at {horizon_h}h horizon.")
        print(f"    The weather features measured at time T have minimal linear")
        print(f"    correlation with AQI change 48h later. The model is essentially")
        print(f"    guessing, which explains the negative R².")

    return results


# ─── Persistence baseline analysis ───────────────────────────────────────────

def analyze_persistence(df: pd.DataFrame):
    """
    For each horizon, compute how the persistence baseline (AQI stays the same)
    compares to the model. This reveals at which horizon the model stops
    adding value over naive persistence.
    """
    print(f"\n{'=' * 70}")
    print(f"  PERSISTENCE BASELINE COMPARISON")
    print(f"{'=' * 70}")

    for horizon_h in [6, 12, 24, 48]:
        X, y = engineer_features(df, horizon_h)
        mask = y.notna()
        X, y = X[mask], y[mask]

        cutoff = X.index.max() - pd.Timedelta(days=60)
        val_mask = X.index >= cutoff
        X_val, y_val = X[val_mask], y[val_mask]

        base_aqi = X_val['aqi_current'].values
        y_abs = y_val.values + base_aqi

        # Persistence = current AQI (predict no change)
        persistence_mae = mean_absolute_error(y_abs, base_aqi)
        persistence_r2 = r2_score(y_abs, base_aqi)

        # Model prediction
        model = joblib.load(MODELS_DIR / f"lgbm_point_{horizon_h}h.pkl")
        imputer = joblib.load(MODELS_DIR / f"imputer_{horizon_h}h.pkl")
        X_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
        pred_res = model.predict(X_imp)
        pred_abs = pred_res + base_aqi

        model_mae = mean_absolute_error(y_abs, pred_abs)
        model_r2 = r2_score(y_abs, pred_abs)
        skill = 1 - (model_mae / persistence_mae) if persistence_mae > 0 else 0

        marker = "✓" if model_mae < persistence_mae else "⚠ WORSE THAN PERSISTENCE"
        print(f"\n  {horizon_h}h:  Pers MAE={persistence_mae:.2f}  Model MAE={model_mae:.2f}  "
              f"Skill={skill:.3f}  Pers R²={persistence_r2:.3f}  Model R²={model_r2:.3f}  {marker}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not HIST_PATH.exists():
        print(f"[diagnose] ERROR: {HIST_PATH} not found. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("  Folsom AQI — Model Diagnostic Deep-Dive")
    print("=" * 70)

    df = pd.read_parquet(HIST_PATH)
    print(f"  Data: {len(df):,} rows | {df.index.min()} → {df.index.max()}")

    # 1. Feature importance (48h)
    fi_results = analyze_feature_importance(48, top_n=20)

    # 2. Residual analysis (12h and 48h)
    res_12h = analyze_residuals(df, 12)
    res_48h = analyze_residuals(df, 48)

    # 3. Correlation analysis (48h)
    corr_results = analyze_correlations(df, 48)

    # 4. Persistence comparison
    analyze_persistence(df)

    # 5. Action recommendations
    print(f"\n{'=' * 70}")
    print(f"  ACTION RECOMMENDATIONS")
    print(f"{'=' * 70}")

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ RECOMMENDATION 1: Fix 12h Coverage (84.7% → ≥90%)                 │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │ Root cause: Fat-tailed error distribution. The q05/q95 quantile    │
  │ models learn the central mass but underestimate extreme events.     │
  │                                                                     │
  │ Fix A — Coverage-aware Optuna objective (tune.py):                 │
  │   Change the quantile model objective to a composite score:        │
  │     score = pinball_loss + λ * max(0, 0.90 - coverage)²           │
  │   where λ=100 penalizes under-coverage quadratically.              │
  │   This forces the sampler toward wider intervals.                  │
  │                                                                     │
  │ Fix B — Post-hoc conformal calibration (inference.py):             │
  │   After training, compute conformity scores on the calibration     │
  │   set (last 60 days). At inference, inflate the CI by the          │
  │   empirical (1-α) quantile of conformity scores. This guarantees  │
  │   marginal coverage without retraining.                            │
  │                                                                     │
  │ Fix C — Widen quantile alphas:                                     │
  │   Train q05→q005 (α=0.005) and q95→q995 (α=0.995).              │
  │   Wider native quantiles capture more tail mass automatically.     │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ RECOMMENDATION 2: Save 48h R² (currently negative)                 │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │ Root cause: The model over-relies on lag-0 features (current AQI)  │
  │ which have zero predictive power 2 days out. Weather features at   │
  │ time T also have near-zero correlation with AQI change at T+48h.  │
  │                                                                     │
  │ Fix A — Add weather FORECAST features (features.py):               │
  │   Open-Meteo provides hourly forecasts for the next 7 days.        │
  │   Instead of using weather at time T, use the FORECAST weather     │
  │   at time T+48h. This gives the model direct access to what the    │
  │   atmosphere will actually look like when the prediction lands.    │
  │   Key features: forecast wind, forecast BLH, forecast precip.      │
  │                                                                     │
  │ Fix B — Add multi-day weather trend features (features.py):        │
  │   Extend pressure/temp differencing to 48h and 72h windows:       │
  │     pressure_diff_48h = surface_pressure.diff(48)                  │
  │     temp_diff_48h = temperature_2m.diff(48)                        │
  │   Weather fronts evolve over days, not hours.                      │
  │                                                                     │
  │ Fix C — Reduce model complexity at 48h (train.py / tune.py):      │
  │   The Optuna-tuned params may have too much capacity for a         │
  │   low-signal horizon. Constrain the 48h search space:              │
  │     num_leaves: [15, 63], max_depth: [3, 6]                       │
  │   Simpler models generalize better when signal is weak.            │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ RECOMMENDATION 3: General robustness improvement                   │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │ Fix — Horizon-adaptive regularization in tune.py:                  │
  │   Create separate search spaces per horizon tier:                  │
  │     Short (6h, 12h):  num_leaves=[31,255], max_depth=[4,12]       │
  │     Long  (24h, 48h): num_leaves=[15,63],  max_depth=[3,6],       │
  │                        min_data_in_leaf=[30,200]                   │
  │   This prevents the sampler from finding complex configs that      │
  │   overfit at long horizons where signal-to-noise is low.           │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")

    # Save diagnostic results
    diag_path = MODELS_DIR / "diagnostic_results.json"
    with open(diag_path, "w") as f:
        json.dump({
            "feature_importance_48h": fi_results,
            "residuals_12h": res_12h,
            "residuals_48h": res_48h,
        }, f, indent=2)
    print(f"  Diagnostic data saved → {diag_path}")


if __name__ == "__main__":
    main()
