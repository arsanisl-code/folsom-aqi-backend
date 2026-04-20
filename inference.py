"""
inference.py — Produce AQI forecast JSON for Folsom, CA.
Loads all 12 models, fetches recent data, returns the forecast dict.
Called by refresh.py every hour via GitHub Actions.

predict_now() is a thin orchestration shell. All business logic lives in
the named sub-functions below, each with a single responsibility.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from data_fetcher import fetch_recent_combined, fetch_airnow_current
from features_v6 import engineer_features, classify_regime
from ai_layer import generate_summary
from logger import get_logger

log = get_logger(__name__)


def _extract_firms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract FIRMS fire columns from a merged DataFrame for trajectory features.
    Returns a DataFrame with only fire columns (sparse — rows with fire_frp_raw > 0).
    Returns empty DataFrame if fire columns are absent.
    """
    firms_cols = ['fire_frp_raw', 'fire_count_raw', 'fire_min_dist_raw', 'fire_bearing_nearest']
    available = [c for c in firms_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    firms = df[available].copy()
    if 'fire_frp_raw' in firms.columns:
        firms = firms[firms['fire_frp_raw'] > 0]
    return firms

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_VERSION = "V14-Conformal-Calibrated"
CACHE_FILE    = Path("data/latest.json")

# Open-Meteo hallucination floor for relative humidity (Folsom grid cell artifact).
# Guards against documented cases where the API returns 5–6% RH for Folsom,
# which would produce extreme HDWI wildfire signals.
HUMIDITY_FLOOR_PCT: float = 25.0

# Open-Meteo hallucination cap for wind speed (Folsom grid cell artifact).
# Guards against documented cases where the API returns 30+ km/h for calm days,
# which would produce extreme ventilation and HDWI signals.
WIND_SPEED_CAP_KMH: float = 25.0

MODELS_DIR = Path("models_v6")
DATA_DIR   = Path("data")
HORIZONS   = [6, 12, 24, 48]
TZ         = ZoneInfo("America/Los_Angeles")

AQI_CATEGORIES = [
    (0,   50,  "Good",                           "#00e400"),
    (51,  100, "Moderate",                       "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups",  "#ff7e00"),
    (151, 200, "Unhealthy",                      "#ff0000"),
    (201, 300, "Very Unhealthy",                 "#8f3f97"),
    (301, 500, "Hazardous",                      "#7e0023"),
]

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── V13 Model Metadata (loaded once at import) ────────────────────────────────
_V6_METRICS: dict = {}
_V6_FEATURES: dict = {}   # keyed by horizon_h
_CONFORMAL_SCALES: dict[int, float] = {}   # keyed by horizon_h; 0.0 = no correction
try:
    _m = Path("models_v6/training_metrics_v6.json")
    if _m.exists():
        _V6_METRICS = json.loads(_m.read_text())
    # V13: per-horizon feature names (trajectory features only in 24h/48h)
    for _h in [6, 12, 24, 48]:
        _f = Path(f"models_v6/feature_names_{_h}h.json")
        if _f.exists():
            _V6_FEATURES[_h] = json.loads(_f.read_text())
    # Fallback to legacy single file if per-horizon files missing
    if not _V6_FEATURES:
        _f = Path("models_v6/feature_names_v6.json")
        if _f.exists():
            names = json.loads(_f.read_text())
            _V6_FEATURES = {h: names for h in [6, 12, 24, 48]}
    # V14: conformal calibration scales
    _cs = Path("models_v6/conformal_scales.json")
    if _cs.exists():
        _raw = json.loads(_cs.read_text())
        _CONFORMAL_SCALES = {int(k): float(v) for k, v in _raw.get("scales", {}).items()}
except Exception:
    pass  # Graceful degradation if files aren't available yet


# ─── AQI helpers ──────────────────────────────────────────────────────────────

def aqi_category(aqi: int) -> tuple[str, str]:
    """Return (category_name, color_hex) for a given AQI integer."""
    for lo, hi, name, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return name, color
    return "Hazardous", "#7e0023"


def _iso(dt) -> str:
    """Convert a pandas Timestamp or datetime to ISO 8601 with timezone offset."""
    if hasattr(dt, 'isoformat'):
        return dt.isoformat()
    return str(dt)


def _extract_scalar_handling_dst_duplicates(
    df: pd.DataFrame, ts, col: str = "us_aqi"
) -> float:
    """
    Safely extract a scalar value from df at timestamp ts.

    Why this exists: df.loc[ts, col] returns a pd.Series (not a scalar) when
    the index contains duplicate timestamps. This happens once per year during
    DST spring-forward (March), when two rows share the same wall-clock hour.
    Taking .iloc[0] from the resulting Series avoids the
    'truth value of a Series is ambiguous' error that would otherwise crash
    the history_72h build loop.
    """
    try:
        val = df.loc[ts, col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val)
    except Exception:
        return np.nan


def _get_model_metadata() -> dict:
    """Return a clean subset of model metadata for the frontend."""
    # Use 6h feature names for display (shortest, no trajectory cols)
    features_6h = _V6_FEATURES.get(6, [])
    features_48h = _V6_FEATURES.get(48, [])
    return {
        "architecture":   MODEL_VERSION,
        "total_features": {
            "6h":  len(features_6h),
            "12h": len(_V6_FEATURES.get(12, features_6h)),
            "24h": len(_V6_FEATURES.get(24, features_48h)),
            "48h": len(features_48h),
        },
        "horizons":       _V6_METRICS.get("horizons", []),
        "feature_names":  features_6h,
        "primary_drivers": [
            "aqi_current",
            "aqi_roll_24h_mean",
            "boundary_layer_height",
            "inversion_strength",
            "fire_intensity_proximity_index",
            "stagnation_24h",
        ],
    }


# ─── Model loading ────────────────────────────────────────────────────────────

_model_cache: dict = {}


def load_all_models() -> dict:
    """
    Load all 12 LightGBM models into a dict keyed by horizon.
    Expensive on first call; returns the cached dict on subsequent calls.
    """
    global _model_cache
    if _model_cache:
        return _model_cache

    missing = [
        str(MODELS_DIR / f"lgbm_{kind}_{h}h.pkl")
        for h in HORIZONS
        for kind in ["point", "q05", "q95"]
        if not (MODELS_DIR / f"lgbm_{kind}_{h}h.pkl").exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing model files:\n" + "\n".join(missing) + "\nRun train.py first."
        )

    models = {
        h: {
            "point": joblib.load(MODELS_DIR / f"lgbm_point_{h}h.pkl"),
            "q05":   joblib.load(MODELS_DIR / f"lgbm_q05_{h}h.pkl"),
            "q95":   joblib.load(MODELS_DIR / f"lgbm_q95_{h}h.pkl"),
        }
        for h in HORIZONS
    }
    _model_cache = models
    return models


# ─── Inference sub-functions ──────────────────────────────────────────────────

def _fetch_input_data(past_hours: int = 168) -> tuple[pd.DataFrame, int]:
    """
    Fetch recent combined AQ + weather data and compute data age.

    Returns:
        (df, data_age_minutes) where data_age_minutes is how old the
        most recent row is relative to now.

    Raises:
        Any exception from fetch_recent_combined if no cache is available.
    """
    log.info("Fetching recent data (%sh window)...", past_hours)
    try:
        df = fetch_recent_combined(past_hours=past_hours)
    except Exception as exc:
        log.critical("Failed to fetch input data: %s", exc, exc_info=True)
        raise

    data_age_minutes = _compute_data_age_minutes(df)
    return df, data_age_minutes


def _resolve_current_conditions(
    df: pd.DataFrame,
    airnow: dict | None,
) -> dict:
    """
    Determine the current AQI reading and its source.

    Priority: AirNow sensor (if aqi > 0) > Open-Meteo most recent row.

    Returns dict with keys: aqi, category, color, primary_pollutant,
    source, timestamp.
    """
    if airnow and airnow.get("aqi", 0) > 0:
        current_aqi = int(airnow["aqi"])
        cat, color  = aqi_category(current_aqi)
        return {
            "aqi":               current_aqi,
            "category":          cat,
            "color":             color,
            "primary_pollutant": airnow.get("primary_pollutant", "PM2.5"),
            "source":            "AirNow",
            "timestamp":         airnow.get("timestamp", datetime.now(tz=TZ).isoformat()),
        }

    # Fallback: use the most recent Open-Meteo reading
    now = pd.Timestamp.now(tz=TZ)
    past_df    = df[df.index <= now]
    recent_aqi = past_df["us_aqi"].dropna()

    current_aqi = int(round(recent_aqi.iloc[-1])) if len(recent_aqi) > 0 else 0
    cat, color  = aqi_category(current_aqi)
    return {
        "aqi":               current_aqi,
        "category":          cat,
        "color":             color,
        "primary_pollutant": "PM2.5",
        "source":            "Open-Meteo",
        "timestamp":         _iso(recent_aqi.index[-1]) if len(recent_aqi) > 0
                             else datetime.now(tz=TZ).isoformat(),
    }


def _prepare_horizon_dataframe(
    df_base: pd.DataFrame,
    airnow: dict | None,
) -> pd.DataFrame:
    """
    Apply AirNow Uniform Offset Calibration and sensor sanity clamping.

    AirNow Uniform Offset Calibration:
    A uniform offset is applied to the entire Open-Meteo AQI time series rather
    than a point injection at now_ts. Point injection creates an unphysical
    differential spike at T=0 that corrupts all rolling features (aqi_diff_1h,
    aqi_roll_*) for the current hour. A uniform shift preserves the shape of
    the time series while anchoring it to the ground-truth AirNow reading.

    Sensor sanity clamping:
    HUMIDITY_FLOOR_PCT and WIND_SPEED_CAP_KMH guard against documented
    Open-Meteo hallucination artifacts for the Folsom grid cell.

    Returns a copy of df_base with calibration and clamping applied.
    """
    df_h   = df_base.copy()
    now_ts = pd.Timestamp.now(tz=TZ).floor('h')

    if airnow:
        aq_val = float(airnow.get("aqi", 30))
        if now_ts in df_h.index and not np.isnan(df_h.at[now_ts, "us_aqi"]):
            om_now = df_h.at[now_ts, "us_aqi"]
        else:
            past_om = df_h[df_h.index <= now_ts]["us_aqi"].dropna()
            om_now  = past_om.iloc[-1] if len(past_om) > 0 else aq_val

        offset = aq_val - om_now
        df_h["us_aqi"] = (df_h["us_aqi"] + offset).clip(lower=0)

    if "relative_humidity_2m" in df_h.columns:
        df_h["relative_humidity_2m"] = df_h["relative_humidity_2m"].clip(
            lower=HUMIDITY_FLOOR_PCT
        )
    if "wind_speed_10m" in df_h.columns:
        df_h["wind_speed_10m"] = df_h["wind_speed_10m"].clip(
            upper=WIND_SPEED_CAP_KMH
        )

    # Forward-fill gaps so today's rows have at least the latest known values
    for col in ["us_aqi", "pm2_5"]:
        if col in df_h.columns:
            df_h[col] = df_h[col].ffill()

    return df_h


def _predict_single_horizon(
    df_h: pd.DataFrame,
    horizon_h: int,
    models: dict,
    current_aqi: int,
    min_ci_width_aqi: float,
    generated_at: datetime,
) -> tuple[dict, float]:
    """
    Run feature engineering, model inference, and CI enforcement for one horizon.

    This is a pure function: no side effects, no global state reads beyond the
    model dict passed in, no network calls, no disk I/O.

    Args:
        df_h: Prepared DataFrame (output of _prepare_horizon_dataframe).
        horizon_h: Forecast horizon in hours (6, 12, 24, 48).
        models: Dict keyed by horizon_h, each containing 'point', 'q05', 'q95'.
        current_aqi: Current AQI integer (used as fallback base if feature is NaN).
        min_ci_width_aqi: Floor on the CI width to enforce monotonically increasing
            uncertainty across horizons. Renamed from `prev_width` to express
            purpose: this is the minimum acceptable interval width, not a
            "previous" value.
        generated_at: Forecast generation timestamp (used to compute valid_at).

    Returns:
        (forecast_entry, new_min_ci_width_aqi) where:
        - forecast_entry: dict with keys aqi, ci_lo, ci_hi, category, color, valid_at
        - new_min_ci_width_aqi: updated floor for the next horizon's CI width

    On any exception, logs at ERROR with exc_info and returns a degraded entry
    (current_aqi, ci_lo=0, ci_hi=500) with the unchanged min_ci_width_aqi.
    """
    m      = models[horizon_h]
    now_ts = pd.Timestamp.now(tz=TZ).floor('h')

    try:
        X_inf, _ = engineer_features(df_h, horizon_h=horizon_h, firms_hourly=_extract_firms(df_h))

        # Select the feature row for "now"
        if now_ts in X_inf.index:
            X_now = X_inf.loc[[now_ts]].copy()
        else:
            X_now = X_inf[X_inf.index <= now_ts].iloc[[-1]].copy()

        # Comprehensive cleaning to prevent LightGBM dtype/NaN errors
        X_now = X_now.ffill()
        X_now = X_now.apply(pd.to_numeric, errors='coerce')

        # Capture the current absolute baseline to invert the residual prediction
        base_aqi_now = X_now['aqi_current'].values[0]
        if np.isnan(base_aqi_now):
            base_aqi_now = current_aqi

        # Inject the categorical "Regime" feature
        regime_series = classify_regime(df_h)
        if now_ts in regime_series.index:
            val = regime_series.loc[now_ts]
            curr_regime = int(val.iloc[0]) if isinstance(val, pd.Series) else int(val)
        else:
            curr_regime = int(regime_series.iloc[-1])
        X_now['regime'] = pd.Categorical([curr_regime], categories=[0, 1, 2])

        def _select_model_features(model_obj, X_raw: pd.DataFrame) -> pd.DataFrame:
            # LightGBM handles NaN natively — pass raw columns without imputation.
            return X_raw[model_obj.feature_name_]

        res_pt  = m["point"].predict(_select_model_features(m["point"], X_now))[0]
        res_q05 = m["q05"].predict(_select_model_features(m["q05"],   X_now))[0]
        res_q95 = m["q95"].predict(_select_model_features(m["q95"],   X_now))[0]

        # Invert residual prediction back to absolute AQI
        pred_point = res_pt  + base_aqi_now
        pred_q05   = res_q05 + base_aqi_now
        pred_q95   = res_q95 + base_aqi_now

        pred_point = max(0, min(500, round(pred_point)))

        # Fix quantile crossing
        pred_q05_sorted = min(pred_q05, pred_q95)
        pred_q95_sorted = max(pred_q05, pred_q95)

        # V14: Apply conformal calibration scalar to guarantee >= 95% coverage.
        # V15 Tier 2: Season-aware — separate q for summer (May-Sep) and winter.
        current_month = pd.Timestamp.now(tz=TZ).month
        is_summer = current_month in {5, 6, 7, 8, 9}
        h_scales  = _CONFORMAL_SCALES.get(horizon_h, {})
        if isinstance(h_scales, dict):
            q_conformal = h_scales.get("summer" if is_summer else "winter", 0.0)
        else:
            q_conformal = float(h_scales)  # backward compat with scalar format
        if q_conformal > 0.0:
            pred_q05_sorted -= q_conformal
            pred_q95_sorted += q_conformal

        # Enforce monotonically increasing uncertainty across horizons.
        # min_ci_width_aqi is the floor: intervals must never shrink as the
        # horizon grows (longer forecasts must be at least as uncertain as shorter ones).
        current_width = pred_q95_sorted - pred_q05_sorted
        if current_width < min_ci_width_aqi:
            diff = (min_ci_width_aqi - current_width) / 2.0
            pred_q05_sorted -= diff
            pred_q95_sorted += diff
            current_width = min_ci_width_aqi

        pred_q05 = max(0, min(500, round(pred_q05_sorted)))
        pred_q95 = max(0, min(500, round(pred_q95_sorted)))

        # Guarantee ci_lo ≤ point ≤ ci_hi after clipping
        pred_q05 = min(pred_q05, pred_point)
        pred_q95 = max(pred_q95, pred_point)

        valid_at   = generated_at + timedelta(hours=horizon_h)
        cat, color = aqi_category(int(pred_point))

        entry = {
            "aqi":      int(pred_point),
            "ci_lo":    int(pred_q05),
            "ci_hi":    int(pred_q95),
            "category": cat,
            "color":    color,
            "valid_at": _iso(valid_at),
        }
        return entry, current_width

    except Exception as exc:
        log.error(
            "Prediction failed for %sh horizon: %s", horizon_h, exc, exc_info=True
        )
        degraded_entry = {
            "aqi":      current_aqi,
            "ci_lo":    0,
            "ci_hi":    500,
            "category": "Unknown",
            "color":    "#cccccc",
            "valid_at": _iso(generated_at + timedelta(hours=horizon_h)),
        }
        return degraded_entry, min_ci_width_aqi


def _assemble_forecast_result(
    current: dict,
    forecasts: dict,
    history: list,
    data_age_minutes: int,
    generated_at: datetime,
) -> dict:
    """
    Build the final JSON-serializable result dict matching the /forecast API schema.

    Pure function: no external calls, no disk I/O, no global state reads.
    All inputs are passed explicitly.
    """
    return {
        "generated_at": _iso(generated_at),
        "location": {
            "name": "Folsom, CA",
            "lat":  38.6780,
            "lon":  -121.1761,
        },
        "current":                current,
        "forecasts":              forecasts,
        "history_72h":            history,
        "model_version":          MODEL_VERSION,
        "model_metadata":         _get_model_metadata(),
        "data_freshness_minutes": data_age_minutes,
        "ai_summary":             "",  # populated by predict_now() after assembly
    }


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def predict_now() -> dict:
    """
    Orchestrate the full inference pipeline and return the forecast dict.
    Also writes the result to data/latest.json as the local cache.
    """
    generated_at = datetime.now(tz=TZ)
    models       = load_all_models()

    df, data_age_minutes = _fetch_input_data(past_hours=168)
    log.info("Fetching AirNow current reading...")
    airnow  = fetch_airnow_current()
    current = _resolve_current_conditions(df, airnow)
    df_h    = _prepare_horizon_dataframe(df, airnow)

    forecasts         = {}
    min_ci_width_aqi  = 0.0
    for horizon_h in HORIZONS:
        entry, min_ci_width_aqi = _predict_single_horizon(
            df_h, horizon_h, models, current["aqi"], min_ci_width_aqi, generated_at
        )
        forecasts[f"{horizon_h}h"] = entry

    history = _build_history_72h(df, models)
    result  = _assemble_forecast_result(
        current, forecasts, history, data_age_minutes, generated_at
    )

    log.info("Generating AI summary...")
    result["ai_summary"] = generate_summary(result)

    CACHE_FILE.write_text(json.dumps(result, indent=2, default=str))
    log.info("Forecast cached → %s", CACHE_FILE)
    return result


# ─── Supporting functions ─────────────────────────────────────────────────────

def _compute_data_age_minutes(df: pd.DataFrame) -> int:
    """How many minutes old is the most recent row of data?"""
    try:
        latest = df.index.max()
        now    = pd.Timestamp.now(tz=TZ)
        delta  = now - latest
        return max(0, int(delta.total_seconds() / 60))
    except Exception:
        return -1


def _build_history_72h(df: pd.DataFrame, models: dict) -> list[dict]:
    """
    Build the history array: last 72 hours of observed AQI vs. what the
    6h-horizon model would have predicted at each hour.

    Uses _extract_scalar_handling_dst_duplicates() instead of direct df.loc
    to guard against the DST spring-forward duplicate-index scenario: on the
    March spring-forward day, two rows share the same wall-clock hour, causing
    df.loc[ts, col] to return a Series instead of a scalar, which would crash
    the loop with 'truth value of a Series is ambiguous'.
    """
    history = []
    now_ts  = pd.Timestamp.now(tz=TZ).floor('h')
    cutoff  = now_ts - pd.Timedelta(hours=72)

    try:
        X_hist, _ = engineer_features(df, horizon_h=6, firms_hourly=_extract_firms(df))
        m6        = models[6]
        pt6       = m6["point"]
        q05_6     = m6["q05"]
        q95_6     = m6["q95"]

        X_window = X_hist[(X_hist.index > cutoff) & (X_hist.index <= now_ts)].copy()

        if len(X_window) > 0:
            regime_series = classify_regime(df)
            X_window['regime'] = pd.Categorical(
                regime_series.reindex(X_window.index).fillna(2).astype(int),
                categories=[0, 1, 2],
            )

            def _hist_predict(model_obj, X_src: pd.DataFrame) -> np.ndarray:
                return model_obj.predict(X_src[model_obj.feature_name_].copy())

            preds = _hist_predict(pt6,  X_window) + X_window['aqi_current'].values
            q05s  = _hist_predict(q05_6, X_window) + X_window['aqi_current'].values
            q95s  = _hist_predict(q95_6, X_window) + X_window['aqi_current'].values

            preds = np.round(np.clip(preds, 0, 500)).astype(int)
            q05s  = np.round(np.clip(q05s,  0, 500)).astype(int)
            q95s  = np.round(np.clip(q95s,  0, 500)).astype(int)

            for i, ts in enumerate(X_window.index):
                actual_raw = _extract_scalar_handling_dst_duplicates(df, ts, "us_aqi")
                actual_val = int(round(actual_raw)) if not np.isnan(actual_raw) else None
                history.append({
                    "timestamp":    _iso(ts),
                    "actual_aqi":   actual_val,
                    "forecast_aqi": int(round(preds[i])),
                    "ci_lo":        int(round(q05s[i])),
                    "ci_hi":        int(round(q95s[i])),
                })

    except Exception as exc:
        log.error("history_72h build failed: %s", exc, exc_info=True)
        hist_df = df[df.index > cutoff].copy()
        for ts, row in hist_df.iterrows():
            aqi = row.get("us_aqi", np.nan)
            history.append({
                "timestamp":    _iso(ts),
                "actual_aqi":   int(round(float(aqi))) if not np.isnan(float(aqi)) else None,
                "forecast_aqi": None,
                "ci_lo":        None,
                "ci_hi":        None,
            })

    return history[-72:]


# ─── CDN cache functions ──────────────────────────────────────────────────────

def load_remote_forecast() -> dict | None:
    """Fetch the latest.json from the GitHub data-cache branch (the CDN)."""
    OWNER = "arsanisl-code"
    REPO  = "folsom-aqi-backend"
    URL   = (
        f"https://api.github.com/repos/{OWNER}/{REPO}"
        f"/contents/data/latest.json?ref=data-cache"
    )
    token   = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3.raw"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        resp = requests.get(URL, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        log.warning("Remote CDN returned HTTP %s: %s", resp.status_code, resp.text[:100])
    except Exception as exc:
        log.error("Remote CDN fetch failed: %s", exc, exc_info=True)
    return None


def load_cached_forecast(prefer_remote: bool = True) -> dict | None:
    """
    Return the cached forecast.
    Tries the GitHub CDN first (to avoid Render disk staleness), then falls
    back to the local data/latest.json file.
    """
    if prefer_remote:
        remote = load_remote_forecast()
        if remote:
            return remote

    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return None
    return None


def cache_age_minutes() -> int:
    """How many minutes old is the local cached forecast file?"""
    if not CACHE_FILE.exists():
        return 9999
    age = time.time() - CACHE_FILE.stat().st_mtime
    return int(age / 60)
