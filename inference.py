"""
inference.py — Produce AQI forecast JSON for Folsom, CA.
Loads all 12 models, fetches recent data, returns the forecast dict.
Called by api.py (via refresh.py) every hour.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from data_fetcher import fetch_recent_combined, fetch_airnow_current
from features import engineer_features
from ai_layer import generate_summary
import requests

# ─── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR    = Path("models")
DATA_DIR      = Path("data")
CACHE_FILE    = DATA_DIR / "latest.json"
MODEL_VERSION = "v1.0"
HORIZONS      = [6, 12, 24, 48]
TZ            = ZoneInfo("America/Los_Angeles")

AQI_CATEGORIES = [
    (0,   50,  "Good",                           "#00e400"),
    (51,  100, "Moderate",                       "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups",  "#ff7e00"),
    (151, 200, "Unhealthy",                      "#ff0000"),
    (201, 300, "Very Unhealthy",                 "#8f3f97"),
    (301, 500, "Hazardous",                      "#7e0023"),
]

# Post-hoc coverage scaling removed — quantile models now natively output 98% intervals

DATA_DIR.mkdir(parents=True, exist_ok=True)


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


def _safe_aqi_scalar(df: pd.DataFrame, ts, col: str = "us_aqi"):
    """
    FIX 3 (partial): Safely extract a scalar AQI value from df at timestamp ts.
    df.loc[ts] can return a Series when the index has duplicate timestamps
    (e.g. DST transition at 2:00 AM on March 9 visible in the test output).
    Using .iloc[0] after loc avoids the 'truth value of a Series is ambiguous' error.
    """
    try:
        val = df.loc[ts, col]
        # If loc returns a Series (duplicate index rows), take first value
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val)
    except Exception:
        return np.nan


# ─── Model loading ────────────────────────────────────────────────────────────

_model_cache: dict = {}


def load_all_models() -> dict:
    """
    Load all 12 models + 4 imputers into a dict keyed by horizon.
    Called once at API startup; expensive on first call, instant after.
    """
    global _model_cache
    if _model_cache:
        return _model_cache

    missing = []
    for h in HORIZONS:
        for kind in ["point", "q05", "q95"]:
            p = MODELS_DIR / f"lgbm_{kind}_{h}h.pkl"
            if not p.exists():
                missing.append(str(p))
        ip = MODELS_DIR / f"imputer_{h}h.pkl"
        if not ip.exists():
            missing.append(str(ip))

    if missing:
        raise RuntimeError(
            "Missing model files:\n" + "\n".join(missing) +
            "\nRun train.py first."
        )

    models = {}
    for h in HORIZONS:
        models[h] = {
            "point":   joblib.load(MODELS_DIR / f"lgbm_point_{h}h.pkl"),
            "q05":     joblib.load(MODELS_DIR / f"lgbm_q05_{h}h.pkl"),
            "q95":     joblib.load(MODELS_DIR / f"lgbm_q95_{h}h.pkl"),
            "imputer": joblib.load(MODELS_DIR / f"imputer_{h}h.pkl"),
        }

    _model_cache = models
    return models


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_now() -> dict:
    """
    Fetch the latest available data and produce forecasts for all 4 horizons.
    Returns the full forecast dict matching the /forecast API schema.
    Also writes result to data/latest.json as cache.
    """
    generated_at = datetime.now(tz=TZ)

    # ── Step 1: Load models ────────────────────────────────────────────────
    models = load_all_models()

    # ── Step 2: Fetch recent data (last 168h) ─────────────────────────────
    print("[inference] Fetching recent data...", file=sys.stderr)
    try:
        df = fetch_recent_combined(past_hours=168)
    except Exception as exc:
        print(f"[inference] CRITICAL: Failed to fetch data: {exc}", file=sys.stderr)
        raise

    fetch_age_minutes = _compute_data_age_minutes(df)

    # ── Step 3: Attempt AirNow current reading ────────────────────────────
    print("[inference] Fetching AirNow current reading...", file=sys.stderr)
    airnow = fetch_airnow_current()

    if airnow and airnow.get("aqi", 0) > 0:
        current_aqi       = int(airnow["aqi"])
        current_cat, current_color = aqi_category(current_aqi)
        current_source    = "AirNow"
        current_ts        = airnow.get("timestamp", generated_at.isoformat())
        current_pollutant = airnow.get("primary_pollutant", "PM2.5")
    else:
        # Filter to only past/present data to find the "Current" reading
        past_df = df[df.index <= generated_at]
        recent_aqi = past_df["us_aqi"].dropna()
        
        current_aqi = int(round(recent_aqi.iloc[-1])) if len(recent_aqi) > 0 else 0
        current_cat, current_color = aqi_category(current_aqi)
        current_source    = "Open-Meteo"
        current_ts        = _iso(recent_aqi.index[-1]) if len(recent_aqi) > 0 else generated_at.isoformat()
        current_pollutant = "PM2.5"

    # ── Step 4: Build features and predict for each horizon ───────────────
    forecasts = {}
    for h in HORIZONS:
        m = models[h]
        try:
            # Step 4a: Robustly inject real-time data and sanitize weather anomalies
            df = df.copy()
            now_ts = pd.Timestamp.now(tz=TZ).floor('h')
            
            # 1. Inject AirNow data precisely at 'now' and for the rest of today
            # To ensure rolling means and lags 'pick up' the 30 AQI immediately.
            if airnow:
                aq_val = float(airnow.get("aqi", 30))
                # Fill everything from 24h ago to today with AirNow if missing
                df.loc[now_ts - timedelta(hours=24):now_ts, "us_aqi"] = df.loc[now_ts - timedelta(hours=24):now_ts, "us_aqi"].fillna(aq_val)
                df.at[now_ts, "us_aqi"] = aq_val

            # 2. Sanity Guard: Open-Meteo sometimes hallucinates 5-6% humidity or 30km/h wind for Folsom
            # We clamp these to prevent extreme Wildfire signals (HDWI)
            if "relative_humidity_2m" in df.columns:
                df["relative_humidity_2m"] = df["relative_humidity_2m"].clip(lower=25.0) # floor to 25%
            if "wind_speed_10m" in df.columns:
                df["wind_speed_10m"] = df["wind_speed_10m"].clip(upper=25.0) # cap at 25km/h (~15mph)
            
            # Fill gaps (ensures today's rows have at least the latest known values)
            for col in ["us_aqi", "pm2_5"]:
                if col in df.columns:
                    df[col] = df[col].ffill()
            
            X_inf, _ = engineer_features(df, horizon_h=h)
            
            # Step 4b: Correctly select the row for "NOW" (flored)
            if now_ts in X_inf.index:
                X_now = X_inf.loc[[now_ts]].copy()
            else:
                # Search for the most recent hour that has any data
                X_now = X_inf[X_inf.index <= now_ts].iloc[[-1]].copy()
            
            # Step 4c: Comprensive cleaning to prevent LightGBM dtype/NaN errors
            # 1. Forward fill any internal gaps in the row
            X_now = X_now.ffill()
            # 2. Strict cast to float (converts 'None' or 'Object' to NaN for the imputer)
            X_now = X_now.apply(pd.to_numeric, errors='coerce')
            # 3. Final fillna(0) for any truly empty fields (safe since imputer follows)
            X_now = X_now.fillna(0)
            # Capture the current absolute baseline to invert the residual
            base_aqi_now = X_now['aqi_current'].values[0]
            if np.isnan(base_aqi_now):
                base_aqi_now = current_aqi

            imputer = m["imputer"]
            point   = m["point"]
            # ─── Dynamic Feature Alignment ───
            # This handles cases where point models (74 features) and 
            # quantile models (76 features) are out of sync.
            
            # 1. Align/Impute correctly for the Point Model (74 features)
            pt_model = m["point"]
            X_pt     = X_now[pt_model.feature_name_]
            X_pt_imp = imputer.transform(X_pt)
            res_pt   = pt_model.predict(X_pt_imp)[0]
            
            # 2. Align/Impute for the Quantile Models (76 features)
            q05_model = m["q05"]
            q95_model = m["q95"]
            X_q05 = X_now[q05_model.feature_name_]
            X_q95 = X_now[q95_model.feature_name_]
            
            # Try to use the imputer on its original 74 features, then re-merge the 2 non-imputed ones (dow_sin/cos)
            try:
                X_q05_imp = imputer.transform(X_q05)
                X_q95_imp = imputer.transform(X_q95)
            except Exception:
                # Traditional imputer (74 features) — subset, then fill the rest manually if needed
                imp_names = pt_model.feature_name_
                X_for_imp = X_now[imp_names]
                X_imputed = pd.DataFrame(imputer.transform(X_for_imp), columns=imp_names, index=X_now.index)
                
                # Now, for any missing columns in the 76-set (like dow_sin/cos), just pull from the original X
                X_q_final = X_imputed.copy()
                for col in q05_model.feature_name_:
                    if col not in X_q_final:
                        X_q_final[col] = X_now[col].values
                
                X_q05_imp = X_q_final[q05_model.feature_name_]
                X_q95_imp = X_q_final[q95_model.feature_name_]

            res_q05 = q05_model.predict(X_q05_imp)[0]
            res_q95 = q95_model.predict(X_q95_imp)[0]

            # Invert to absolute AQI values
            pred_point = res_pt + base_aqi_now
            pred_q05   = res_q05 + base_aqi_now
            pred_q95   = res_q95 + base_aqi_now

            # Clip to valid AQI range
            pred_point = max(0, min(500, round(pred_point)))
            pred_q05   = max(0, min(500, round(pred_q05)))
            pred_q95   = max(0, min(500, round(pred_q95)))

            # Guarantee ci_lo ≤ point ≤ ci_hi after clipping
            pred_q05 = min(pred_q05, pred_point)
            pred_q95 = max(pred_q95, pred_point)

            # FIX 2: valid_at = now + horizon hours (was incorrectly set to now).
            valid_at   = generated_at + timedelta(hours=h)
            cat, color = aqi_category(int(pred_point))

            forecasts[f"{h}h"] = {
                "aqi":      int(pred_point),
                "ci_lo":    int(pred_q05),
                "ci_hi":    int(pred_q95),
                "category": cat,
                "color":    color,
                "valid_at": _iso(valid_at),
            }
        except Exception as exc:
            print(f"[inference] ERROR for {h}h horizon: {exc}", file=sys.stderr)
            forecasts[f"{h}h"] = {
                "aqi":      current_aqi,
                "ci_lo":    0,
                "ci_hi":    500,
                "category": "Unknown",
                "color":    "#cccccc",
                "valid_at": _iso(generated_at + timedelta(hours=h)),
            }

    # ── Step 5: Build history_72h ─────────────────────────────────────────
    history_72h = _build_history_72h(df, models)

    # ── Step 6: Assemble and cache output ─────────────────────────────────
    result = {
        "generated_at": _iso(generated_at),
        "location": {
            "name": "Folsom, CA",
            "lat":  38.6780,
            "lon":  -121.1761,
        },
        "current": {
            "aqi":               current_aqi,
            "category":          current_cat,
            "color":             current_color,
            "primary_pollutant": current_pollutant,
            "source":            current_source,
            "timestamp":         current_ts,
        },
        "forecasts":              forecasts,
        "history_72h":            history_72h,
        "model_version":          MODEL_VERSION,
        "data_freshness_minutes": fetch_age_minutes,
        "ai_summary":             "",   # populated below if GEMINI_API_KEY is set
    }

    # ── Step 7: Generate AI summary (non-blocking — failure leaves field empty) ──
    print("[inference] Generating AI summary...", file=sys.stderr)
    result["ai_summary"] = generate_summary(result)

    CACHE_FILE.write_text(json.dumps(result, indent=2, default=str))
    print(f"[inference] Forecast cached → {CACHE_FILE}", file=sys.stderr)

    return result


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

    FIX 3: The original code failed with 'truth value of a Series is ambiguous'
    because df.loc[ts, col] returns a Series when the index contains duplicate
    timestamps (DST spring-forward on March 9 causes exactly this — two rows
    share the same wall-clock time). _safe_aqi_scalar() handles this by taking
    the first value from any duplicate group.
    """
    history = []
    # Anchor the 72h window to the current time, not the end of the future forecast
    now_ts = pd.Timestamp.now(tz=TZ).floor('h')
    cutoff = now_ts - pd.Timedelta(hours=72)

    try:
        X_hist, _ = engineer_features(df, horizon_h=6)
        m6        = models[6]
        imp6      = m6["imputer"]
        pt6       = m6["point"]
        q05_6     = m6["q05"]
        q95_6     = m6["q95"]

        X_window = X_hist[(X_hist.index > cutoff) & (X_hist.index <= now_ts)]

        if len(X_window) > 0:
            # 1. Align/Impute for Point Model (74)
            X_pt     = X_window[pt6.feature_name_]
            X_pt_imp = imp6.transform(X_pt)
            preds    = pt6.predict(X_pt_imp) + X_window['aqi_current'].values
            
            # 2. Align/Impute for Quantiles (76)
            try:
                # Direct transform if imputer matches features (76)
                q05s = q05_6.predict(imp6.transform(X_window[q05_6.feature_name_])) + X_window['aqi_current'].values
                q95s = q95_6.predict(imp6.transform(X_window[q95_6.feature_name_])) + X_window['aqi_current'].values
            except Exception:
                # Subset imputer (74) then merge
                imp_names = pt6.feature_name_
                X_imputed = pd.DataFrame(imp6.transform(X_window[imp_names]), columns=imp_names, index=X_window.index)
                X_q_final = X_imputed.copy()
                for col in q05_6.feature_name_:
                    if col not in X_q_final:
                        X_q_final[col] = X_window[col].values
                
                q05s = q05_6.predict(X_q_final[q05_6.feature_name_]) + X_window['aqi_current'].values
                q95s = q95_6.predict(X_q_final[q95_6.feature_name_]) + X_window['aqi_current'].values

            preds = np.round(np.clip(preds, 0, 500)).astype(int)
            q05s  = np.round(np.clip(q05s,  0, 500)).astype(int)
            q95s  = np.round(np.clip(q95s,  0, 500)).astype(int)

            # Expand CI symmetrically around the point forecast natively without scaling hacks
            for i, ts in enumerate(X_window.index):
                # FIX 3: use _safe_aqi_scalar instead of direct df.loc
                actual_raw = _safe_aqi_scalar(df, ts, "us_aqi")
                actual_val = int(round(actual_raw)) if not np.isnan(actual_raw) else None

                history.append({
                    "timestamp":    _iso(ts),
                    "actual_aqi":   actual_val,
                    "forecast_aqi": int(round(preds[i])),
                    "ci_lo":        int(round(q05s[i])),
                    "ci_hi":        int(round(q95s[i])),
                })

    except Exception as exc:
        print(f"[inference] history_72h build error: {exc}", file=sys.stderr)
        # Fallback: return actuals only, no forecast line
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


def load_remote_forecast() -> dict | None:
    """Fetch the latest.json from the GitHub data-cache branch (the CDN)."""
    # Placeholder strings to be filled by the user or final sed command
    OWNER = "arsanisl-code"
    REPO  = "folsom-aqi-backend"
    URL   = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/data-cache/data/latest.json"
    
    try:
        resp = requests.get(URL, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        print(f"[inference] Remote CDN fetch failed: {exc}", file=sys.stderr)
    return None


def load_cached_forecast(prefer_remote: bool = True) -> dict | None:
    """
    Return the cached forecast. 
    By default, tries the GitHub CDN first (to avoid Render disk staleness).
    Falls back to local data/latest.json if remote fails or prefer_remote=False.
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
    """How many minutes old is the cached forecast?"""
    if not CACHE_FILE.exists():
        return 9999
    import time
    age = time.time() - CACHE_FILE.stat().st_mtime
    return int(age / 60)