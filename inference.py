"""
inference.py — Produce AQI forecast JSON for Folsom, CA.
Loads all models, fetches recent data, returns the forecast dict.
Called by refresh.py every hour via GitHub Actions.

predict_now() is a thin orchestration shell. All business logic lives in
the named sub-functions below, each with a single responsibility.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from ai_layer import generate_summary
from config import DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON, DEFAULT_STATION
from data_fetcher import fetch_airnow_current, fetch_recent_combined
from features import engineer_features
from logger import get_logger

log = get_logger(__name__)


class _NNLSMeta:
    """
    NNLS meta-learner with bias column and normalized weights.
    Redefined here so joblib can unpickle the ensemble models.
    """

    def __init__(self):
        self.coef_: np.ndarray = np.array([])
        self.bias_: float = 0.0
        self._raw_w: np.ndarray = np.array([])

    def predict(self, X_oof: np.ndarray) -> np.ndarray:
        return self.predict_normalized(X_oof)

    def predict_normalized(self, X_oof: np.ndarray) -> np.ndarray:
        return X_oof @ self.coef_ + self.bias_

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_VERSION = "Folsom-AQI-Navigator-V12"
CACHE_FILE = Path("data/latest.json")

# Open-Meteo hallucination floor for relative humidity (Folsom grid cell artifact).
HUMIDITY_FLOOR_PCT: float = 25.0

# Open-Meteo hallucination cap for wind speed (Folsom grid cell artifact).
WIND_SPEED_CAP_KMH: float = 25.0

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
HORIZONS = [6, 12, 24, 48]
TZ = ZoneInfo("America/Los_Angeles")

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Metadata (loaded once at import) ────────────────────────────────
_METRICS: dict = {}
_FEATURES: dict = {}  # keyed by horizon_h
_CONFORMAL_SCALES: dict[int, float] = {}  # keyed by horizon_h; 0.0 = no correction
try:
    _m = Path("models/training_metrics.json")
    if not _m.exists():
         _m = Path("models/tournament_report.json") # Fallback to ensemble report
    if _m.exists():
        _METRICS = json.loads(_m.read_text())

    for _h in [6, 12, 24, 48]:
        _f = Path(f"models/feature_names_{_h}h.json")
        if not _f.exists():
             _f = Path("models/feature_names.json")
        if _f.exists():
            _FEATURES[_h] = json.loads(_f.read_text())
    # Fallback to legacy single file if per-horizon files missing
    if not _FEATURES:
        _f = Path("models/feature_names.json")
        if _f.exists():
            names = json.loads(_f.read_text())
            _FEATURES = dict.fromkeys([6, 12, 24, 48], names)

    _cs = Path("models/conformal_scales.json")
    if _cs.exists():
        _raw = json.loads(_cs.read_text())
        _CONFORMAL_SCALES = {int(k): float(v) for k, v in _raw.get("scales", {}).items()}

    _CONFORMAL_MARGINS = {}
    _cm = Path("models/conformal_margins.json")
    if _cm.exists():
        _CONFORMAL_MARGINS = {int(k): float(v) for k, v in json.loads(_cm.read_text()).items()}

except Exception as exc:
    pass  # Graceful degradation if files aren't available yet


# ─── AQI helpers ──────────────────────────────────────────────────────────────


def aqi_category(aqi: int) -> tuple[str, str]:
    """Return (category_name, color_hex) for a given AQI integer."""
    for lo, hi, name, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return name, color
    return "Hazardous", "#7e0023"


def load_all_models() -> dict:
    """
    Load all trained models into memory. Returns a nested dict:
    {
      6:  { 'point': model, 'q05': model, 'q95': model, 'imputer': imp, 'meta': meta, 'phys_cols': [] },
      12: { ... },
      ...
    }
    """
    all_models = {}
    for h in HORIZONS:
        h_models = {}
        try:
            # Check for Ensemble components first (Latest)
            meta_path = MODELS_DIR / f"meta_learner_{h}h.pkl"
            if meta_path.exists():
                h_models['meta'] = joblib.load(meta_path)
                h_models['lgbm_full'] = joblib.load(MODELS_DIR / f"lgbm_full_{h}h.pkl")
                h_models['xgb'] = joblib.load(MODELS_DIR / f"xgb_{h}h.pkl")
                h_models['lgbm_physics'] = joblib.load(MODELS_DIR / f"lgbm_physics_{h}h.pkl")
                h_models['imputer'] = joblib.load(MODELS_DIR / f"imputer_{h}h.pkl")

                # Load physics feature list
                phys_path = MODELS_DIR / f"physics_cols_{h}h.json"
                if phys_path.exists():
                    h_models['phys_cols'] = json.loads(phys_path.read_text())

                all_models[h] = h_models
                continue

            # Fallback to Point models (Legacy)
            p_path = MODELS_DIR / f"lgbm_point_{h}h.pkl"
            if p_path.exists():
                h_models["point"] = joblib.load(p_path)
                h_models["q05"] = joblib.load(MODELS_DIR / f"lgbm_q05_{h}h.pkl")
                h_models["q95"] = joblib.load(MODELS_DIR / f"lgbm_q95_{h}h.pkl")
                h_models["imputer"] = joblib.load(MODELS_DIR / f"imputer_{h}h.pkl")
                all_models[h] = h_models
        except Exception as e:
            log.warning(f"Failed to load models for {h}h: {e}")
    return all_models


# ─── History helpers ──────────────────────────────────────────────────────────


def _safe_aqi_scalar(df: pd.DataFrame, ts, col: str) -> float:
    """
    Safely extract a scalar from df at index ts for column col.
    Handles DST spring-forward duplicates by taking the first value.
    Returns np.nan if not found.
    """
    try:
        val = df.loc[ts, col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val)
    except (KeyError, IndexError):
        return np.nan


def _build_history_72h(df: pd.DataFrame, models: dict) -> list[dict]:
    """
    Build the last 72 hours of observed AQI vs. what the 6, 12, 24, 48h ensembles
    would have predicted (retrodiction).
    """
    now_ts = pd.Timestamp.now(tz=TZ).floor("h")
    cutoff = now_ts - pd.Timedelta(hours=72)
    history_map: dict[pd.Timestamp, dict] = {}

    # Initialize with actuals
    hist_df = df[(df.index > cutoff) & (df.index <= now_ts)].copy()
    for ts, row in hist_df.iterrows():
        aqi = row.get("us_aqi", np.nan)
        history_map[ts] = {
            "timestamp": ts.isoformat(),
            "actual_aqi": int(round(float(aqi))) if not np.isnan(float(aqi)) else None
        }

    # Add forecasts for each horizon
    for h in [6, 12, 24, 48]:
        try:
            m = models.get(h)
            feat_list = _FEATURES.get(h)
            if not m or not feat_list:
                continue

            X_h, _ = engineer_features(df, horizon_h=h)
            X_window = X_h[(X_h.index > (cutoff - pd.Timedelta(hours=h))) & (X_h.index <= now_ts)].copy()
            if len(X_window) == 0:
                continue

            X_aligned = X_window.reindex(columns=feat_list, fill_value=0)
            X_imp = pd.DataFrame(m["imputer"].transform(X_aligned), columns=feat_list, index=X_window.index)
            base_aqi = X_window["aqi_current"].values if "aqi_current" in X_window.columns else np.zeros(len(X_window))

            if "meta" in m:
                p_full = m["lgbm_full"].predict(X_imp)
                p_xgb = m["xgb"].predict(X_imp)
                phys_cols = m.get("phys_cols", [])
                p_phys = m["lgbm_physics"].predict(X_imp[[c for c in phys_cols if c in X_imp.columns]]) if phys_cols else p_full

                # Blend
                abs_full = np.clip(p_full + base_aqi, 0, 500)
                abs_xgb = np.clip(p_xgb + base_aqi, 0, 500)
                abs_phys = np.clip(p_phys + base_aqi, 0, 500)

                preds = np.zeros(len(X_window))
                for i in range(len(X_window)):
                    res_in = np.array([[abs_full[i] - base_aqi[i], abs_xgb[i] - base_aqi[i], abs_phys[i] - base_aqi[i]]])
                    preds[i] = base_aqi[i] + m["meta"].predict(res_in)[0]
            else:
                preds = m["point"].predict(X_imp) + base_aqi

            preds = np.round(np.clip(preds, 0, 500)).astype(int)
            for i, ts in enumerate(X_window.index):
                target_ts = ts + pd.Timedelta(hours=h)
                if target_ts in history_map:
                    history_map[target_ts][f"forecast_{h}h"] = int(preds[i])
                    # For legacy compatibility, also set the main forecast_aqi to the 6h prediction
                    if h == 6:
                        history_map[target_ts]["forecast_aqi"] = int(preds[i])
                        # Simple 10% margin for legacy CI
                        margin = max(5, int(preds[i] * 0.1))
                        history_map[target_ts]["ci_lo"] = int(preds[i] - margin)
                        history_map[target_ts]["ci_hi"] = int(preds[i] + margin)

        except Exception as e:
            log.warning("Retrospective %sh failed: %s", h, e)

    # Sort and convert to list
    sorted_ts = sorted(history_map.keys())
    return [history_map[ts] for ts in sorted_ts]


# ─── Prediction Core ──────────────────────────────────────────────────────────


def _get_model_metadata() -> dict:
    """Load model accuracy metrics from the training report."""
    try:
        report_path = Path("models/tournament_report.json")
        if report_path.exists():
            report = json.loads(report_path.read_text())
            horizons = []
            for h_data in report.get("horizons", []):
                h = h_data.get("horizon_h")
                # Use ensemble metrics if available, else lgbm_full
                metrics = h_data.get("ensemble_", h_data.get("lgbm_full_", {}))
                horizons.append({
                    "horizon_h": h,
                    "val_mae": metrics.get("mae"),
                    "val_r2": metrics.get("r2")
                })
            return {
                "architecture": report.get("architecture"),
                "horizons": horizons
            }
    except Exception:
        pass
    return {}


def predict_now() -> dict:
    """
    Main orchestration function.
    1. Fetches current data from APIs.
    2. Builds features.
    3. Runs models for 6, 12, 24, 48h.
    4. Applies conformal calibration.
    5. Generates AI summary.
    6. Returns structured JSON.
    """
    start_time = time.time()
    log.info("Starting inference pipeline...")

    # 1. Fetch current data
    df = fetch_recent_combined(forecast_days=5)
    if df.empty:
        log.error("Failed to fetch data. Aborting.")
        return {"error": "Data fetch failed"}

    # 2. Extract current state (use last non-nan for current AQI if available)
    aqi_series = df["us_aqi"].ffill()
    if aqi_series.isna().all():
        log.error("No valid AQI data in dataframe.")
        return {"error": "No AQI data"}

    current_aqi = int(round(aqi_series.iloc[-1]))
    current_cat, current_color = aqi_category(current_aqi)
    primary_pollutant = "PM2.5"  # Default for Open-Meteo Air Quality API

    # AirNow check for current station (Ground Truth)
    station_info = fetch_airnow_current()
    if station_info and "aqi" in station_info:
        current_aqi = station_info["aqi"]
        current_cat, current_color = aqi_category(current_aqi)
        primary_pollutant = station_info.get("primary_pollutant", "PM2.5")
        source_name = f"AirNow ({station_info.get('station', DEFAULT_STATION)})"
    else:
        source_name = "Open-Meteo (Interpolated)"

    # 3. Build Forecasts
    models = load_all_models()
    forecasts = {}

    for h in HORIZONS:
        if h not in models:
            continue

        try:
            # Build features for this horizon
            X_full, _ = engineer_features(df, h)
            X_recent = X_full.tail(1)

            # Align features with training
            feat_list = _FEATURES.get(h)
            if not feat_list:
                log.warning(f"No feature list found for {h}h. Skipping.")
                continue

            X_final = X_recent.reindex(columns=feat_list, fill_value=0)

            # Impute
            imputed_arr = models[h]["imputer"].transform(X_final)
            X_imputed = pd.DataFrame(imputed_arr, columns=feat_list)

            # Predict
            if 'meta' in models[h]:
                # Ensemble Prediction
                p_full = models[h]['lgbm_full'].predict(X_imputed)[0]
                p_xgb = models[h]['xgb'].predict(X_imputed)[0]

                # Physics branch
                phys_cols = models[h].get('phys_cols', [])
                if phys_cols:
                    X_phys = X_imputed[phys_cols]
                    p_phys = models[h]['lgbm_physics'].predict(X_phys)[0]
                else:
                    p_phys = p_full # fallback

                # Blend with Meta-learner
                # Models output raw residuals. Clamp them to absolute space (0-500)
                # then subtract current_aqi to get clamped residuals for the meta-learner
                pred_lgbm_full = max(0, min(500, p_full + current_aqi))
                pred_xgb = max(0, min(500, p_xgb + current_aqi))
                pred_lgbm_physics = max(0, min(500, p_phys + current_aqi))

                res_full = pred_lgbm_full - current_aqi
                res_xgb = pred_xgb - current_aqi
                res_phys = pred_lgbm_physics - current_aqi

                meta_input = np.array([[res_full, res_xgb, res_phys]])
                residual = models[h]['meta'].predict(meta_input)[0]
                point_pred = current_aqi + residual

                # ── Data-Driven Split Conformal Prediction Intervals ──
                # Use empirical 90th percentile of absolute validation residuals
                margin = _CONFORMAL_MARGINS.get(h, max(5.0, point_pred * 0.15))
                q05_abs = point_pred - margin
                q95_abs = point_pred + margin

            else:
                # Standard Point Prediction
                point_res = models[h]["point"].predict(X_imputed)[0]
                point_pred = current_aqi + point_res
                q05_abs = current_aqi + models[h]["q05"].predict(X_imputed)[0]
                q95_abs = current_aqi + models[h]["q95"].predict(X_imputed)[0]

            # 4. Conformal Calibration (Season-Aware)
            # scale = scale_factor from conformal_scales.json
            scale = _CONFORMAL_SCALES.get(h, 1.0)

            # Center the interval on point prediction
            half_width = abs(q95_abs - q05_abs) / 2.0
            calibrated_half_width = half_width * scale

            ci_lo = int(round(np.clip(point_pred - calibrated_half_width, 0, 500)))
            ci_hi = int(round(np.clip(point_pred + calibrated_half_width, 0, 500)))
            point_final = int(round(np.clip(point_pred, 0, 500)))

            cat, color = aqi_category(point_final)
            from datetime import timedelta
            valid_at_ts = datetime.now(TZ) + timedelta(hours=h)

            forecasts[f"{h}h"] = {
                "aqi": point_final,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "category": cat,
                "color": color,
                "valid_at": valid_at_ts.isoformat(),
            }
        except Exception as e:
            log.error(f"Error predicting {h}h: {e}")

    # 5. History (Last 72h) — actual vs forecast for the dashboard chart
    history = _build_history_72h(df, models)

    # 6. AI Summary - populated below after dict assembly

    # Calculate data freshness
    data_age = 0
    try:
        # Filter to only include observed data (past/present) to avoid forecast timestamps from Open-Meteo
        now_dt = datetime.now(TZ)
        observed = df[df.index <= now_dt]
        if not observed.empty:
            latest_data_ts = observed[observed["us_aqi"].notna()].index.max()
            data_age = int((now_dt - latest_data_ts).total_seconds() / 60)
    except Exception:
        pass

    # 7. Final Payload
    output = {
        "generated_at": datetime.now(TZ).isoformat(),
        "data_freshness_minutes": data_age,
        "location": {
            "name": DEFAULT_CITY,
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON
        },
        "current": {
            "aqi": current_aqi,
            "category": current_cat,
            "color": current_color,
            "primary_pollutant": primary_pollutant,
            "source": source_name,
            "timestamp": datetime.now(TZ).isoformat(),
        },
        "forecasts": forecasts,
        "history_72h": history,
        "ai_summary": "",
        "model_version": MODEL_VERSION,
        "model_metadata": _get_model_metadata()
    }

    # 6. Generate AI summary
    output["ai_summary"] = generate_summary(output)

    # Save to cache
    CACHE_FILE.write_text(json.dumps(output, indent=2, default=str))

    elapsed = time.time() - start_time
    log.info(f"Inference complete in {elapsed:.2f}s")
    return output


def load_cached_forecast(prefer_remote: bool = False) -> dict:
    """
    Return the cached forecast JSON.
    If prefer_remote=True, attempts to fetch from the GitHub CDN (data-cache branch).
    This is used by the Render API to bypass local rate limits.
    """
    if prefer_remote:
        # Folsom AQI Navigator CDN URL (data-cache branch)
        # We append a cache-busting timestamp to bypass Fastly's aggressive caching of raw.githubusercontent.com
        url = f"https://raw.githubusercontent.com/arsanisl-code/folsom-aqi-backend/data-cache/latest.json?t={int(time.time())}"
        try:
            headers = {}
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
            
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            else:
                log.warning(f"Remote cache returned HTTP {resp.status_code}")
        except Exception as e:
            log.warning(f"Failed to fetch remote cache: {e}")
            # Fall through to local cache

    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def cache_age_minutes() -> int:
    """Return age of the cache file in minutes."""
    if not CACHE_FILE.exists():
        return 999
    mtime = CACHE_FILE.stat().st_mtime
    return int((time.time() - mtime) / 60)
