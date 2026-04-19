"""
data_fetcher.py — Fetch AQI and weather data from Open-Meteo and AirNow.
All fetches use retry with exponential backoff and local parquet cache.
Returns pd.DataFrame with DatetimeIndex in America/Los_Angeles.
Never crashes — returns last valid cache on any failure.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import urllib.parse
import io
from dotenv import load_dotenv

from logger import get_logger

log = get_logger(__name__)

load_dotenv()  # Load .env for AIRNOW_API_KEY

# ─── Constants ────────────────────────────────────────────────────────────────

LAT = 38.6780
LON = -121.1761
TZ  = "America/Los_Angeles"

CACHE_DIR = Path("data/cache")
CACHE_MAX_AGE_SECONDS = 7200  # 2 hours

AQ_ENDPOINT      = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# ERA5 reanalysis archive — covers 1940-present but lacks pressure-level variables
# (temperature_850hPa, temperature_700hPa, geopotential_height_500hPa).
# Used for all weather chunks whose end_date is before ERA5_CUTOFF.
ARCHIVE_ERA5_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
# Date on/after which the forecast-archive API has reliable data AND pressure-level vars.
# Verified empirically: Open-Meteo AQ data also starts ~2022-08-04.
ERA5_CUTOFF = datetime(2022, 8, 1)
AIRNOW_ENDPOINT  = "https://www.airnowapi.org/aq/observation/latLong/current/"

AQ_VARS = [
    "pm2_5", "pm10", "us_aqi", "carbon_monoxide",
    "nitrogen_dioxide", "ozone", "european_aqi", "dust", "uv_index",
    "aerosol_optical_depth"
]

# Core weather variables available from both ARCHIVE_ENDPOINT and ARCHIVE_ERA5_ENDPOINT.
WEATHER_VARS_CORE = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "boundary_layer_height",
    "precipitation", "cloud_cover", "visibility", "direct_radiation",
    "shortwave_radiation", "cloud_cover_low",
    "soil_temperature_0_to_7cm",
]

# Pressure-level variables only available from ARCHIVE_ENDPOINT (forecast archive,
# post-2022-08-01). ERA5 reanalysis does not expose these fields.
# features_v6.py guards all three with `if col in df.columns`, so NaN columns
# degrade gracefully — no crash, just missing inversion/blocking features.
WEATHER_VARS_PRESSURE_LEVEL = [
    "temperature_850hPa",          # V5: Upper-atmosphere temp for inversion detection
    "temperature_700hPa",          # V5.2: Inversion depth (700-850hPa gradient)
    "geopotential_height_500hPa",  # V5.1: Synoptic blocking ridges
]

# Full variable list used when fetching from ARCHIVE_ENDPOINT (post-cutoff).
WEATHER_VARS = WEATHER_VARS_CORE + WEATHER_VARS_PRESSURE_LEVEL


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_firms_recent(days: int = 2) -> pd.DataFrame:
    """Fetch recent NASA FIRMS active fire data for the local region."""
    api_key = os.environ.get("FIRMS_MAP_KEY")
    if not api_key:
        log.warning("FIRMS_MAP_KEY missing. Wildfire features will be 0.")
        return pd.DataFrame()
        
    sensors = ["MODIS_NRT", "VIIRS_SNPP_NRT"]
    bbox = "-123.0,37.0,-119.0,40.5"
    dfs = []
    
    for sensor in sensors:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{sensor}/{bbox}/{days}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.text.strip()) > 0:
                df_sensor = pd.read_csv(io.StringIO(resp.text))
                if not df_sensor.empty:
                    dfs.append(df_sensor)
        except Exception as e:
            log.error("FIRMS API error for %s: %s", sensor, e, exc_info=True)
            
    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    if 'latitude' not in df.columns or 'longitude' not in df.columns or 'frp' not in df.columns:
        return pd.DataFrame()
    
    # Haversine distance
    lat1, lon1 = np.radians(df['latitude']), np.radians(df['longitude'])
    lat2, lon2 = np.radians(LAT), np.radians(LON)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    df['distance_km'] = 6371 * (2 * np.arcsin(np.sqrt(a)))
    
    # V8: Compute bearing from Folsom TO each fire (degrees, 0=North, clockwise)
    # Used by the feature engineer for vectorized smoke advection dot product
    fire_lat_rad = np.radians(df['latitude'])
    fire_lon_rad = np.radians(df['longitude'])
    folsom_lat_rad = np.radians(LAT)
    folsom_lon_rad = np.radians(LON)
    d_lon = fire_lon_rad - folsom_lon_rad
    x = np.sin(d_lon) * np.cos(fire_lat_rad)
    y = np.cos(folsom_lat_rad) * np.sin(fire_lat_rad) - \
        np.sin(folsom_lat_rad) * np.cos(fire_lat_rad) * np.cos(d_lon)
    df['bearing_deg'] = np.degrees(np.arctan2(x, y)) % 360
    
    # Convert 'acq_date' + 'acq_time' to datetime
    df['acq_time'] = df['acq_time'].astype(str).str.zfill(4)
    df['datetime_utc'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'], format='%Y-%m-%d %H%M', errors='coerce').dt.tz_localize('UTC')
    df = df.dropna(subset=['datetime_utc'])
    df['datetime_local'] = df['datetime_utc'].dt.tz_convert(TZ).dt.floor('h')
    
    # For each hour, find the bearing of the nearest fire (the one whose smoke matters most)
    nearest_idx = df.groupby('datetime_local')['distance_km'].idxmin()
    nearest_bearings = df.loc[nearest_idx].set_index('datetime_local')['bearing_deg']
    
    # Aggregate hourly
    hourly = df.groupby('datetime_local').agg(
        fire_frp_raw=('frp', 'sum'),
        fire_count_raw=('frp', 'count'),
        fire_min_dist_raw=('distance_km', 'min')
    )
    hourly['fire_bearing_nearest'] = nearest_bearings
    
    hourly.index.name = 'time'
    return hourly


def fetch_firms_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical NASA FIRMS fire data for the Folsom bounding box.

    Why this exists:
      fetch_firms_recent() only covers the last 2 days and is used at inference
      time. During training, fire_frp_raw is always zero/NaN because the full
      history fetch never called FIRMS. This means the model trains on AQI spikes
      caused by wildfires (e.g., 2020 Creek Fire, 2021 Caldor Fire, 2022 Mosquito
      Fire) without any fire features to explain them — a systematic leakage of
      causal signal.

      This function fills that gap by fetching FIRMS data for the full training
      period in 5-day chunks (the API maximum), caching each chunk as parquet.

    Requires FIRMS_MAP_KEY in environment. Returns empty DataFrame if key is
    missing — the caller fills fire columns with zeros, matching inference behavior
    when no fires are detected.

    Args:
        start_date: 'YYYY-MM-DD' string.
        end_date:   'YYYY-MM-DD' string.

    Returns:
        DataFrame indexed by hourly timestamp (America/Los_Angeles) with columns:
        fire_frp_raw, fire_count_raw, fire_min_dist_raw, fire_bearing_nearest.
        Empty DataFrame if FIRMS_MAP_KEY is absent or no fires detected.
    """
    api_key = os.environ.get("FIRMS_MAP_KEY", "")
    if not api_key:
        log.warning(
            "FIRMS_MAP_KEY not set — skipping historical fire data fetch. "
            "Fire features will be zero during training. "
            "Set FIRMS_MAP_KEY in .env to enable historical fire data."
        )
        return pd.DataFrame()

    _ensure_cache_dir()
    tag   = f"firms_{start_date}_{end_date}"
    cache = _cache_key(tag)
    if cache.exists():
        log.info("FIRMS cache hit: %s", cache)
        return pd.read_parquet(cache)

    bbox    = "-123.0,37.0,-119.0,40.5"
    # Use Standard Processing (_SP) sensors for historical data — NRT sensors
    # only retain data for a short rolling window (~7 days). SP archives go back
    # to 2000 (MODIS) and 2012 (VIIRS SNPP), covering the full training period.
    sensors = ["MODIS_SP", "VIIRS_SNPP_SP"]
    all_frames: list[pd.DataFrame] = []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    chunk_days = 5  # FIRMS API maximum

    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_dt)
        date_str  = current.strftime("%Y-%m-%d")
        days_in_chunk = (chunk_end - current).days + 1

        for sensor in sensors:
            url = (
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
                f"/{api_key}/{sensor}/{bbox}/{days_in_chunk}/{date_str}"
            )
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.text.strip()) > 0:
                    df_chunk = pd.read_csv(io.StringIO(resp.text))
                    if not df_chunk.empty and "frp" in df_chunk.columns:
                        all_frames.append(df_chunk)
                elif resp.status_code == 429:
                    log.warning("FIRMS rate limited — waiting 30s")
                    time.sleep(30)
                else:
                    log.debug("FIRMS %s %s: status=%s", sensor, date_str, resp.status_code)
            except Exception as exc:
                log.warning("FIRMS chunk %s %s failed: %s", sensor, date_str, exc)

        current += timedelta(days=chunk_days)
        time.sleep(0.5)  # courtesy delay — 5000 tx/10min limit

    if not all_frames:
        log.info("FIRMS history: no fire detections for %s → %s", start_date, end_date)
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)

    # ── Haversine distance ────────────────────────────────────────────────
    lat1 = np.radians(df["latitude"].values)
    lon1 = np.radians(df["longitude"].values)
    lat2, lon2 = np.radians(LAT), np.radians(LON)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["distance_km"] = 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

    # ── Bearing from Folsom to each fire ──────────────────────────────────
    fire_lat_r    = np.radians(df["latitude"].values)
    fire_lon_r    = np.radians(df["longitude"].values)
    folsom_lat_r  = np.radians(LAT)
    folsom_lon_r  = np.radians(LON)
    d_lon_b       = fire_lon_r - folsom_lon_r
    bx = np.sin(d_lon_b) * np.cos(fire_lat_r)
    by = (np.cos(folsom_lat_r) * np.sin(fire_lat_r)
          - np.sin(folsom_lat_r) * np.cos(fire_lat_r) * np.cos(d_lon_b))
    df["bearing_deg"] = np.degrees(np.arctan2(bx, by)) % 360.0

    # ── Datetime conversion ───────────────────────────────────────────────
    df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
    df["datetime_utc"] = pd.to_datetime(
        df["acq_date"] + " " + df["acq_time"],
        format="%Y-%m-%d %H%M", errors="coerce"
    ).dt.tz_localize("UTC")
    df = df.dropna(subset=["datetime_utc"])
    df["datetime_local"] = df["datetime_utc"].dt.tz_convert(TZ).dt.floor("h")

    # ── Hourly aggregation ────────────────────────────────────────────────
    nearest_idx      = df.groupby("datetime_local")["distance_km"].idxmin()
    nearest_bearings = df.loc[nearest_idx].set_index("datetime_local")["bearing_deg"]

    hourly = df.groupby("datetime_local").agg(
        fire_frp_raw=("frp",         "sum"),
        fire_count_raw=("frp",       "count"),
        fire_min_dist_raw=("distance_km", "min"),
    )
    hourly["fire_bearing_nearest"] = nearest_bearings
    hourly.index.name = "time"

    # ── Seam deduplication ────────────────────────────────────────────────
    n_dupes = hourly.index.duplicated().sum()
    if n_dupes > 0:
        log.warning("FIRMS history: %d duplicate timestamps — keeping last", n_dupes)
        hourly = hourly[~hourly.index.duplicated(keep="last")]

    hourly.to_parquet(cache)
    log.info(
        "FIRMS history: %d fire-hours cached → %s  (range: %s → %s)",
        len(hourly), cache, start_date, end_date,
    )
    return hourly


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(tag: str) -> Path:
    return CACHE_DIR / f"{tag}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < CACHE_MAX_AGE_SECONDS


def _fetch_with_retry(url: str, params: dict, max_retries: int = 4) -> dict:
    """GET request with rate-limit-aware exponential backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            log.info("Attempt %s: %s...", attempt + 1, full_url[:120])
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            body = ""
            try:
                body = exc.response.text[:500]
            except Exception:
                pass
            log.error("HTTP %s: %s", status, body)
            last_exc = exc
            # 429 = rate limited — wait much longer
            if status == 429:
                retry_after = int(exc.response.headers.get("Retry-After", 30))
                wait = max(retry_after, 30)
                log.warning("Rate limited! Waiting %ss...", wait)
                time.sleep(wait)
                continue
        except Exception as exc:
            log.error("Attempt %s failed: %s", attempt + 1, exc, exc_info=True)
            last_exc = exc
        wait = 2 ** (attempt + 1)
        if attempt < max_retries - 1:
            log.info("Retrying in %ss...", wait)
            time.sleep(wait)
    raise RuntimeError(f"All {max_retries} retries failed for {url}: {last_exc}")


def _hourly_to_df(data: dict, variables: list[str]) -> pd.DataFrame:
    """Convert Open-Meteo hourly JSON response to DataFrame with tz-aware index."""
    hourly = data["hourly"]
    times  = pd.to_datetime(hourly["time"])
    df = pd.DataFrame(index=times)
    for var in variables:
        if var in hourly:
            df[var] = hourly[var]
        else:
            df[var] = float("nan")
    # Localize: Open-Meteo returns naive local time strings without UTC offset.
    # ambiguous="NaT"  → marks the one ambiguous hour on DST fall-back day as NaT
    #                    (instead of crashing). Affects ~1 row per year.
    # nonexistent="shift_forward" → handles the spring-forward gap the same way.
    # We then drop NaT-indexed rows so the rest of the pipeline stays clean.
    if df.index.tz is None:
        df.index = df.index.tz_localize(
            TZ, ambiguous="NaT", nonexistent="shift_forward"
        )
        df = df[df.index.notna()]   # drop the ~1 ambiguous DST row per year
    else:
        df.index = df.index.tz_convert(TZ)
    df.index.name = "timestamp"
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_air_quality_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical hourly AQ data from Open-Meteo.
    start_date, end_date: 'YYYY-MM-DD' strings.
    Uses parquet cache — skips API if cache is fresh.
    Returns DataFrame or last valid cache on failure.
    """
    _ensure_cache_dir()
    tag   = f"aq_{start_date}_{end_date}"
    cache = _cache_key(tag)

    if _cache_is_fresh(cache):
        log.info("AQ cache hit: %s", cache)
        return pd.read_parquet(cache)

    params = {
        "latitude":  LAT,
        "longitude": LON,
        "hourly":    ",".join(AQ_VARS),
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   TZ,
    }
    try:
        data = _fetch_with_retry(AQ_ENDPOINT, params)
        df   = _hourly_to_df(data, AQ_VARS)
        df.to_parquet(cache)
        log.info("AQ data fetched: %s rows, saved to %s", len(df), cache)
        return df
    except Exception as exc:
        log.error("ERROR fetching AQ history: %s", exc, exc_info=True)
        if cache.exists():
            log.warning("Returning stale cache: %s", cache)
            return pd.read_parquet(cache)
        raise


def fetch_weather_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical weather from Open-Meteo, routing to the correct endpoint
    based on the requested date range.

    Routing logic (verified empirically):
      - Dates entirely before ERA5_CUTOFF (2022-08-01):
          Use ARCHIVE_ERA5_ENDPOINT (ERA5 reanalysis, 1940-present).
          Only WEATHER_VARS_CORE are requested — ERA5 does not expose pressure-level
          variables. Missing columns are filled with NaN so features_v6.py degrades
          gracefully (all pressure-level features are guarded by `if col in df.columns`).
      - Dates on or after ERA5_CUTOFF:
          Use ARCHIVE_ENDPOINT (forecast archive, full WEATHER_VARS including 850hPa).
      - Chunks that straddle the cutoff are split into two sub-requests and
          concatenated, ensuring no data is lost at the seam.

    Why split at 2022-08-01:
      The forecast archive starts returning reliable data around this date.
      ERA5 is the gold-standard reanalysis product and is preferred for older dates
      because it has better spatial interpolation for the Folsom grid cell.
    """
    _ensure_cache_dir()
    tag   = f"wx_{start_date}_{end_date}"
    cache = _cache_key(tag)

    if _cache_is_fresh(cache):
        log.info("WX cache hit: %s", cache)
        return pd.read_parquet(cache)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    frames: list[pd.DataFrame] = []

    # ── Sub-request A: ERA5 portion (before cutoff) ───────────────────────
    if start_dt < ERA5_CUTOFF:
        era5_end = min(end_dt, ERA5_CUTOFF - timedelta(days=1))
        era5_end_str = era5_end.strftime("%Y-%m-%d")
        log.info("  WX ERA5 sub-request: %s → %s", start_date, era5_end_str)
        params_era5 = {
            "latitude":   LAT,
            "longitude":  LON,
            "hourly":     ",".join(WEATHER_VARS_CORE),
            "start_date": start_date,
            "end_date":   era5_end_str,
            "timezone":   TZ,
        }
        data_era5 = _fetch_with_retry(ARCHIVE_ERA5_ENDPOINT, params_era5)
        df_era5   = _hourly_to_df(data_era5, WEATHER_VARS_CORE)
        # Fill pressure-level columns with NaN so downstream merge is schema-consistent
        for col in WEATHER_VARS_PRESSURE_LEVEL:
            df_era5[col] = np.nan
        frames.append(df_era5)

    # ── Sub-request B: Forecast archive portion (on/after cutoff) ─────────
    if end_dt >= ERA5_CUTOFF:
        archive_start = max(start_dt, ERA5_CUTOFF)
        archive_start_str = archive_start.strftime("%Y-%m-%d")
        log.info("  WX archive sub-request: %s → %s", archive_start_str, end_date)
        params_arc = {
            "latitude":   LAT,
            "longitude":  LON,
            "hourly":     ",".join(WEATHER_VARS),
            "start_date": archive_start_str,
            "end_date":   end_date,
            "timezone":   TZ,
        }
        data_arc = _fetch_with_retry(ARCHIVE_ENDPOINT, params_arc)
        df_arc   = _hourly_to_df(data_arc, WEATHER_VARS)
        frames.append(df_arc)

    if not frames:
        raise RuntimeError(
            f"fetch_weather_history produced no data for {start_date} → {end_date}"
        )

    df = pd.concat(frames).sort_index()

    # ── Seam integrity check (Rule 1: Zero Silent Failures) ───────────────
    # Duplicates at the 2022-08-01 boundary would silently corrupt rolling features.
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        log.warning(
            "  WX seam: %d duplicate timestamps found at cutoff boundary — "
            "keeping last (forecast archive preferred over ERA5 at overlap).",
            n_dupes,
        )
        df = df[~df.index.duplicated(keep="last")]

    df.to_parquet(cache)
    log.info("WX data fetched: %s rows, saved to %s", len(df), cache)
    return df


def fetch_recent_combined(past_hours: int = 168) -> pd.DataFrame:
    """
    Fetch the last `past_hours` of AQ + weather data for inference.
    Uses Open-Meteo's past_days parameter for efficient recent-data fetch.
    Returns merged DataFrame.
    """
    _ensure_cache_dir()
    past_days = max(1, (past_hours // 24) + 1)
    tag       = f"recent_combined_ph{past_hours}"
    cache     = _cache_key(tag)

    if _cache_is_fresh(cache):
        log.info("Recent combined cache hit: %s", cache)
        return pd.read_parquet(cache)

    # AQ: use past_days for history + forecast to ensure 'Today' is fully covered
    aq_params = {
        "latitude":       LAT,
        "longitude":      LON,
        "hourly":         ",".join(AQ_VARS),
        "past_days":      past_days,
        "forecast_days":  3,           # V4.0: bumped from 2 for 48h feature coverage
        "timezone":       TZ,
    }
    # Weather: use past_days + extended forecast for 48h forward-shifted features
    wx_params = {
        "latitude":       LAT,
        "longitude":      LON,
        "hourly":         ",".join(WEATHER_VARS),
        "past_days":      past_days,
        "forecast_days":  5,           # V4.0: bumped from 3 to cover 48h+ horizon features
        "timezone":       TZ,
    }
    try:
        aq_data  = _fetch_with_retry(AQ_ENDPOINT, aq_params)
        aq_df    = _hourly_to_df(aq_data, AQ_VARS)
        time.sleep(2)  # Courtesy delay to avoid 429 rate-limiting on shared Render IPs
        wx_data  = _fetch_with_retry(WEATHER_ENDPOINT, wx_params)
        wx_df    = _hourly_to_df(wx_data, WEATHER_VARS)

        merged   = _merge_aq_weather(aq_df, wx_df)
        
        # Inject FIRMS satellite fire data (V6.0)
        firms_df = fetch_firms_recent()
        if not firms_df.empty:
            merged = merged.join(firms_df, how='left')
            merged['fire_frp_raw'] = merged['fire_frp_raw'].fillna(0.0)
            merged['fire_count_raw'] = merged['fire_count_raw'].fillna(0.0)
            merged['fire_min_dist_raw'] = merged['fire_min_dist_raw'].fillna(999.0)
        else:
            merged['fire_frp_raw'] = 0.0
            merged['fire_count_raw'] = 0.0
            merged['fire_min_dist_raw'] = 999.0

        merged.to_parquet(cache)
        log.info("Recent combined: %s rows", len(merged))
        return merged

    except Exception as exc:
        log.error("ERROR fetching recent data: %s", exc, exc_info=True)
        if cache.exists():
            log.warning("Returning stale cache: %s", cache)
            return pd.read_parquet(cache)
        raise


def fetch_airnow_current() -> dict | None:
    """
    Fetch current AQI reading from AirNow sensor network.
    Returns dict with keys: aqi, category, primary_pollutant, timestamp.
    Returns None on any failure (caller falls back to Open-Meteo).
    """
    api_key = os.environ.get("AIRNOW_API_KEY", "")
    if not api_key:
        log.warning("AIRNOW_API_KEY not set, skipping AirNow fetch.")
        return None

    params = {
        "format":    "application/json",
        "latitude":  LAT,
        "longitude": LON,
        "distance":  25,
        "API_KEY":   api_key,
    }
    try:
        data = _fetch_with_retry(AIRNOW_ENDPOINT, params, max_retries=2)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        # Pick the entry with the highest AQI (dominant pollutant)
        best = max(data, key=lambda x: x.get("AQI", 0))
        cat  = best.get("Category", {})

        return {
            "aqi":               int(best.get("AQI", 0)),
            "category":          cat.get("Name", "Unknown"),
            "primary_pollutant": best.get("ParameterName", "Unknown"),
            "timestamp":         best.get("DateObserved", "").strip() + "T" +
                                 f"{best.get('HourObserved', 0):02d}:00:00",
        }
    except Exception as exc:
        log.error("AirNow fetch failed: %s", exc, exc_info=True)
        return None


def fetch_full_history() -> pd.DataFrame:
    """
    Fetch merged AQ + weather data for training, starting from 2019-01-01.

    Why 2019:
      ERA5 weather data is available from 1940, but Open-Meteo AQ data only
      starts ~2022-08-04. The 2019-2022 weather rows provide "feature warm-up"
      for rolling windows (168h mean, 48h stagnation, etc.) so that the first
      AQI-valid rows in August 2022 have properly computed lag features rather
      than imputed-median values. The AQI target for 2019-2022 rows will be NaN
      and is dropped by `mask = y_full.notna()` in train.py — these rows only
      contribute to feature context, not to the training target.

    Chunking: yearly to stay within Open-Meteo API rate limits.
    """
    start = "2019-01-01"
    end   = datetime.now().strftime("%Y-%m-%d")

    aq_frames, wx_frames = [], []

    chunk_start = datetime.strptime(start, "%Y-%m-%d")
    chunk_end   = datetime.strptime(end,   "%Y-%m-%d")

    current = chunk_start
    while current < chunk_end:
        next_chunk = min(current + timedelta(days=365), chunk_end)
        s = current.strftime("%Y-%m-%d")
        e = next_chunk.strftime("%Y-%m-%d")

        log.info("Fetching chunk %s → %s", s, e)
        aq_frames.append(fetch_air_quality_history(s, e))
        wx_frames.append(fetch_weather_history(s, e))
        current = next_chunk

    aq_all = pd.concat(aq_frames).sort_index()
    wx_all = pd.concat(wx_frames).sort_index()

    # Deduplicate chunk boundaries
    aq_all = aq_all[~aq_all.index.duplicated(keep="last")]
    wx_all = wx_all[~wx_all.index.duplicated(keep="last")]

    # ── ERA5/archive seam integrity check ────────────────────────────────
    # Verify no duplicate timestamps survive into the final merged frame.
    # A duplicate here would silently corrupt all rolling-window features.
    seam_start = pd.Timestamp("2022-07-31", tz=TZ)
    seam_end   = pd.Timestamp("2022-08-02", tz=TZ)
    seam_wx    = wx_all.loc[seam_start:seam_end]
    n_seam_dupes = seam_wx.index.duplicated().sum()
    if n_seam_dupes > 0:
        raise RuntimeError(
            f"ERA5/archive seam integrity check FAILED: "
            f"{n_seam_dupes} duplicate timestamps in wx_all around 2022-08-01. "
            "This would corrupt rolling features. Investigate fetch_weather_history."
        )
    log.info("  ERA5/archive seam check passed (0 duplicates around 2022-08-01).")

    merged = _merge_aq_weather(aq_all, wx_all)

    # ── FIRMS historical fire data ────────────────────────────────────────
    # Fetch fire detections for the full training period and join onto the
    # merged DataFrame. This populates fire_frp_raw, fire_count_raw,
    # fire_min_dist_raw, and fire_bearing_nearest — the same columns that
    # fetch_recent_combined populates at inference time.
    # Gracefully skips if FIRMS_MAP_KEY is not set (fire features → zero).
    firms_df = fetch_firms_history(start, end)
    if not firms_df.empty:
        merged = merged.join(firms_df, how="left")
        merged["fire_frp_raw"]       = merged["fire_frp_raw"].fillna(0.0)
        merged["fire_count_raw"]     = merged["fire_count_raw"].fillna(0.0)
        merged["fire_min_dist_raw"]  = merged["fire_min_dist_raw"].fillna(999.0)
        log.info(
            "  FIRMS joined: %d fire-hours out of %d total rows (%.1f%%)",
            (merged["fire_frp_raw"] > 0).sum(),
            len(merged),
            100.0 * (merged["fire_frp_raw"] > 0).sum() / len(merged),
        )
    else:
        merged["fire_frp_raw"]      = 0.0
        merged["fire_count_raw"]    = 0.0
        merged["fire_min_dist_raw"] = 999.0
        log.info("  FIRMS: no key or no data — fire features set to zero.")

    log.info(
        "  Full history: %s rows  |  AQI non-null: %s  |  Range: %s → %s",
        f"{len(merged):,}",
        f"{pd.to_numeric(merged.get('us_aqi', pd.Series()), errors='coerce').notna().sum():,}",
        merged.index.min().date(),
        merged.index.max().date(),
    )
    return merged


def _merge_aq_weather(aq_df: pd.DataFrame, wx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge AQ and weather DataFrames on their hourly DatetimeIndex.
    Uses outer join so neither side loses rows; NaNs handled later by imputer.
    """
    merged = aq_df.join(wx_df, how="outer", rsuffix="_wx")
    # Drop any duplicate columns that may appear from rsuffix
    dup_cols = [c for c in merged.columns if c.endswith("_wx")]
    merged.drop(columns=dup_cols, inplace=True)
    merged.sort_index(inplace=True)
    return merged