"""
data_fetcher.py — Fetch AQI and weather data from Open-Meteo and AirNow.
All fetches use retry with exponential backoff and local parquet cache.
Returns pd.DataFrame with DatetimeIndex in America/Los_Angeles.
Never crashes — returns last valid cache on any failure.
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import urllib.parse
from dotenv import load_dotenv

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
AIRNOW_ENDPOINT  = "https://www.airnowapi.org/aq/observation/latLong/current/"

AQ_VARS = [
    "pm2_5", "pm10", "us_aqi", "carbon_monoxide",
    "nitrogen_dioxide", "ozone", "european_aqi", "dust", "uv_index",
    "aerosol_optical_depth"
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "boundary_layer_height",
    "precipitation", "cloud_cover", "visibility", "direct_radiation",
    "shortwave_radiation", "cloud_cover_low",
    "soil_temperature_0_to_7cm",
    "temperature_850hPa",  # V5: Upper-atmosphere temp for inversion detection
    "temperature_700hPa",   # V5.2: For inversion depth (700-850hPa gradient)
    "geopotential_height_500hPa",  # V5.1: Synoptic blocking ridges
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

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
            print(f"[data_fetcher] Attempt {attempt+1}: {full_url[:120]}...", file=sys.stderr)
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
            print(f"[data_fetcher] HTTP {status}: {body}", file=sys.stderr)
            last_exc = exc
            # 429 = rate limited — wait much longer
            if status == 429:
                retry_after = int(exc.response.headers.get("Retry-After", 30))
                wait = max(retry_after, 30)
                print(f"[data_fetcher] Rate limited! Waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
        except Exception as exc:
            print(f"[data_fetcher] Attempt {attempt+1} failed: {exc}", file=sys.stderr)
            last_exc = exc
        wait = 2 ** (attempt + 1)
        if attempt < max_retries - 1:
            print(f"[data_fetcher] Retrying in {wait}s...", file=sys.stderr)
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
        print(f"[data_fetcher] AQ cache hit: {cache}")
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
        print(f"[data_fetcher] AQ data fetched: {len(df)} rows, saved to {cache}")
        return df
    except Exception as exc:
        print(f"[data_fetcher] ERROR fetching AQ history: {exc}", file=sys.stderr)
        if cache.exists():
            print(f"[data_fetcher] Returning stale cache: {cache}", file=sys.stderr)
            return pd.read_parquet(cache)
        raise


def fetch_weather_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical + forecast weather from Open-Meteo.
    Uses past_days/forecast_days for recent windows, date range for historical.
    """
    _ensure_cache_dir()
    tag   = f"wx_{start_date}_{end_date}"
    cache = _cache_key(tag)

    if _cache_is_fresh(cache):
        print(f"[data_fetcher] WX cache hit: {cache}")
        return pd.read_parquet(cache)

    params = {
        "latitude":       LAT,
        "longitude":      LON,
        "hourly":         ",".join(WEATHER_VARS),
        "start_date":     start_date,
        "end_date":       end_date,
        "timezone":       TZ,
    }
    try:
        data = _fetch_with_retry(ARCHIVE_ENDPOINT, params)
        df   = _hourly_to_df(data, WEATHER_VARS)
        df.to_parquet(cache)
        print(f"[data_fetcher] WX data fetched: {len(df)} rows, saved to {cache}")
        return df
    except Exception as exc:
        print(f"[data_fetcher] ERROR fetching weather history: {exc}", file=sys.stderr)
        if cache.exists():
            print(f"[data_fetcher] Returning stale cache: {cache}", file=sys.stderr)
            return pd.read_parquet(cache)
        raise


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
        print(f"[data_fetcher] Recent combined cache hit: {cache}")
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
        merged.to_parquet(cache)
        print(f"[data_fetcher] Recent combined: {len(merged)} rows")
        return merged

    except Exception as exc:
        print(f"[data_fetcher] ERROR fetching recent data: {exc}", file=sys.stderr)
        if cache.exists():
            print(f"[data_fetcher] Returning stale cache: {cache}", file=sys.stderr)
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
        print("[data_fetcher] AIRNOW_API_KEY not set, skipping AirNow fetch.", file=sys.stderr)
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
        print(f"[data_fetcher] AirNow fetch failed: {exc}", file=sys.stderr)
        return None


def fetch_full_history() -> pd.DataFrame:
    """
    Fetch 4 years of merged AQ + weather data for training.
    Splits into yearly chunks to stay within API limits.
    """
    start = "2021-01-01" 
    end   = datetime.now().strftime("%Y-%m-%d")

    aq_frames, wx_frames = [], []

    # Chunk by year to keep requests manageable
    chunk_start = datetime.strptime(start, "%Y-%m-%d")
    chunk_end   = datetime.strptime(end,   "%Y-%m-%d")

    current = chunk_start
    while current < chunk_end:
        next_chunk = min(current + timedelta(days=365), chunk_end)
        s = current.strftime("%Y-%m-%d")
        e = next_chunk.strftime("%Y-%m-%d")

        print(f"[data_fetcher] Fetching chunk {s} → {e}")
        aq_frames.append(fetch_air_quality_history(s, e))
        wx_frames.append(fetch_weather_history(s, e))
        current = next_chunk

    aq_all = pd.concat(aq_frames).sort_index()
    wx_all = pd.concat(wx_frames).sort_index()

    # Remove duplicates that appear at chunk boundaries
    aq_all = aq_all[~aq_all.index.duplicated(keep='last')]
    wx_all = wx_all[~wx_all.index.duplicated(keep='last')]

    return _merge_aq_weather(aq_all, wx_all)


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