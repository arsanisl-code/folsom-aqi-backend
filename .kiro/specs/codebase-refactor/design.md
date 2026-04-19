# Design Document

## Overview

This document specifies the implementation plan for the Folsom AQI Forecast codebase refactor. The refactor targets seven quality pillars without altering any externally visible API contract or model behavior: Reduced Cognitive Load, High Changeability, Resilience, Observability, Self-Documenting Intent, Testability, and Code Hygiene.

The changes are purely structural. No ML model weights, API schemas, or forecast algorithms change. Every diff is a rename, extract, or delete — never a logic change.

### Guiding Principles

- **Behavioral equivalence is non-negotiable.** engineer_features() must produce bit-identical output before and after decomposition.
- **Deletion before addition.** Dead code is removed first so the working set is minimal before new structure is added.
- **One concern per function.** Each extracted sub-function owns exactly one feature group or pipeline stage.
- **Named constants over magic numbers.** Every numeric literal with physical meaning becomes a module-level constant with an explanatory comment.
- **Logging replaces printing.** All print() calls are replaced with structured logger calls; no new print() calls are introduced.

## Architecture

The refactor does not change the runtime topology. The system remains:

```
GitHub Actions (cron) → refresh.py → inference.py → data_fetcher.py
                                                   → features_v6.py
                                                   → ai_layer.py
                                                   → data/latest.json

FastAPI (Render)       → api.py → load_cached_forecast() → GitHub CDN / data/latest.json

Streamlit Cloud        → frontend/app.py → API_URL/forecast
                                         → ai_layer.py (backend, shared)
```

### New Module: `backend/logger.py`

A single new file is introduced. All other changes are modifications to existing files.

### Dependency Graph (post-refactor)

```
refresh.py
  └── inference.py
        ├── data_fetcher.py  (logger)
        ├── features_v6.py   (logger)
        └── ai_layer.py      (logger)

api.py
  └── inference.py (load_cached_forecast, cache_age_minutes)

train.py
  ├── data_fetcher.py
  └── features_v6.py  ← was features.py

frontend/app.py
  └── backend/ai_layer.py  ← was frontend/ai_layer.py (deleted)
```

---

## Components and Interfaces

### 1. `backend/logger.py` (new file)

**Purpose:** Single source of truth for all structured logging configuration.

```python
# backend/logger.py

import logging
import os
import sys

def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Return a Logger configured with:
      - StreamHandler(stdout) for DEBUG and INFO
      - StreamHandler(stderr) for WARNING, ERROR, CRITICAL
      - Format: %(asctime)s %(levelname)-8s %(name)s — %(message)s
      - Level defaults to LOG_LEVEL env var, falling back to INFO.

    Idempotent: calling get_logger("foo") twice returns the same logger
    with handlers attached only once.
    """
```

**Implementation details:**

- Use a `_LevelFilter` inner class (subclass of `logging.Filter`) to route by level:
  - stdout handler: `addFilter(_LevelFilter(max_level=logging.INFO))`
  - stderr handler: `addFilter(_LevelFilter(min_level=logging.WARNING))`
- The `_LevelFilter` checks `record.levelno` against its threshold.
- Guard against duplicate handlers: `if not logger.handlers:` before attaching.
- `level` parameter: if `None`, read `os.environ.get("LOG_LEVEL", "INFO")` and convert with `logging.getLevelName()`.
- Date format: `"%Y-%m-%dT%H:%M:%S"` (ISO 8601, no microseconds for readability).
- Format string: `"%(asctime)s %(levelname)-8s %(name)s — %(message)s"`.

**Usage pattern in every module:**

```python
from logger import get_logger
log = get_logger(__name__)

# Replace: print("msg", file=sys.stderr)
# With:    log.error("msg")

# Replace: print(f"[data_fetcher] ERROR ...: {exc}", file=sys.stderr)
# With:    log.error("fetch failed: %s", exc, exc_info=True)
```

The `exc_info=True` kwarg satisfies Requirement 1.9 (exception type + message in ERROR/CRITICAL lines).

---

### 2. `backend/features_v6.py` — Decomposition

**Current state:** One 400-line `engineer_features()` function.

**Target state:** `engineer_features()` becomes a 7-line orchestration shell calling private sub-functions.

#### Module-level constants (new, replacing magic numbers)

```python
# Empirical Central Valley stagnation threshold (California Air Resources Board)
VENT_THRESHOLD_M2_PER_S: float = 3000.0

# Temperature below which residential wood burning in Folsom increases significantly
# (based on local emission inventory data; 7°C ≈ 45°F)
COLD_SMOKE_THRESHOLD_C: float = 7.0

# Magnus formula coefficients for saturation vapor pressure approximation
MAGNUS_A: float = 17.27
MAGNUS_B: float = 237.3

# Open-Meteo hallucination floor for relative humidity (Folsom grid cell artifact)
HUMIDITY_CLAMP_MIN_PCT: float = 25.0

# Open-Meteo hallucination cap for wind speed (Folsom grid cell artifact)
WIND_SPEED_CAP_KMH: float = 25.0

# Minimum precipitation to count as a wet-deposition scavenging event
RAIN_THRESHOLD_MM: float = 0.1
```

#### Sub-function signatures

```python
def _add_aqi_lag_features(X: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Mutates X in-place. Adds Groups 1–2:
    - AQI lags (0, 1, 2, 3, 6, 12, 24, 48h)
    - AQI diffs (1h, 24h), acceleration, acceleration 6h mean
    - Rolling stats (mean, max, std) for windows [3,6,12,24,48,168]h
    - EWMA (6h, 24h spans)
    - PM2.5 EWMA (6h span)
    """

def _add_pm25_and_combustion_features(X: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Mutates X in-place. Adds Group 3 + Feature Group 1:
    - PM2.5 current + lags (1,3,6,24h) + rolling (6h, 24h mean)
    - CO current, lags, rolling, diff (if column present)
    - PM2.5/CO wildfire discrimination ratio (if both present)
    - Dust current, rolling, diff (if column present)
    - NO2 current, rolling (if column present)
    """

def _add_meteorological_features(
    X: pd.DataFrame, df: pd.DataFrame, horizon_h: int
) -> None:
    """
    Mutates X in-place. Adds Group 4:
    - Raw met columns: BLH, wind speed, pressure, humidity, temp, precip,
      cloud cover, direct radiation
    - Physical interactions: blh_x_wind_speed, aqi_x_wind, aqi_x_rad
    - Photochemical forcing index (current + 6h mean + 12h accumulation)
    - Forward photochemical forcing at T+horizon_h
    - Forward NWP features (wind, BLH, temp, humidity, pressure, precip)
      shifted by -horizon_h
    - Forward rolling means (temp, wind, humidity, pressure, precip accumulation)
    - Forward ventilation coefficient and deficit
    - Forward fire danger index (HDWI at T+h)
    - AOD current + diffs (if column present)
    - Wind direction sin/cos + U/V decomposition (current and forward)
    """

def _add_atmospheric_stability_features(
    X: pd.DataFrame, df: pd.DataFrame, horizon_h: int
) -> None:
    """
    Mutates X in-place. Adds Group 5 (stability sub-groups 5a–5f):
    5a. Stagnation index (6h, 24h, 48h rolling sums + streak)
    5b. Inversion proxy (strength, 12h max)
    5b2. True inversion delta 850hPa (if column present)
    4b. 700hPa inversion depth (if both 700/850 columns present)
    5c. Ventilation deficit (current + 24h mean)
    5d. Synoptic blocking index Z500 (if column present)
    5e. Cold degree hours (current + 48h sum)
    5f. Tule fog precursor (dew point depression + 12h rolling)
    Also adds fog_nitrate_index.
    """

def _add_wildfire_features(
    X: pd.DataFrame, df: pd.DataFrame, horizon_h: int
) -> None:
    """
    Mutates X in-place. Adds wildfire proxy features:
    - HDWI (Hot-Dry-Windy Index using VPD × wind)
    - Antecedent precipitation deficit (30-day sum)
    - Hours/days since last rain (dry streak counter)
    - Extreme heat/dry flag
    - Pressure and temperature front differencing (3h, 6h, 12h, 24h, 48h)
    - FIRMS FRP features (if fire_frp_raw column present):
      current, 24h sum, min distance, count, intensity-proximity index,
      advection score (directional if fire_bearing_nearest present),
      24h max advection, forward advection at T+horizon_h
    """

def _add_temporal_features(
    X: pd.DataFrame, df: pd.DataFrame, horizon_h: int
) -> None:
    """
    Mutates X in-place. Adds Group 6:
    - Cyclic encodings: hour sin/cos, DOW sin/cos, DOY sin/cos,
      month sin/cos, day_of_week sin/cos
    - is_weekend flag
    - Second harmonic (commute traffic): hour_sin_2, hour_cos_2,
      weekday_rush_proxy
    - Future hour cyclic: future_hour_sin, future_hour_cos
    """

def _add_regulatory_features(X: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Mutates X in-place. Adds Group 7:
    - cbyb_season_flag (Nov–Feb wood-burning season)
    - weekend_burning_proxy (is_weekend × cold_degree_hours)
    - radiation_accum_6h (if shortwave_radiation column present)
    Also adds V8 multi-scale atmospheric momentum features:
    - aqi_momentum_6h, aqi_momentum_24h, momentum_accel_6h
    - aqi_zscore_7d, fat_tail_flag, fat_tail_persistence_48h
    And V9 second-order interaction features:
    - stability_index, trapping_power, fwd_ventilation_stress,
      volatility_frontal, summer_photochem_accum
    And V9 evening BLH collapse velocity:
    - blh_collapse_rate, fwd_blh_collapse_rate, evening_trap_flag
    """
```

#### Refactored `engineer_features()` shell

```python
def engineer_features(df: pd.DataFrame, horizon_h: int) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df.index)
    df = df.apply(pd.to_numeric, errors='coerce')

    _add_aqi_lag_features(X, df)
    _add_pm25_and_combustion_features(X, df)
    _add_meteorological_features(X, df, horizon_h)
    _add_atmospheric_stability_features(X, df, horizon_h)
    _add_wildfire_features(X, df, horizon_h)
    _add_temporal_features(X, df, horizon_h)
    _add_regulatory_features(X, df)

    y = df['us_aqi'].shift(-horizon_h) - df['us_aqi']
    y.name = 'target_residual'
    return X, y
```

**Critical constraint:** The `X.copy()` defragmentation calls that currently appear mid-function must be preserved inside the sub-functions at the same logical points to maintain memory layout equivalence. Specifically:
- After EWMA block in `_add_aqi_lag_features`: `X = X.copy()` — **not applicable** since X is mutated in-place; instead, the caller's `X` reference is updated. The solution: sub-functions that need defragmentation call `X.update(X.copy())` or the orchestrator calls `X = X.copy()` after each sub-function call. The simplest correct approach: keep `X.copy()` calls inside each sub-function by reassigning via `X.__dict__.update(pd.DataFrame(X).copy().__dict__)` — this is fragile. **Preferred approach:** the orchestrator calls `X = X.copy()` after each sub-function returns, since the sub-functions mutate X in-place and the copy is just a defragmentation hint to pandas. This preserves behavioral equivalence.

**Variable rename:** `blh` (local variable in Group 5) → `boundary_layer_height`. The DataFrame column name `boundary_layer_height` is unchanged. The local variable `wind` (used in Group 5) is already descriptive and is not renamed.

---

### 3. `backend/inference.py` — Decomposition

**Current state:** One 200-line `predict_now()` function with inline business logic.

**Target state:** `predict_now()` becomes a 10-line orchestration shell.

#### Module-level constants (new)

```python
# Open-Meteo hallucination floor for relative humidity (Folsom grid cell artifact)
# Guards against documented cases where the API returns 5-6% RH for Folsom
HUMIDITY_FLOOR_PCT: float = 25.0

# Open-Meteo hallucination cap for wind speed (Folsom grid cell artifact)
# Guards against documented cases where the API returns 30+ km/h for calm days
WIND_SPEED_CAP_KMH: float = 25.0
```

#### Sub-function signatures

```python
def _fetch_input_data(past_hours: int = 168) -> tuple[pd.DataFrame, int]:
    """
    Fetch recent combined data and compute data age.

    Returns:
        (df, data_age_minutes) where data_age_minutes is how old the
        most recent row is relative to now.

    Raises:
        RuntimeError if fetch fails and no cache is available.
    """

def _resolve_current_conditions(
    df: pd.DataFrame,
    airnow: dict | None,
) -> dict:
    """
    Determine the current AQI reading and its source.

    Priority: AirNow (if aqi > 0) > Open-Meteo most recent row.

    Returns dict with keys: aqi, category, color, primary_pollutant,
    source, timestamp.
    """

def _prepare_horizon_dataframe(
    df_base: pd.DataFrame,
    airnow: dict | None,
) -> pd.DataFrame:
    """
    Apply AirNow Uniform Offset Calibration and sensor sanity clamping.

    # AirNow Uniform Offset Calibration:
    # A uniform offset is applied to the entire Open-Meteo AQI curve rather
    # than a point injection at now_ts. Point injection creates an unphysical
    # differential spike at T=0 that corrupts all rolling features (aqi_diff_1h,
    # aqi_roll_*) for the current hour. A uniform shift preserves the shape of
    # the time series while anchoring it to the ground-truth AirNow reading.

    # Sensor sanity clamping:
    # HUMIDITY_FLOOR_PCT = 25.0 — guards against documented Open-Meteo
    #   hallucination artifacts where the Folsom grid cell returns 5-6% RH.
    # WIND_SPEED_CAP_KMH = 25.0 — guards against documented artifacts where
    #   the API returns 30+ km/h for days that were observationally calm.

    Returns a copy of df_base with calibration and clamping applied.
    """

def _predict_single_horizon(
    df_h: pd.DataFrame,
    horizon_h: int,
    models: dict,
    current_aqi: int,
    min_ci_width_aqi: float,
) -> tuple[dict, float]:
    """
    Run feature engineering, model inference, CI enforcement for one horizon.

    This is a pure function: no side effects, no global state reads.
    All inputs are passed explicitly.

    Args:
        df_h: Prepared DataFrame (output of _prepare_horizon_dataframe).
        horizon_h: Forecast horizon in hours (6, 12, 24, 48).
        models: Dict keyed by horizon_h, each containing 'point', 'q05', 'q95'.
        current_aqi: Current AQI integer (used as fallback base).
        min_ci_width_aqi: Minimum CI width to enforce (monotonic uncertainty growth).
            Renamed from prev_width to express purpose: this is the floor on the
            confidence interval width, not a "previous" value.

    Returns:
        (forecast_entry, new_min_ci_width_aqi) where:
        - forecast_entry: dict with keys aqi, ci_lo, ci_hi, category, color, valid_at
        - new_min_ci_width_aqi: updated floor for the next horizon's CI width

    On any exception, logs at ERROR and returns a degraded entry
    (current_aqi, ci_lo=0, ci_hi=500) with the unchanged min_ci_width_aqi.
    """

def _assemble_forecast_result(
    current: dict,
    forecasts: dict,
    history: list,
    data_age_minutes: int,
    generated_at: datetime,
) -> dict:
    """
    Build the final JSON-serializable result dict matching the /forecast schema.

    Does not call any external services. Pure assembly from pre-computed parts.
    """
```

#### Refactored `predict_now()` shell

```python
def predict_now() -> dict:
    generated_at = datetime.now(tz=TZ)
    models = load_all_models()

    df, data_age_minutes = _fetch_input_data(past_hours=168)
    airnow = fetch_airnow_current()
    current = _resolve_current_conditions(df, airnow)
    df_prepared = _prepare_horizon_dataframe(df, airnow)

    forecasts = {}
    min_ci_width_aqi = 0.0
    for h in HORIZONS:
        entry, min_ci_width_aqi = _predict_single_horizon(
            df_prepared, h, models, current["aqi"], min_ci_width_aqi
        )
        forecasts[f"{h}h"] = entry

    history = _build_history_72h(df, models)
    result = _assemble_forecast_result(current, forecasts, history, data_age_minutes, generated_at)
    result["ai_summary"] = generate_summary(result)

    CACHE_FILE.write_text(json.dumps(result, indent=2, default=str))
    log.info("Forecast cached → %s", CACHE_FILE)
    return result
```

#### Symbol rename: `_safe_aqi_scalar` → `_extract_scalar_handling_dst_duplicates`

The new name explains *why* the function exists: `df.loc[ts, col]` returns a `pd.Series` when the index has duplicate timestamps, which occurs during DST spring-forward (March, ~1 row/year). The function takes `.iloc[0]` to handle this case.

---

### 4. `backend/api.py` — Scope Bug Fix + Health Signal

#### Scope bug fix (Requirement 4)

The `lifespan` context manager currently assigns `_models_loaded = False` as a local variable, shadowing the module-level variable. The fix:

```python
# Module level (already exists):
_models_loaded: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_loaded
    # CDN-proxy mode: ML inference is handled by the GitHub Actions worker.
    # Models are NOT loaded into Render's process memory because the 512MB
    # memory limit (Render free tier) causes R14 OOM restarts when all 12
    # LightGBM models (~20,000 trees each) are held in RAM simultaneously.
    _models_loaded = False
    yield
```

#### Staleness health signal (Requirement 12)

```python
# Maximum acceptable age of the cached forecast before signaling staleness.
# GitHub Actions runs every hour; 90 minutes allows one missed run before alerting.
STALE_FORECAST_THRESHOLD_MINUTES: int = 90

@app.get("/health")
async def health():
    cached = load_cached_forecast(prefer_remote=True)
    last_refresh = cached.get("generated_at", "never") if cached else "never"
    age = cache_age_minutes()
    return {
        "status":                    "ok",
        "models_loaded":             _models_loaded,
        "startup_time":              _startup_time,
        "last_refresh":              last_refresh,
        "cache_age_minutes":         age,
        "data_stale":                age > STALE_FORECAST_THRESHOLD_MINUTES,
        "stale_threshold_minutes":   STALE_FORECAST_THRESHOLD_MINUTES,
    }
```

HTTP status remains 200 even when `data_stale: true` (Requirement 12.4).

---

### 5. `backend/train.py` — Alignment Fix

Two line changes + one import change:

```python
# Before:
from features import engineer_features, get_feature_names
MODELS_DIR = Path("models")
OPTUNA_PARAMS_PATH = MODELS_DIR / "best_optuna_params.json"

# After:
from features_v6 import engineer_features, get_feature_names
MODELS_DIR = Path("models_v6")
OPTUNA_PARAMS_PATH = MODELS_DIR / "best_optuna_params.json"
```

The `OPTUNA_PARAMS_PATH` update is implicit since it references `MODELS_DIR`. No other changes to `train.py` are required.

---

### 6. `backend/ai_layer.py` — Consolidation

#### New function: `answer_question_with_key`

```python
def answer_question_with_key(
    question: str,
    forecast_data: dict,
    api_key: str,
) -> str:
    """
    Answer a user question using the Gemini REST API directly.

    This variant accepts the API key as a parameter so that callers
    (e.g., the Streamlit frontend) can pass a key sourced from their
    own secrets store (st.secrets) without this module needing to know
    about Streamlit's secrets API.

    Uses the same REST endpoint as the frontend's current _call_gemini():
      POST https://generativelanguage.googleapis.com/v1/models/
           gemini-2.0-flash:generateContent?key={api_key}

    Returns a user-facing error string on failure (never raises).
    """
```

**Implementation:** Extract the REST call logic from `frontend/app.py`'s `_call_gemini()` into this function. The system prompt used is `_SYSTEM_PROMPT` (already defined in `backend/ai_layer.py`). The `_build_context_block()` function (renamed to `_format_forecast_as_ai_context()` per Requirement 8.7) is reused.

**Rename:** `_build_context_block` → `_format_forecast_as_ai_context`

---

### 7. `frontend/app.py` — Dashboard Updates

#### AI layer consolidation (Requirement 6)

Remove from `frontend/app.py`:
- `_call_gemini()` function
- `_build_context()` function  
- `_GEMINI_ENDPOINT` constant
- `_GEMINI_BASE_SYSTEM` constant
- `_EXPERT_BLOCK` usage in AI calls

Add import:
```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from ai_layer import answer_question_with_key, generate_summary
```

Update `ask_ai()`:
```python
def ask_ai(question: str, data: dict) -> str:
    api_key = _get_gemini_key()
    if not api_key:
        return _answer_from_live_forecast_data(question, data)
    response = answer_question_with_key(question, data, api_key)
    if any(ind in response for ind in _API_ERROR_INDICATORS):
        return _answer_from_live_forecast_data(question, data)
    return response
```

#### Symbol renames (Requirement 8)

| Old name | New name |
|---|---|
| `_local_expert_answer()` | `_answer_from_live_forecast_data()` |
| `_build_expert_knowledge()` | `_build_model_accuracy_context()` |

#### Live accuracy metadata (Requirement 11)

New helper:
```python
def _get_horizon_accuracy(horizon_h: int, forecast_data: dict) -> dict:
    """
    Return {"mae": float, "r2": float} for a given horizon from live forecast metadata.
    Falls back to hardcoded defaults if model_metadata is absent.
    """
    FALLBACK = {6: {"mae": 3.5, "r2": 0.87}, 12: {"mae": 5.0, "r2": 0.75},
                24: {"mae": 7.0, "r2": 0.60}, 48: {"mae": 8.5, "r2": 0.50}}
    try:
        horizons = forecast_data["model_metadata"]["horizons"]
        for h in horizons:
            if h["horizon_h"] == horizon_h:
                return {"mae": h["val_mae"], "r2": h["val_r2"]}
    except (KeyError, TypeError):
        pass
    return FALLBACK.get(horizon_h, {"mae": None, "r2": None})
```

`_answer_from_live_forecast_data()` calls `_get_horizon_accuracy(6, data)` and `_get_horizon_accuracy(48, data)` instead of embedding numeric literals.

---

### 8. Dead Code Deletion Sequence (Requirement 7)

The deletion order is safety-critical. Follow this exact sequence:

**Step 1 — Update `train.py`** (Requirement 5 changes)
- Change import from `features` to `features_v6`
- Change `MODELS_DIR` to `Path("models_v6")`
- Verify `train.py` runs without error

**Step 2 — Delete `backend/features.py`**
- Pre-condition: `train.py` no longer imports from `features`
- Verify: `grep -r "from features import\|import features" backend/*.py` returns no hits in production files

**Step 3 — Update `frontend/app.py`** (Requirement 6 changes)
- Add import from `backend/ai_layer.py`
- Remove inline `_call_gemini`, `_build_context`, `_GEMINI_ENDPOINT`
- Verify Streamlit app loads without error

**Step 4 — Delete `frontend/ai_layer.py`**
- Pre-condition: `frontend/app.py` no longer imports from `frontend/ai_layer.py`
- Verify: `grep -r "from ai_layer import\|import ai_layer" frontend/*.py` returns no hits

**Step 5 — Delete remaining dead code files**

Files to delete from `backend/`:
- `app.py` (duplicate of frontend)
- `backtest.py`, `backtest_v5.py`, `backtest_v6.py`
- `features_v5.py`
- `audit2.py`, `audit_v9.py`
- `compare_v9.py`
- `diagnose.py`, `diagnose_backtest.py`, `diagnostics_v6.py`
- `debug_features.py`
- `check_deps.py`

Files to delete from `frontend/`:
- `old_app.py`
- `qr_generator.py` (not reachable from production entry point)

Note: Several files listed in Requirement 7.1 (`train_v5.py`, `train_v6.py`, `tune.py`, `tune_v5.py`, `tune_v6.py`, `validate.py`, `validate_v5.py`, `validate_v6.py`, `hpo_v9.py`, `inspect_imputer.py`, `inspect_model.py`, `new.py`, `regime_diagnostic.py`, `trace.py`, `vibe_v2_app_utf8.py`, `generate_excel.py`, `generate_report.py`, `make_plots.py`, `process_firms_history.py`) do not appear in the current workspace file tree and are presumed already deleted or never present. Delete any that do exist.

---

## Data Models

### Forecast Entry (unchanged)

```python
{
    "aqi":      int,        # 0–500
    "ci_lo":    int,        # lower confidence bound
    "ci_hi":    int,        # upper confidence bound
    "category": str,        # EPA category name
    "color":    str,        # hex color
    "valid_at": str,        # ISO 8601 datetime
}
```

### Health Response (extended)

```python
{
    "status":                  "ok",
    "models_loaded":           bool,
    "startup_time":            str,
    "last_refresh":            str,
    "cache_age_minutes":       int,
    "data_stale":              bool,   # NEW
    "stale_threshold_minutes": int,    # NEW
}
```

### Logger Record Format

```
2025-07-15T14:32:01 INFO     backend.inference — Fetching recent data...
2025-07-15T14:32:03 INFO     backend.data_fetcher — Recent combined: 240 rows
2025-07-15T14:32:05 ERROR    backend.inference — ERROR for 48h horizon: <class 'KeyError'>: 'regime'
```

---

