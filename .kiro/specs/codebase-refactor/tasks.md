# Implementation Plan: Codebase Refactor

## Overview

Structural refactor of the Folsom AQI Forecast system across 10 phases. All changes are renames, extractions, or deletions — no logic changes. Behavioral equivalence of `engineer_features()` and `predict_now()` is non-negotiable. Follow the phase order strictly; the dead-code deletion sequence in Phase 10 has safety-critical pre-conditions.

## Tasks

- [x] 1. Phase 1 — Create shared logging module
  - [x] 1.1 Create `backend/logger.py` with `get_logger(name)` factory
    - Implement `_LevelFilter` inner class routing DEBUG/INFO to stdout and WARNING+ to stderr
    - Guard against duplicate handlers with `if not logger.handlers:`
    - Default level from `LOG_LEVEL` env var, falling back to `INFO`
    - Format: `"%(asctime)s %(levelname)-8s %(name)s — %(message)s"`, date `"%Y-%m-%dT%H:%M:%S"`
    - _Requirements: 1.1, 1.2, 1.3, 1.10_

- [x] 2. Phase 2 — Decompose `backend/features_v6.py`
  - [x] 2.1 Add module-level named constants replacing all magic numbers
    - Add `VENT_THRESHOLD_M2_PER_S`, `COLD_SMOKE_THRESHOLD_C`, `MAGNUS_A`, `MAGNUS_B`, `HUMIDITY_CLAMP_MIN_PCT`, `WIND_SPEED_CAP_KMH`, `RAIN_THRESHOLD_MM` with explanatory inline comments
    - _Requirements: 3.3, 9.3, 9.4_
  - [x] 2.2 Extract `_add_aqi_lag_features(X, df)` sub-function
    - Move Groups 1–2 (AQI lags, diffs, rolling stats, EWMA) out of `engineer_features()` into this private function
    - Function mutates `X` in-place; preserve any `X.copy()` defragmentation calls at the same logical points
    - _Requirements: 3.1, 3.6_
  - [x] 2.3 Extract `_add_pm25_and_combustion_features(X, df)` sub-function
    - Move Group 3 + Feature Group 1 (PM2.5, CO, dust, NO₂) into this private function
    - _Requirements: 3.1, 3.6_
  - [x] 2.4 Extract `_add_meteorological_features(X, df, horizon_h)` sub-function
    - Move Group 4 (BLH, wind, pressure, humidity, temp, forward NWP features) into this private function
    - Add inline comment on forward-shifted NWP block explaining training/inference symmetry
    - _Requirements: 3.1, 3.6, 9.5_
  - [x] 2.5 Extract `_add_atmospheric_stability_features(X, df, horizon_h)` sub-function
    - Move Group 5 sub-groups 5a–5f (stagnation, inversion, ventilation deficit, synoptic blocking, cold degree hours, tule fog) into this private function
    - Rename local variable `blh` → `boundary_layer_height` within this function
    - _Requirements: 3.1, 3.6, 8.4_
  - [x] 2.6 Extract `_add_wildfire_features(X, df, horizon_h)` sub-function
    - Move wildfire proxy features (HDWI, FIRMS FRP, advection score) into this private function
    - _Requirements: 3.1, 3.6_
  - [x] 2.7 Extract `_add_temporal_features(X, df, horizon_h)` sub-function
    - Move Group 6 (cyclic hour/DOW/DOY encodings, commute harmonics, future hour) into this private function
    - _Requirements: 3.1, 3.6_
  - [x] 2.8 Extract `_add_regulatory_features(X, df)` sub-function
    - Move Group 7 (CBYB season flag, weekend burning proxy, radiation accumulation, V8/V9 features) into this private function
    - _Requirements: 3.1, 3.6_
  - [x] 2.9 Refactor `engineer_features()` to 7-line orchestration shell
    - Replace inline feature computation with sequential calls to the 7 sub-functions above
    - Orchestrator calls `X = X.copy()` after each sub-function to preserve defragmentation behavior
    - _Requirements: 3.2, 3.6_
  - [x] 2.10 Annotate `es`, `ea`, `vpd` with Magnus formula explanation
    - Add inline comment block explaining the Magnus formula derivation where these abbreviations are first used, or rename to `saturation_vapor_pressure_kpa`, `actual_vapor_pressure_kpa`, `vapor_pressure_deficit_kpa`
    - _Requirements: 3.4, 8.3_

- [x] 3. Phase 3 — Decompose `backend/inference.py`
  - [x] 3.1 Add module-level constants `HUMIDITY_FLOOR_PCT` and `WIND_SPEED_CAP_KMH`
    - Add both constants with inline comments explaining the Open-Meteo Folsom grid cell hallucination artifacts they guard against
    - _Requirements: 2.4, 9.2_
  - [x] 3.2 Extract `_fetch_input_data(past_hours)` sub-function
    - Move data fetch + data-age computation out of `predict_now()` into this function
    - Signature: `(past_hours: int = 168) -> tuple[pd.DataFrame, int]`
    - _Requirements: 2.1, 10.3_
  - [x] 3.3 Extract `_resolve_current_conditions(df, airnow)` sub-function
    - Move AirNow vs Open-Meteo priority logic into this function
    - Signature: `(df: pd.DataFrame, airnow: dict | None) -> dict`
    - Must return a valid dict when `airnow=None` (sourced from DataFrame)
    - _Requirements: 2.1, 10.8_
  - [x] 3.4 Extract `_prepare_horizon_dataframe(df_base, airnow)` sub-function
    - Move AirNow Uniform Offset Calibration and sensor sanity clamping into this function
    - Add inline comment explaining why uniform offset is used instead of point injection
    - _Requirements: 2.1, 9.1_
  - [x] 3.5 Extract `_predict_single_horizon(df_h, horizon_h, models, current_aqi, min_ci_width_aqi)` sub-function
    - Move feature engineering call, model inference, and CI enforcement into this function
    - Catch all exceptions, log at ERROR, return degraded entry without propagating
    - Rename parameter `prev_width` → `min_ci_width_aqi`
    - _Requirements: 2.1, 2.3, 2.5, 8.2_
  - [x] 3.6 Extract `_assemble_forecast_result(current, forecasts, history, data_age_minutes, generated_at)` sub-function
    - Move final JSON-serializable result assembly into this pure function
    - _Requirements: 2.1_
  - [x] 3.7 Refactor `predict_now()` to orchestration shell
    - Replace inline logic with sequential calls to the 5 sub-functions above
    - Shell should match the design's ~10-line target
    - _Requirements: 2.2_
  - [x] 3.8 Rename `_safe_aqi_scalar` → `_extract_scalar_handling_dst_duplicates`
    - Update the function definition and all call sites (including the inline comment in `_build_history_72h()` explaining the DST spring-forward scenario)
    - _Requirements: 8.1, 9.7_
  - [x] 3.9 Replace all `print()` calls in `inference.py` with structured logger calls
    - Add `from logger import get_logger` and `log = get_logger(__name__)` at module top
    - Map INFO-level prints to `log.info()`, ERROR-level to `log.error()`, CRITICAL to `log.critical()`
    - Add `exc_info=True` on ERROR/CRITICAL calls that catch exceptions
    - _Requirements: 1.5, 1.9_

- [x] 4. Phase 4 — Fix `backend/api.py`
  - [x] 4.1 Fix `_models_loaded` scope bug in `lifespan`
    - Add `global _models_loaded` inside the `lifespan` context manager before the assignment
    - Add inline comment explaining CDN-proxy mode and Render R14 OOM constraint
    - _Requirements: 4.1, 4.2, 4.4, 9.6_
  - [x] 4.2 Add `STALE_FORECAST_THRESHOLD_MINUTES = 90` constant and update `/health` endpoint
    - Add module-level constant with inline comment (one missed GitHub Actions run = 90 min)
    - Add `data_stale` and `stale_threshold_minutes` fields to the health response dict
    - HTTP status remains 200 regardless of staleness
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  - [x] 4.3 Replace all `print()` calls in `api.py` with structured logger calls
    - Add `from logger import get_logger` and `log = get_logger(__name__)` at module top
    - _Requirements: 1.7_

- [x] 5. Phase 5 — Align `backend/train.py`
  - [x] 5.1 Update `train.py` imports and `MODELS_DIR`
    - Change `from features import ...` → `from features_v6 import ...`
    - Change `MODELS_DIR = Path("models")` → `MODELS_DIR = Path("models_v6")`
    - Remove any remaining references to the `features` module
    - _Requirements: 5.1, 5.2, 5.4, 5.5_
  - [x] 5.2 Replace all `print()` calls in `train.py` with structured logger calls
    - Add `from logger import get_logger` and `log = get_logger(__name__)` at module top
    - _Requirements: 1.8_

- [x] 6. Phase 6 — Consolidate `backend/ai_layer.py`
  - [x] 6.1 Rename `_build_context_block` → `_format_forecast_as_ai_context`
    - Update the function definition and all internal call sites within `ai_layer.py`
    - _Requirements: 8.7, 8.8_
  - [x] 6.2 Add `answer_question_with_key(question, forecast_data, api_key)` function
    - Extract REST call logic from `frontend/app.py`'s `_call_gemini()` into this new function
    - Accepts `api_key` as parameter so callers can pass Streamlit-secrets-sourced keys
    - Uses `_format_forecast_as_ai_context()` internally; returns user-facing error string on failure (never raises)
    - _Requirements: 6.3_
  - [x] 6.3 Replace all `print()` calls in `ai_layer.py` with structured logger calls
    - Add `from logger import get_logger` and `log = get_logger(__name__)` at module top
    - _Requirements: 1.6_

- [x] 7. Phase 7 — Replace `print()` in `backend/data_fetcher.py`
  - [x] 7.1 Replace all `print()` calls in `data_fetcher.py` with structured logger calls
    - Add `from logger import get_logger` and `log = get_logger(__name__)` at module top
    - Map cache-hit and successful-fetch prints to `log.info()`, missing-key warnings to `log.warning()`, fetch failures to `log.error()` with `exc_info=True`
    - _Requirements: 1.4, 1.9_

- [x] 8. Phase 8 — Replace `print()` in `backend/refresh.py`
  - [x] 8.1 Replace all `print()` calls in `refresh.py` with structured logger calls
    - Add `from logger import get_logger` and `log = get_logger(__name__)` at module top
    - Remove `flush=True` arguments (logger flushes handlers automatically)
    - _Requirements: 1.8_

- [x] 9. Phase 9 — Update `frontend/app.py`
  - [x] 9.1 Add `_get_horizon_accuracy(horizon_h, forecast_data)` helper
    - Reads `{"mae": float, "r2": float}` from `forecast_data["model_metadata"]["horizons"]`
    - Falls back to hardcoded defaults when `model_metadata` is absent or malformed
    - _Requirements: 11.3_
  - [x] 9.2 Rename `_local_expert_answer` → `_answer_from_live_forecast_data`
    - Update function definition and all call sites in `frontend/app.py`
    - _Requirements: 8.5, 8.8_
  - [x] 9.3 Rename `_build_expert_knowledge` → `_build_model_accuracy_context`
    - Update function definition and all call sites in `frontend/app.py`
    - _Requirements: 8.6, 8.8_
  - [x] 9.4 Update `_answer_from_live_forecast_data` to use `_get_horizon_accuracy()`
    - Replace hardcoded accuracy strings (e.g., `"≈3.5 AQI units"`, `"0.87"`) with values from `_get_horizon_accuracy(6, data)` and `_get_horizon_accuracy(48, data)`
    - _Requirements: 11.1, 11.2, 11.4_
  - [x] 9.5 Remove inline AI definitions and add import from `backend/ai_layer`
    - Remove `_call_gemini()`, `_build_context()`, `_GEMINI_ENDPOINT`, `_GEMINI_BASE_SYSTEM` from `frontend/app.py`
    - Add `sys.path.insert` and `from ai_layer import answer_question_with_key, generate_summary`
    - Update `ask_ai()` to call `answer_question_with_key(question, data, api_key)`
    - _Requirements: 6.1, 6.2, 6.4, 6.5_

- [x] 10. Checkpoint — Verify all phases 1–9 are complete before deletion
  - Confirm `train.py` imports from `features_v6` (pre-condition for deleting `features.py`)
  - Confirm `frontend/app.py` imports from `backend/ai_layer` (pre-condition for deleting `frontend/ai_layer.py`)
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Phase 10 — Delete dead code (safety-critical order)
  - [x] 11.1 Delete `backend/features.py`
    - Pre-condition: `train.py` no longer imports from `features`
    - _Requirements: 7.1, 7.4, 7.5_
  - [x] 11.2 Delete `frontend/ai_layer.py`
    - Pre-condition: `frontend/app.py` no longer imports from `frontend/ai_layer.py`
    - _Requirements: 6.6, 7.3, 7.4_
  - [x] 11.3 Delete `backend/app.py`
    - _Requirements: 7.1_
  - [x] 11.4 Delete `frontend/old_app.py`
    - _Requirements: 7.2_
  - [x] 11.5 Delete `frontend/qr_generator.py`
    - _Requirements: 7.1_
  - [x] 11.6 Delete backend backtest and audit files
    - Delete: `backtest.py`, `backtest_v5.py`, `backtest_v6.py`, `audit2.py`, `audit_v9.py`, `compare_v9.py`
    - _Requirements: 7.1_
  - [x] 11.7 Delete backend diagnostic and debug files
    - Delete: `diagnose.py`, `diagnose_backtest.py`, `diagnostics_v6.py`, `debug_features.py`, `check_deps.py`
    - _Requirements: 7.1_
  - [x] 11.8 Delete remaining versioned and utility dead code files
    - Delete from `backend/`: `features_v5.py`, and any that exist from: `train_v5.py`, `train_v6.py`, `tune.py`, `tune_v5.py`, `tune_v6.py`, `validate.py`, `validate_v5.py`, `validate_v6.py`, `hpo_v9.py`, `inspect_imputer.py`, `inspect_model.py`, `new.py`, `regime_diagnostic.py`, `trace.py`, `vibe_v2_app_utf8.py`, `generate_excel.py`, `generate_report.py`, `make_plots.py`, `process_firms_history.py`, `qr_generator.py`
    - _Requirements: 7.1_

- [x] 12. Final checkpoint — Verify clean state
  - Confirm no production file imports from any deleted module
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The deletion sequence in Phase 10 (task 11) has hard pre-conditions — do not reorder
- Behavioral equivalence of `engineer_features()` is non-negotiable (Requirement 3.6); verify output is bit-identical before and after decomposition
- The `X.copy()` defragmentation calls must be preserved at the same logical points after each sub-function call in the orchestrator
- All `print()` replacements must use `exc_info=True` on any call that catches an exception (Requirement 1.9)
