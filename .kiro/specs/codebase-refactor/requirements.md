# Requirements Document

## Introduction

This spec defines a Principal Software Engineer-level refactor of the Folsom AQI Forecast system — a FastAPI + LightGBM ML pipeline (backend) and a Streamlit dashboard (frontend). The codebase has accumulated significant technical debt through iterative experimentation: dead code from versioned development scripts, a monolithic inference function, unstructured logging, duplicated modules across repos, and untestable units. The refactor targets seven quality pillars — Reduced Cognitive Load, High Changeability, Resilience, Observability, Self-Documenting Intent, Testability, and Code Hygiene & Lean Execution — without altering any externally visible API contract or model behavior.

**Scope boundary:** Only the production runtime files listed below are in scope for modification. All other files are candidates for deletion (see Requirement 7).

Production runtime files:
- `backend/api.py`, `backend/inference.py`, `backend/data_fetcher.py`, `backend/features_v6.py`, `backend/ai_layer.py`, `backend/refresh.py`, `backend/train.py`
- `frontend/app.py`, `frontend/ai_layer.py`
- `backend/requirements.txt`, `frontend/requirements.txt`, `backend/Procfile`, `backend/.env`
- `backend/models/`, `backend/models_v6/`, `backend/data/` (preserved as-is, not modified)

---

## Glossary

- **System**: The Folsom AQI Forecast system as a whole (backend + frontend).
- **Backend**: The FastAPI service deployed on Render (`backend/`).
- **Frontend**: The Streamlit dashboard deployed on Streamlit Cloud (`frontend/`).
- **Inference_Pipeline**: The module responsible for loading models, fetching data, engineering features, and producing the forecast JSON (`backend/inference.py`).
- **Feature_Engineer**: The module that transforms raw meteorological and AQI data into the ML feature matrix (`backend/features_v6.py`).
- **Data_Fetcher**: The module that retrieves AQ and weather data from Open-Meteo, AirNow, and NASA FIRMS APIs (`backend/data_fetcher.py`).
- **AI_Layer**: The module that wraps Gemini API calls for forecast summaries and Q&A (`backend/ai_layer.py` and `frontend/ai_layer.py`).
- **API_Server**: The FastAPI application that serves cached forecast data (`backend/api.py`).
- **Refresh_Runner**: The cron entry point that triggers `predict_now()` (`backend/refresh.py`).
- **Training_Pipeline**: The offline script that trains the 12 LightGBM models (`backend/train.py`).
- **Dashboard**: The Streamlit UI that displays forecasts and hosts the Navigator AI chat (`frontend/app.py`).
- **Logger**: A structured logging utility that replaces all `print()` calls in the production runtime.
- **Dead_Code**: Any file, function, class, import, or branch that is not reachable from the production runtime entry points (`api.py`, `refresh.py`, `frontend/app.py`).
- **Magic_Number**: A numeric literal embedded directly in logic without a named constant or inline explanation.
- **EARS**: Easy Approach to Requirements Syntax — the pattern language used for all acceptance criteria in this document.

---

## Requirements

### Requirement 1: Structured Logging Standard

**User Story:** As a developer on-call, I want all runtime log output to carry a consistent level, module tag, and timestamp, so that I can filter and triage issues in production without grepping through mixed stdout/stderr streams.

#### Acceptance Criteria

1. THE Logger SHALL emit every log entry as a single line containing: ISO 8601 timestamp, log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`), module name, and message text.
2. THE Logger SHALL write `INFO` and below to stdout and `WARNING` and above to stderr, matching the existing behavior of the `print()` calls it replaces.
3. WHEN a module is imported, THE Logger SHALL be configured via Python's `logging` standard library using a shared `get_logger(name)` factory function defined in a single `backend/logger.py` module.
4. THE Data_Fetcher SHALL replace every `print(...)` and `print(..., file=sys.stderr)` call with a Logger call at the appropriate level (`INFO` for cache hits and successful fetches, `WARNING` for missing API keys, `ERROR` for fetch failures and stale-cache fallbacks).
5. THE Inference_Pipeline SHALL replace every `print(...)` and `print(..., file=sys.stderr)` call with a Logger call at the appropriate level (`INFO` for normal progress, `ERROR` for horizon failures, `CRITICAL` for data fetch failures).
6. THE AI_Layer SHALL replace every `print(...)` and `print(..., file=sys.stderr)` call with a Logger call at the appropriate level (`INFO` for successful generation, `ERROR` for API failures).
7. THE API_Server SHALL replace every `print(...)` and `print(..., file=sys.stderr)` call with a Logger call at the appropriate level.
8. THE Refresh_Runner SHALL replace every `print(...)` and `flush=True` call with a Logger call; the Logger SHALL flush handlers automatically.
9. IF a log entry is emitted at `ERROR` or `CRITICAL` level, THEN THE Logger SHALL include the exception type and message in the log line.
10. THE Logger factory function SHALL accept an optional `level` parameter that defaults to the value of the `LOG_LEVEL` environment variable, falling back to `INFO` if the variable is unset.

---

### Requirement 2: Decompose the Monolithic `predict_now()` Function

**User Story:** As a developer, I want `predict_now()` broken into focused, named sub-functions, so that I can read, test, and modify each stage of the inference pipeline independently without holding the entire 200-line function in my head.

#### Acceptance Criteria

1. THE Inference_Pipeline SHALL decompose `predict_now()` into at minimum the following named sub-functions, each with a single responsibility:
   - `_fetch_input_data(past_hours: int) -> tuple[pd.DataFrame, int]` — fetches recent combined data and returns `(df, data_age_minutes)`.
   - `_resolve_current_conditions(df: pd.DataFrame, airnow: dict | None) -> dict` — determines the current AQI reading and its source.
   - `_prepare_horizon_dataframe(df_base: pd.DataFrame, airnow: dict | None) -> pd.DataFrame` — applies AirNow offset calibration and sensor sanity clamping.
   - `_predict_single_horizon(df_h: pd.DataFrame, horizon_h: int, models: dict, current_aqi: int, prev_width: float) -> tuple[dict, float]` — runs feature engineering, model inference, CI enforcement, and returns `(forecast_entry, new_width)`.
   - `_assemble_forecast_result(current: dict, forecasts: dict, history: list, data_age: int) -> dict` — builds the final JSON-serializable result dict.
2. THE `predict_now()` function SHALL be reduced to an orchestration shell that calls the sub-functions above in sequence, with no inline business logic of its own.
3. WHEN `_predict_single_horizon()` raises an exception, THE Inference_Pipeline SHALL catch it, log it at `ERROR` level, and return a degraded forecast entry (current AQI, full CI range) without propagating the exception to the caller.
4. THE `_prepare_horizon_dataframe()` function SHALL document the two named constants it uses — `HUMIDITY_FLOOR_PCT = 25.0` and `WIND_SPEED_CAP_KMH = 25.0` — as module-level named constants with inline comments explaining the physical rationale.
5. THE `prev_width` variable SHALL be renamed to `min_ci_width_aqi` to describe its purpose (enforcing monotonically increasing uncertainty) rather than its implementation.

---

### Requirement 3: Decompose `engineer_features()` into Named Sub-Functions

**User Story:** As a developer, I want the 400-line `engineer_features()` function broken into named sub-functions grouped by feature family, so that I can locate, understand, and modify a specific feature group without reading the entire function.

#### Acceptance Criteria

1. THE Feature_Engineer SHALL decompose `engineer_features()` into named private sub-functions, each responsible for exactly one feature group, including at minimum:
   - `_add_aqi_lag_features(X, df)` — AQI lags, diffs, rolling stats, EWMA (Groups 1–2).
   - `_add_pm25_and_combustion_features(X, df)` — PM2.5, CO, dust, NO₂ (Group 3 + Feature Group 1).
   - `_add_meteorological_features(X, df, horizon_h)` — BLH, wind, pressure, humidity, temperature, forward-shifted NWP features (Group 4).
   - `_add_atmospheric_stability_features(X, df, horizon_h)` — stagnation index, inversion proxy, ventilation deficit, synoptic blocking, cold degree hours, tule fog precursor (Group 5).
   - `_add_wildfire_features(X, df, horizon_h)` — HDWI, FIRMS FRP, advection score (Groups 5 wildfire + Group 8).
   - `_add_temporal_features(X, df, horizon_h)` — cyclic hour/DOW/DOY encodings, commute harmonics, future hour (Group 6).
   - `_add_regulatory_features(X, df)` — CBYB season flag, weekend burning proxy, radiation accumulation (Group 7).
2. THE `engineer_features()` function SHALL be reduced to a sequential call to each sub-function above, with no inline feature computation of its own.
3. THE Feature_Engineer SHALL define all Magic_Numbers as named module-level constants with inline comments explaining their physical meaning, including at minimum:
   - `VENT_THRESHOLD_M2_PER_S = 3000.0` — empirical Central Valley ventilation threshold.
   - `COLD_SMOKE_THRESHOLD_C = 7.0` — temperature below which residential wood burning begins.
   - `MAGNUS_A = 17.27` and `MAGNUS_B = 237.3` — Magnus formula coefficients for saturation vapor pressure.
   - `HUMIDITY_CLAMP_MIN_PCT = 25.0` — Open-Meteo hallucination floor for relative humidity.
   - `WIND_SPEED_CAP_KMH = 25.0` — Open-Meteo hallucination cap for wind speed.
   - `RAIN_THRESHOLD_MM = 0.1` — minimum precipitation to count as a wet-deposition event.
4. THE Feature_Engineer SHALL replace the variable names `es`, `ea`, and `vpd` with `saturation_vapor_pressure_kpa`, `actual_vapor_pressure_kpa`, and `vapor_pressure_deficit_kpa` respectively, or add an inline comment block explaining the Magnus formula derivation where these abbreviations are first used.
5. THE Feature_Engineer SHALL replace all uses of the loop variable `h` (when used as both a horizon variable and a generic loop counter) with `horizon_h` consistently throughout the module.
6. WHEN `engineer_features()` is called, THE Feature_Engineer SHALL return the same feature matrix as the current implementation for any given input DataFrame and horizon value (behavioral equivalence).

---

### Requirement 4: Fix the `api.py` Lifespan Scope Bug

**User Story:** As a developer, I want the `_models_loaded` flag in `api.py` to accurately reflect the server's state, so that the `/health` endpoint does not silently report stale information.

#### Acceptance Criteria

1. THE API_Server SHALL declare `_models_loaded` as a module-level variable (not a local variable inside the `lifespan` context manager), so that assignments to it are visible to the `/health` endpoint handler.
2. WHEN the `lifespan` context manager sets `_models_loaded = False`, THE API_Server SHALL use the `global` keyword to ensure the assignment updates the module-level variable.
3. THE `/health` endpoint SHALL return the value of the module-level `_models_loaded` variable.
4. THE API_Server SHALL add an inline comment on the `_models_loaded = False` assignment explaining that CDN-proxy mode intentionally keeps this `False` and why (Render memory constraint).

---

### Requirement 5: Decouple `train.py` from the Superseded `features.py`

**User Story:** As a developer, I want `train.py` to use `features_v6` — the same feature engineering module used by `inference.py` — so that the training and inference pipelines are guaranteed to produce identical feature sets.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL import `engineer_features` and `get_feature_names` from `features_v6`, not from `features`.
2. THE Training_Pipeline SHALL save model artifacts to `models_v6/` (matching the directory used by `inference.py`), not to `models/`.
3. WHEN `train.py` is executed, THE Training_Pipeline SHALL produce a feature matrix with the same column names and order as the feature matrix produced by `inference.py` for the same input data.
4. THE Training_Pipeline SHALL remove the import of `features.get_feature_names` and any reference to the `features` module.
5. THE Training_Pipeline SHALL update the `MODELS_DIR` constant from `Path("models")` to `Path("models_v6")`.

---

### Requirement 6: Eliminate Duplicated AI Layer Logic

**User Story:** As a developer, I want a single canonical `ai_layer.py` so that changes to the Gemini prompt, model name, or error handling only need to be made in one place.

#### Acceptance Criteria

1. THE System SHALL maintain exactly one canonical `ai_layer.py` — the backend version at `backend/ai_layer.py`.
2. THE Dashboard SHALL import `generate_summary` and `answer_question` from `ai_layer` rather than re-implementing Gemini API calls inline.
3. THE AI_Layer SHALL expose a `answer_question_with_key(question: str, forecast_data: dict, api_key: str) -> str` variant that accepts the API key as a parameter, so the Dashboard can pass the Streamlit-secrets-sourced key without the AI_Layer needing to know about `st.secrets`.
4. THE Dashboard SHALL remove the inline `_call_gemini()`, `_build_context()`, and `_GEMINI_ENDPOINT` definitions and replace them with calls to the shared AI_Layer functions.
5. THE Dashboard SHALL retain the `_local_expert_answer()` fallback function as a Dashboard-only concern (it depends on `ADVISORIES` and Streamlit-specific data structures).
6. WHEN `frontend/ai_layer.py` is removed, THE System SHALL continue to function correctly because the Dashboard imports from the shared backend module.

---

### Requirement 7: Remove All Dead Code

**User Story:** As a developer, I want all files and code that are not part of the production runtime removed from the repository, so that I can navigate the codebase without being misled by obsolete experiments.

#### Acceptance Criteria

1. THE System SHALL delete the following files from `backend/` as they are confirmed dead code not reachable from any production entry point:
   - `features.py` (superseded by `features_v6.py`)
   - `app.py` (duplicate of `frontend/app.py`, not used by the backend)
   - `backtest.py`, `backtest_v5.py`, `backtest_v6.py`
   - `features_v5.py`
   - `train_v5.py`, `train_v6.py`
   - `tune.py`, `tune_v5.py`, `tune_v6.py`
   - `validate.py`, `validate_v5.py`, `validate_v6.py`
   - `audit2.py`, `audit_v9.py`
   - `compare_v9.py`
   - `diagnose.py`, `diagnose_backtest.py`, `diagnostics_v6.py`
   - `debug_features.py`
   - `hpo_v9.py`
   - `inspect_imputer.py`, `inspect_model.py`
   - `new.py`
   - `regime_diagnostic.py`
   - `trace.py`
   - `vibe_v2_app_utf8.py`
   - `check_deps.py`, `generate_excel.py`, `generate_report.py`, `make_plots.py`, `process_firms_history.py`, `qr_generator.py`
2. THE System SHALL delete `frontend/old_app.py` as it is explicitly superseded dead code.
3. THE System SHALL delete `frontend/ai_layer.py` after the Dashboard is updated to import from the shared backend AI_Layer (per Requirement 6).
4. WHEN any of the above files are deleted, THE System SHALL verify that no production runtime file imports from the deleted file before deletion.
5. THE `backend/features.py` file SHALL NOT be deleted until `train.py` has been updated to import from `features_v6` (per Requirement 5), to avoid breaking the Training_Pipeline during the transition.
6. THE System SHALL remove all unused imports from each production runtime file after dead code deletion, including but not limited to any `from features import ...` remaining in `train.py` after migration.

---

### Requirement 8: Rename Symbols to Express Intent

**User Story:** As a developer reading the code for the first time, I want variable and function names to explain *why* they exist, not just *what* they compute, so that I can understand the business logic without needing to trace execution.

#### Acceptance Criteria

1. THE Inference_Pipeline SHALL rename `_safe_aqi_scalar()` to `_extract_scalar_handling_dst_duplicates()` or an equivalent name that explains the DST-duplicate-timestamp reason for its existence, not just its safe-extraction behavior.
2. THE Inference_Pipeline SHALL rename `prev_width` to `min_ci_width_aqi` (see also Requirement 2.5).
3. THE Feature_Engineer SHALL rename or annotate `es`, `ea`, and `vpd` as specified in Requirement 3.4.
4. THE Feature_Engineer SHALL rename the variable `blh` to `boundary_layer_height` wherever it is used as a local variable (distinct from the DataFrame column name `boundary_layer_height`), to eliminate the inconsistency between `blh` and `boundary_layer_height` usages.
5. THE Dashboard SHALL rename `_local_expert_answer()` to `_answer_from_live_forecast_data()` to express that this function answers questions deterministically from the live forecast payload, not from a static knowledge base.
6. THE Dashboard SHALL rename `_build_expert_knowledge()` to `_build_model_accuracy_context()` to express that this function builds a context block about model accuracy metrics, not general expert knowledge.
7. THE AI_Layer SHALL rename `_build_context_block()` to `_format_forecast_as_ai_context()` to express that this function formats the forecast dict into a string suitable for AI prompt injection.
8. WHEN a symbol is renamed, THE System SHALL update all call sites and references in production runtime files consistently.

---

### Requirement 9: Add Inline Documentation for Non-Obvious Business Logic

**User Story:** As a developer, I want comments to explain *why* non-obvious decisions were made, not *what* the code does, so that I don't accidentally revert intentional design choices.

#### Acceptance Criteria

1. THE Inference_Pipeline SHALL add an inline comment on the AirNow Uniform Offset Calibration block explaining why a uniform offset is used instead of a point injection (to avoid creating unphysical differential spikes in rolling features).
2. THE Inference_Pipeline SHALL add an inline comment on the `HUMIDITY_FLOOR_PCT` and `WIND_SPEED_CAP_KMH` clamps explaining that these guard against documented Open-Meteo hallucination artifacts for the Folsom grid cell.
3. THE Feature_Engineer SHALL add an inline comment on the `VENT_THRESHOLD_M2_PER_S` constant explaining that 3000 m²/s is the empirical Central Valley stagnation threshold used by the California Air Resources Board.
4. THE Feature_Engineer SHALL add an inline comment on the `COLD_SMOKE_THRESHOLD_C` constant explaining that 7°C (45°F) is the temperature below which residential wood burning in Folsom increases significantly, based on local emission inventory data.
5. THE Feature_Engineer SHALL add an inline comment on the forward-shifted NWP feature block explaining the training/inference symmetry: during training `.shift(-horizon_h)` uses actual future weather as a proxy; during inference the DataFrame already contains genuine NWP forecast rows.
6. THE API_Server SHALL add an inline comment on the CDN-proxy mode lifespan block explaining the Render R14 OOM constraint that motivated removing in-process model loading.
7. THE `_build_history_72h()` function SHALL add an inline comment on the `_safe_aqi_scalar` call explaining the DST spring-forward duplicate-index scenario that this guard handles.
8. THE Training_Pipeline SHALL add an inline comment on the early-stopping cutoff logic explaining the three-zone timeline: fit data, ES eval window, and the validate.py holdout window.
9. THE System SHALL NOT add comments that merely restate what the code does (e.g., `# increment counter` above `i += 1`).

---

### Requirement 10: Make Every Production Module Unit-Testable

**User Story:** As a developer, I want every production module to be testable in isolation with mocked dependencies, so that I can write fast, deterministic unit tests without hitting live APIs or loading model files from disk.

#### Acceptance Criteria

1. THE Feature_Engineer SHALL accept only a `pd.DataFrame` and `horizon_h: int` as inputs to `engineer_features()`, with no file I/O, network calls, or global state reads inside the function — this is already true and SHALL be preserved through the refactor.
2. THE Data_Fetcher SHALL expose all external API calls through functions that accept the endpoint URL as a parameter (defaulting to the production URL), so that tests can inject a mock URL or monkeypatch `requests.get`.
3. THE Inference_Pipeline SHALL expose `_fetch_input_data`, `_resolve_current_conditions`, `_prepare_horizon_dataframe`, and `_predict_single_horizon` as importable module-level functions (not nested closures), so that each can be tested independently with a synthetic DataFrame and a mock model object.
4. THE AI_Layer SHALL expose `_get_model()` as a function that can be monkeypatched in tests, so that `generate_summary()` and `answer_question()` can be tested without a live `GEMINI_API_KEY`.
5. THE API_Server SHALL not call `load_all_models()` or `predict_now()` at import time; all model loading SHALL remain inside the `lifespan` context manager so that the FastAPI app can be imported in tests without triggering disk I/O.
6. THE Training_Pipeline SHALL separate the data-fetching step from the feature-engineering and model-training steps so that `train_horizon()` can be called in tests with a pre-built synthetic DataFrame without triggering a network fetch.
7. WHEN `engineer_features()` is called with a synthetic DataFrame containing all required columns, THE Feature_Engineer SHALL return a non-empty feature matrix with no unhandled exceptions.
8. WHEN `_resolve_current_conditions()` is called with a DataFrame and `airnow=None`, THE Inference_Pipeline SHALL return a current-conditions dict sourced from the DataFrame without raising an exception.

---

### Requirement 11: Enforce Single Responsibility in `_local_expert_answer()`

**User Story:** As a developer, I want the hardcoded accuracy strings in `_local_expert_answer()` to be sourced from the live model metadata, so that they do not go stale when models are retrained.

#### Acceptance Criteria

1. THE Dashboard SHALL replace the hardcoded accuracy strings in `_local_expert_answer()` (e.g., `"≈3.5 AQI units"`, `"0.87"`, `"≈8.5 AQI units"`, `"0.50"`) with values read from the `model_metadata` field of the live forecast payload.
2. WHEN the `model_metadata` field is absent or empty in the forecast payload, THE Dashboard SHALL fall back to the current hardcoded strings rather than displaying `None` or crashing.
3. THE Dashboard SHALL extract the accuracy lookup into a helper function `_get_horizon_accuracy(horizon_h: int, forecast_data: dict) -> dict` that returns `{"mae": float, "r2": float}` for a given horizon, sourced from `forecast_data["model_metadata"]["horizons"]`.
4. THE `_local_expert_answer()` function SHALL call `_get_horizon_accuracy()` rather than embedding numeric literals directly in the answer strings.

---

### Requirement 12: Data Staleness Health Signal

**User Story:** As an operator, I want the `/health` endpoint to signal when forecast data is stale beyond an acceptable threshold, so that I can detect silent failures in the GitHub Actions refresh pipeline.

#### Acceptance Criteria

1. THE API_Server SHALL define a named constant `STALE_FORECAST_THRESHOLD_MINUTES = 90` representing the maximum acceptable age of the cached forecast.
2. WHEN the `/health` endpoint is called and `cache_age_minutes()` returns a value greater than `STALE_FORECAST_THRESHOLD_MINUTES`, THE API_Server SHALL include `"data_stale": true` in the health response JSON.
3. WHEN the `/health` endpoint is called and `cache_age_minutes()` returns a value less than or equal to `STALE_FORECAST_THRESHOLD_MINUTES`, THE API_Server SHALL include `"data_stale": false` in the health response JSON.
4. THE `/health` endpoint SHALL continue to return HTTP 200 even when `data_stale` is `true`, so that Render's health check does not restart the service due to stale data.
5. THE API_Server SHALL add `"stale_threshold_minutes": STALE_FORECAST_THRESHOLD_MINUTES` to the health response so that callers know what threshold was used.
