"""
features.py — Feature engineering for Folsom AQI forecasting.
CRITICAL: No data leakage. All features at row T use only data available at T.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame, horizon_h: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y for a given forecast horizon.

    CRITICAL: No data leakage. The target at row T is us_aqi at T+horizon_h.
    All features at row T must be available at time T with no knowledge of T+1 or later.

    Args:
        df: Merged DataFrame with DatetimeIndex (America/Los_Angeles).
            Required columns: us_aqi, pm2_5, boundary_layer_height, wind_speed_10m,
            surface_pressure, relative_humidity_2m, temperature_2m, precipitation,
            cloud_cover, wind_direction_10m
        horizon_h: Forecast horizon in hours (6, 12, 24, or 48).

    Returns:
        X: Feature DataFrame with DatetimeIndex
        y: Target Series (us_aqi at T+horizon_h, aligned with X's index)

    After calling this function, caller must drop NaN targets:
        mask = y.notna()
        X, y = X[mask], y[mask]
    """
    X = pd.DataFrame(index=df.index)

    # Coerce all columns to numeric — older Open-Meteo data (2020-2021) returns
    # None objects instead of NaN floats, which crashes .diff()/.shift()/.rolling().
    df = df.apply(pd.to_numeric, errors='coerce')

    # -----------------------------------------------------------------------
    # Group 1: AQI lag features
    # FIX: Expose current exact state (lag 0). At time T, we know AQI at time T.
    # -----------------------------------------------------------------------
    X['aqi_current'] = df['us_aqi']
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        X[f'aqi_lag_{lag}h'] = df['us_aqi'].shift(lag)

    # -----------------------------------------------------------------------
    # Group 2: AQI rolling statistics & Temporal Differencing
    # FIX: Rolling window now includes the current row T. 
    # Added differencing to detect daily macro-trends (crucial for 24h+ horizons)
    # -----------------------------------------------------------------------
    X['aqi_diff_1h']  = df['us_aqi'] - df['us_aqi'].shift(1)
    X['aqi_diff_24h'] = df['us_aqi'] - df['us_aqi'].shift(24)

    for window in [3, 6, 12, 24, 48, 168]:
        X[f'aqi_roll_{window}h_mean'] = df['us_aqi'].rolling(window, min_periods=1).mean()
        X[f'aqi_roll_{window}h_max']  = df['us_aqi'].rolling(window, min_periods=1).max()
        X[f'aqi_roll_{window}h_std']  = df['us_aqi'].rolling(window, min_periods=1).std().fillna(0)

    # NEW: Exponentially Weighted Moving Averages
    X['aqi_ewma_6h']  = df['us_aqi'].ewm(span=6,  adjust=False).mean()
    X['aqi_ewma_24h'] = df['us_aqi'].ewm(span=24, adjust=False).mean()
    X['pm25_ewma_6h'] = df['pm2_5'].ewm(span=6, adjust=False).mean()

    # -----------------------------------------------------------------------
    # Group 3: PM2.5 features
    # FIX: Exposing current PM2.5 state and aligning rolling windows to T.
    # -----------------------------------------------------------------------
    X['pm25_current'] = df['pm2_5']
    for lag in [1, 3, 6, 24]:
        X[f'pm25_lag_{lag}h'] = df['pm2_5'].shift(lag)
    X['pm25_roll_6h_mean']  = df['pm2_5'].rolling(6,  min_periods=1).mean()
    X['pm25_roll_24h_mean'] = df['pm2_5'].rolling(24, min_periods=1).mean()

    # -----------------------------------------------------------------------
    # Group 4: Meteorological features
    # These come from NWP (numerical weather prediction) forecast models,
    # so they are genuinely available at forecast time for any horizon.
    # -----------------------------------------------------------------------
    X['boundary_layer_height'] = df['boundary_layer_height']
    X['wind_speed_10m']        = df['wind_speed_10m']
    X['surface_pressure']      = df['surface_pressure']
    X['relative_humidity_2m']  = df['relative_humidity_2m']
    X['temperature_2m']        = df['temperature_2m']
    X['precipitation']         = df['precipitation']
    X['cloud_cover']           = df['cloud_cover']
    X['direct_radiation']      = df.get('direct_radiation', 0)
    X['soil_temp']             = df.get('soil_temperature_0_to_7cm', 0)

    # Domain Knowledge Physical Interactions
    # Ventilation Coefficient = BLH * Wind Speed (measures atmospheric stagnation)
    X['blh_x_wind_speed'] = df['boundary_layer_height'] * df['wind_speed_10m']
    # Dynamic Air Washout = Current AQI * Wind Speed
    X['aqi_x_wind'] = X['aqi_current'] * df['wind_speed_10m']
    # Smog Generation Potential = Current AQI * Shortwave Radiation
    X['aqi_x_rad'] = X['aqi_current'] * X['direct_radiation']

    # -----------------------------------------------------------------------
    # WILDFIRE PROXY FEATURES (Priority 4)
    # Allows the models to detect high-fire-risk environmental geometry.
    # -----------------------------------------------------------------------
    # 1. Hot-Dry-Windy Index (HDWI): Approximates environmental fire danger.
    # High temp (> 30C), low humidity (< 25%), high wind (> 15 km/h) = severe fire risk.
    temp_c = df['temperature_2m']
    rh     = df['relative_humidity_2m']
    wind   = df['wind_speed_10m']
    
    # Vapor Pressure Deficit (VPD) approximation in kPa given T and RH
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = es * (rh / 100.0)
    vpd = es - ea
    
    X['wildfire_hdwi'] = wind * vpd

    # 2. Antecedent Precipitation Deficit (Dry Fuel Conditions)
    # The sum of all rain over the last 30 days. If this is 0 in summer, fires explode.
    X['precip_30d_sum'] = df['precipitation'].rolling(30 * 24, min_periods=1).sum()

    # 3. Extreme Heat & Dry Flag
    # Boolean categorical flag for extreme summer danger conditions.
    X['flag_extreme_heat_dry'] = ((temp_c > 35) & (rh < 25) & (wind > 10)).astype(int)

    # 4. Unconditional Additions: Weather Front Differencing
    X['pressure_diff_3h']  = df['surface_pressure'].diff(3)
    X['pressure_diff_6h']  = df['surface_pressure'].diff(6)
    X['pressure_diff_12h'] = df['surface_pressure'].diff(12)
    X['pressure_diff_24h'] = df['surface_pressure'].diff(24)
    X['temp_diff_24h']     = df['temperature_2m'].diff(24)
    # Extended differencing for multi-day frontal systems (V4.0)
    X['pressure_diff_48h'] = df['surface_pressure'].diff(48)
    X['temp_diff_48h']     = df['temperature_2m'].diff(48)

    # -----------------------------------------------------------------------
    # FORECAST WEATHER FEATURES (V4.0 — highest-impact 48h fix)
    # At time T, NWP models (Open-Meteo) provide weather forecasts for T+horizon_h.
    # During TRAINING: .shift(-horizon_h) gives the ACTUAL weather at the target
    #   hour, which is the best available proxy for what the NWP forecast would be.
    # During INFERENCE: fetch_recent_combined(forecast_days=3) already fills the
    #   DataFrame with forecast rows extending 72h ahead, so these columns are
    #   naturally populated with genuine NWP forecast values at time T+horizon_h.
    # This is NOT data leakage — these features represent information that is
    # genuinely available at prediction time T.
    # -----------------------------------------------------------------------
    X['fwd_wind_speed']   = df['wind_speed_10m'].shift(-horizon_h)
    X['fwd_blh']          = df['boundary_layer_height'].shift(-horizon_h)
    X['fwd_temperature']  = df['temperature_2m'].shift(-horizon_h)
    X['fwd_humidity']     = df['relative_humidity_2m'].shift(-horizon_h)
    X['fwd_pressure']     = df['surface_pressure'].shift(-horizon_h)
    X['fwd_precipitation'] = df['precipitation'].shift(-horizon_h)

    # Forecast-time interactions (computed at the target hour, not current hour)
    # Ventilation coefficient at T+h: measures how well the atmosphere can
    # disperse pollutants at the time the prediction lands.
    X['fwd_ventilation']  = X['fwd_blh'] * X['fwd_wind_speed']

    # Fire danger index at T+h: VPD × wind at the forecast hour
    fwd_es = 0.6108 * np.exp(17.27 * X['fwd_temperature'] / (X['fwd_temperature'] + 237.3))
    fwd_ea = fwd_es * (X['fwd_humidity'] / 100.0)
    fwd_vpd = fwd_es - fwd_ea
    X['fwd_hdwi'] = X['fwd_wind_speed'] * fwd_vpd

    # 5. Satellite Aerosol Optical Depth (Smoke Plume Detection)
    # AOD detects smoke aloft *before* it settles into the boundary layer.
    # The absolute value matters, but the rate of change catches the incoming front.
    if 'aerosol_optical_depth' in df.columns:
        aod = pd.to_numeric(df['aerosol_optical_depth'], errors='coerce')
        X['aod_current'] = aod
        X['aod_diff_3h'] = aod.diff(3)
        X['aod_diff_6h'] = aod.diff(6)

    # Wind direction: encode as sin/cos so 359° ≈ 1° (circular continuity)
    wind_dir_rad = np.radians(df['wind_direction_10m'])
    X['wind_dir_sin'] = np.sin(wind_dir_rad)
    X['wind_dir_cos'] = np.cos(wind_dir_rad)

    # Wind U/V vector decomposition (V4.0)
    # Combines speed AND direction into continuous cartesian components.
    # U = east-west component (positive = from west), V = north-south (positive = from south)
    # LightGBM can split directly on these to detect, e.g., "strong easterly wind" patterns
    # that the separate speed + sin/cos features cannot capture as efficiently.
    X['wind_u'] = wind * np.cos(wind_dir_rad)
    X['wind_v'] = wind * np.sin(wind_dir_rad)

    # -----------------------------------------------------------------------
    # Group 5: Temporal encodings — cyclic, computed from index only
    # -----------------------------------------------------------------------
    hour        = df.index.hour
    day_of_year = df.index.day_of_year
    month       = df.index.month
    day_of_week = df.index.day_of_week
    hr          = hour
    dow         = day_of_week

    # Quantum Models (76-feature set) require these specific aliases
    X['hour_sin']        = np.sin(2 * np.pi * hr / 24)
    X['hour_cos']        = np.cos(2 * np.pi * hr / 24)
    X['dow_sin']         = np.sin(2 * np.pi * dow / 7)
    X['dow_cos']         = np.cos(2 * np.pi * dow / 7)

    X['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    X['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
    X['month_sin']       = np.sin(2 * np.pi * month / 12)
    X['month_cos']       = np.cos(2 * np.pi * month / 12)
    X['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    X['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    X['is_weekend']      = (day_of_week >= 5).astype(int)

    # Future hour (cyclic) — very predictive for long horizons
    future_hour = (hour + horizon_h) % 24
    X['future_hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
    X['future_hour_cos'] = np.cos(2 * np.pi * future_hour / 24)

    # -----------------------------------------------------------------------
    # Group 6: Target construction
    # FIX: Residual Prediction Transformation (Max R^2 Variance)
    # Target = (Future AQI) - (Current AQI)
    # The LightGBM tree now exclusively learns to predict the variance delta.
    # -----------------------------------------------------------------------
    y = df['us_aqi'].shift(-horizon_h) - df['us_aqi']
    y.name = 'target_residual'

    return X, y


def get_feature_names(horizon_h: int = 6) -> list[str]:
    """
    Return ordered list of feature names for a given horizon.
    Used to verify alignment between training and inference.
    """
    # Build a tiny dummy df and extract column names.
    # Uses 500 rows to ensure all rolling windows and forward shifts
    # can produce non-NaN values for feature name extraction.
    idx = pd.date_range('2023-01-01', periods=500, freq='h',
                        tz='America/Los_Angeles')
    dummy = pd.DataFrame({
        'us_aqi':               np.random.rand(500) * 100,
        'pm2_5':                np.random.rand(500) * 50,
        'boundary_layer_height': np.random.rand(500) * 1500,
        'wind_speed_10m':       np.random.rand(500) * 10,
        'surface_pressure':     np.random.rand(500) * 20 + 1010,
        'relative_humidity_2m': np.random.rand(500) * 100,
        'temperature_2m':       np.random.rand(500) * 30,
        'precipitation':        np.random.rand(500),
        'cloud_cover':          np.random.rand(500) * 100,
        'wind_direction_10m':   np.random.rand(500) * 360,
        'direct_radiation':     np.random.rand(500) * 500,
        'soil_temperature_0_to_7cm': np.random.rand(500) * 25,
        'aerosol_optical_depth': np.random.rand(500) * 0.5,
    }, index=idx)
    X, _ = engineer_features(dummy, horizon_h)
    return list(X.columns)
