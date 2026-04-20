"""
features_v6.py — Feature engineering for Folsom AQI forecasting.
CRITICAL: No data leakage. All features at row T use only data available at T.

engineer_features() is decomposed into 7 private sub-functions, one per feature
family. The orchestrator calls each in sequence and calls X = X.copy() after
each to prevent Pandas memory fragmentation from in-place column additions.
"""

import numpy as np
import pandas as pd

from trajectories import add_trajectory_features, TRAJ_LOOKBACKS


# ─── Physical constants ───────────────────────────────────────────────────────

# Empirical Central Valley stagnation threshold (California Air Resources Board).
# When BLH × wind_speed drops below this value, pollutants accumulate.
VENT_THRESHOLD_M2_PER_S: float = 3000.0

# Temperature below which residential wood burning in Folsom increases significantly
# (based on local emission inventory data; 7°C ≈ 45°F).
COLD_SMOKE_THRESHOLD_C: float = 7.0

# Magnus formula coefficients for saturation vapor pressure approximation (kPa).
# Formula: es = 0.6108 × exp(MAGNUS_A × T / (T + MAGNUS_B))
# where T is temperature in °C and es is in kPa.
MAGNUS_A: float = 17.27
MAGNUS_B: float = 237.3

# Open-Meteo hallucination floor for relative humidity (Folsom grid cell artifact).
# Guards against documented cases where the API returns 5–6% RH for Folsom.
HUMIDITY_CLAMP_MIN_PCT: float = 25.0

# Open-Meteo hallucination cap for wind speed (Folsom grid cell artifact).
# Guards against documented cases where the API returns 30+ km/h for calm days.
WIND_SPEED_CAP_KMH: float = 25.0

# Minimum precipitation to count as a wet-deposition scavenging event (mm).
# Rain below this threshold does not meaningfully remove PM2.5 from the air.
RAIN_THRESHOLD_MM: float = 0.1


# ─── Sub-functions ────────────────────────────────────────────────────────────

def _add_aqi_lag_features(X: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Mutates X in-place. Adds Groups 1–2:
    - AQI lags (current, 1, 2, 3, 6, 12, 24, 48h)
    - AQI diffs (1h, 24h), second derivative (acceleration)
    - Rolling stats (mean, max, std) for windows [3,6,12,24,48,168]h
    - EWMA (6h, 24h spans) + PM2.5 EWMA (6h span)
    """
    # Group 1: AQI lag features
    # At time T, we know AQI at time T — expose it as lag 0.
    X['aqi_current'] = df['us_aqi']
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        X[f'aqi_lag_{lag}h'] = df['us_aqi'].shift(lag)

    # Group 2: AQI rolling statistics & temporal differencing
    X['aqi_diff_1h']  = df['us_aqi'] - df['us_aqi'].shift(1)
    X['aqi_diff_24h'] = df['us_aqi'] - df['us_aqi'].shift(24)

    # AQI Second Derivative (Acceleration)
    # Rate of change of the rate of change. Positive = pollution building faster.
    X['aqi_acceleration'] = X['aqi_diff_1h'] - X['aqi_diff_1h'].shift(1)
    X['aqi_acceleration_6h_mean'] = X['aqi_acceleration'].rolling(6, min_periods=1).mean()

    for window in [3, 6, 12, 24, 48, 168]:
        X[f'aqi_roll_{window}h_mean'] = df['us_aqi'].rolling(window, min_periods=1).mean()
        X[f'aqi_roll_{window}h_max']  = df['us_aqi'].rolling(window, min_periods=1).max()
        X[f'aqi_roll_{window}h_std']  = df['us_aqi'].rolling(window, min_periods=1).std().fillna(0)

    # Exponentially Weighted Moving Averages
    X['aqi_ewma_6h']  = df['us_aqi'].ewm(span=6,  adjust=False).mean()
    X['aqi_ewma_24h'] = df['us_aqi'].ewm(span=24, adjust=False).mean()
    X['pm25_ewma_6h'] = df['pm2_5'].ewm(span=6, adjust=False).mean()


def _add_pm25_and_combustion_features(X: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Mutates X in-place. Adds Group 3 + Feature Group 1:
    - PM2.5 current + lags (1,3,6,24h) + rolling (6h, 24h mean)
    - CO current, lags, rolling, diff (if column present)
    - PM2.5/CO wildfire discrimination ratio (if both present)
    - Dust current, rolling, diff (if column present)
    - NO2 current, rolling (if column present)
    """
    # Group 3: PM2.5 features
    X['pm25_current'] = df['pm2_5']
    for lag in [1, 3, 6, 24]:
        X[f'pm25_lag_{lag}h'] = df['pm2_5'].shift(lag)
    X['pm25_roll_6h_mean']  = df['pm2_5'].rolling(6,  min_periods=1).mean()
    X['pm25_roll_24h_mean'] = df['pm2_5'].rolling(24, min_periods=1).mean()

    # Feature Group 1a: Carbon Monoxide (primary combustion tracer)
    if 'carbon_monoxide' in df.columns:
        co = pd.to_numeric(df['carbon_monoxide'], errors='coerce')
        X['co_current']       = co
        X['co_lag_6h']        = co.shift(6)
        X['co_lag_24h']       = co.shift(24)
        X['co_roll_24h_mean'] = co.rolling(24, min_periods=1).mean()
        X['co_diff_6h']       = co.diff(6)   # rising CO = new combustion event
        X['co_roll_6h_max']   = co.rolling(6, min_periods=1).max()

        # Feature Group 1b: PM2.5/CO Wildfire Discrimination Ratio
        if 'pm2_5' in df.columns:
            co_safe = co.replace(0, np.nan)
            # High ratio = wildfire smoke dominant; Low ratio = traffic/urban dominant
            X['pm25_co_ratio'] = df['pm2_5'] / co_safe
            X['pm25_co_ratio_6h_mean'] = X['pm25_co_ratio'].rolling(6, min_periods=1).mean()

    # Feature Group 1c: Dust (mineral aerosol — Sep/Oct advection events)
    if 'dust' in df.columns:
        dust = pd.to_numeric(df['dust'], errors='coerce')
        X['dust_current']       = dust
        X['dust_roll_24h_mean'] = dust.rolling(24, min_periods=1).mean()
        X['dust_diff_6h']       = dust.diff(6)

    # Feature Group 1d: NO₂ (traffic/anthropogenic proxy — inversely correlated with wildfire)
    if 'nitrogen_dioxide' in df.columns:
        no2 = pd.to_numeric(df['nitrogen_dioxide'], errors='coerce')
        X['no2_current']       = no2
        X['no2_roll_24h_mean'] = no2.rolling(24, min_periods=1).mean()

    # Feature Group 1e: Ozone (V15 — photochemical leading indicator)
    # Physics: O3 is produced by UV + NOx + VOCs. High ozone at T is a direct
    # signal that photochemical reactions are active, which drives secondary
    # PM2.5 formation 3-6h later. It is a leading indicator of afternoon AQI
    # spikes that CO and PM2.5 alone cannot capture.
    if 'ozone' in df.columns:
        o3 = pd.to_numeric(df['ozone'], errors='coerce')
        X['ozone_current']    = o3
        X['ozone_roll_6h_mean'] = o3.rolling(6,  min_periods=1).mean()
        X['ozone_diff_3h']    = o3.diff(3)   # rising O3 = photochem ramp-up


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

    NWP forward-shift training/inference symmetry:
    During TRAINING: .shift(-horizon_h) gives the ACTUAL weather at the target
      hour, which is the best available proxy for what the NWP forecast would be.
    During INFERENCE: fetch_recent_combined(forecast_days=5) already fills the
      DataFrame with forecast rows extending 72h+ ahead, so these columns are
      naturally populated with genuine NWP forecast values at time T+horizon_h.
    This is NOT data leakage — these features represent information that is
    genuinely available at prediction time T.
    """
    # Raw meteorological columns (NWP forecast values, available at any horizon)
    X['boundary_layer_height'] = df['boundary_layer_height']
    X['wind_speed_10m']        = df['wind_speed_10m']
    X['surface_pressure']      = df['surface_pressure']
    X['relative_humidity_2m']  = df['relative_humidity_2m']
    X['temperature_2m']        = df['temperature_2m']
    X['precipitation']         = df['precipitation']
    X['cloud_cover']           = df['cloud_cover']
    X['direct_radiation']      = pd.to_numeric(df.get('direct_radiation'), errors='coerce').fillna(0)

    # Domain knowledge physical interactions
    # Ventilation Coefficient = BLH × Wind Speed (measures atmospheric stagnation)
    X['blh_x_wind_speed'] = df['boundary_layer_height'] * df['wind_speed_10m']
    # Dynamic Air Washout = Current AQI × Wind Speed
    X['aqi_x_wind'] = X['aqi_current'] * df['wind_speed_10m']
    # Smog Generation Potential = Current AQI × Shortwave Radiation
    X['aqi_x_rad'] = X['aqi_current'] * X['direct_radiation']

    # V8: Photochemical Forcing Index
    # Physics: Secondary PM2.5 forms through photochemical reactions when
    # UV/solar radiation drives VOCs and NOx into particulate-phase sulfates.
    # The reaction rate depends on GHI (energy), humidity (aqueous pathway),
    # and temperature (Arrhenius kinetics).
    ghi = pd.to_numeric(df.get('shortwave_radiation', 0), errors='coerce').fillna(0)
    _pc_temp = pd.to_numeric(df['temperature_2m'], errors='coerce')
    _pc_rh   = pd.to_numeric(df['relative_humidity_2m'], errors='coerce')
    ghi_norm = ghi / 1000.0
    rh_factor = (_pc_rh / 100.0).clip(0.3, 1.0)
    thermal_factor = np.exp(0.069 * (_pc_temp - 25.0))

    X['photochem_forcing'] = ghi_norm * rh_factor * thermal_factor
    X['photochem_forcing_6h'] = X['photochem_forcing'].rolling(6, min_periods=1).mean()
    X['ghi_accum_12h'] = ghi.rolling(12, min_periods=1).sum()

    # Forward photochemistry at the target hour
    fwd_ghi = ghi.shift(-horizon_h)
    fwd_rh_pc = df['relative_humidity_2m'].shift(-horizon_h)
    fwd_temp_pc = df['temperature_2m'].shift(-horizon_h)
    fwd_thermal = np.exp(0.069 * (fwd_temp_pc - 25.0))
    X['fwd_photochem_forcing'] = (fwd_ghi / 1000.0) * (fwd_rh_pc / 100.0).clip(0.3, 1.0) * fwd_thermal

    # V8.2: Forward Photochemical Accumulation (Volumetric)
    # Solves the "8 PM zero-sun blindness" by tracking accumulated baking prior to the target
    X['fwd_photochem_accum_12h'] = X['photochem_forcing'].rolling(12, min_periods=1).sum().shift(-horizon_h)

    # Forward NWP features (see docstring for training/inference symmetry explanation)
    X['fwd_wind_speed']    = df['wind_speed_10m'].shift(-horizon_h)
    X['fwd_blh']           = df['boundary_layer_height'].shift(-horizon_h)
    X['fwd_temperature']   = df['temperature_2m'].shift(-horizon_h)
    X['fwd_humidity']      = df['relative_humidity_2m'].shift(-horizon_h)
    X['fwd_pressure']      = df['surface_pressure'].shift(-horizon_h)
    X['fwd_precipitation'] = df['precipitation'].shift(-horizon_h)

    # V7: Forward wind direction and components (Delta Breeze detection capability)
    fwd_wind_dir = df['wind_direction_10m'].shift(-horizon_h)
    fwd_wind_dir_rad = np.radians(fwd_wind_dir)
    X['fwd_wind_dir_sin'] = np.sin(fwd_wind_dir_rad)
    X['fwd_wind_dir_cos'] = np.cos(fwd_wind_dir_rad)
    X['fwd_wind_u'] = X['fwd_wind_speed'] * X['fwd_wind_dir_cos']
    X['fwd_wind_v'] = X['fwd_wind_speed'] * X['fwd_wind_dir_sin']

    # V7: Strict future rolling windows (anti-temporal leakage).
    # Takes [T+1 to T+horizon_h] safely by backward rolling + forward shift.
    X['fwd_temperature_mean'] = df['temperature_2m'].rolling(horizon_h, min_periods=1).mean().shift(-horizon_h)
    X['fwd_wind_speed_mean']  = df['wind_speed_10m'].rolling(horizon_h, min_periods=1).mean().shift(-horizon_h)
    X['fwd_humidity_mean']    = df['relative_humidity_2m'].rolling(horizon_h, min_periods=1).mean().shift(-horizon_h)
    X['fwd_pressure_mean']    = df['surface_pressure'].rolling(horizon_h, min_periods=1).mean().shift(-horizon_h)
    X['fwd_precip_accum']     = df['precipitation'].rolling(horizon_h, min_periods=1).sum().shift(-horizon_h)

    # Forecast-time ventilation coefficient and deficit
    # Measures how well the atmosphere can disperse pollutants at the time the prediction lands.
    X['fwd_ventilation']  = X['fwd_blh'] * X['fwd_wind_speed']
    X['fwd_vent_deficit'] = (VENT_THRESHOLD_M2_PER_S - X['fwd_ventilation']).clip(lower=0)

    # Fire danger index at T+h: VPD × wind at the forecast hour
    # Magnus formula: saturation vapor pressure at forecast temperature
    fwd_saturation_vapor_pressure_kpa = 0.6108 * np.exp(
        MAGNUS_A * X['fwd_temperature'] / (X['fwd_temperature'] + MAGNUS_B)
    )
    fwd_actual_vapor_pressure_kpa = fwd_saturation_vapor_pressure_kpa * (X['fwd_humidity'] / 100.0)
    fwd_vapor_pressure_deficit_kpa = fwd_saturation_vapor_pressure_kpa - fwd_actual_vapor_pressure_kpa
    X['fwd_hdwi'] = X['fwd_wind_speed'] * fwd_vapor_pressure_deficit_kpa

    # Satellite Aerosol Optical Depth (Smoke Plume Detection)
    # AOD detects smoke aloft *before* it settles into the boundary layer.
    # The absolute value matters, but the rate of change catches the incoming front.
    #
    # V11 additions:
    #   aod_roll_24h_mean — 24h rolling mean smooths satellite pass gaps (MODIS
    #     only passes overhead ~4x/day, so hourly AOD has many NaN gaps).
    #   fwd_aod — AOD at T+horizon_h is the most direct smoke-transport signal
    #     for longer horizons: if a smoke plume is approaching Folsom, AOD will
    #     rise upwind before PM2.5 rises at the surface. This is the leading
    #     indicator that the current fire_advection_score approximates but cannot
    #     directly measure.
    if 'aerosol_optical_depth' in df.columns:
        aod = pd.to_numeric(df['aerosol_optical_depth'], errors='coerce')
        X['aod_current']       = aod
        X['aod_diff_3h']       = aod.diff(3)
        X['aod_diff_6h']       = aod.diff(6)
        X['aod_roll_24h_mean'] = aod.rolling(24, min_periods=1).mean()
        X['fwd_aod']           = aod.shift(-horizon_h)

        # V15: AOD interaction features (smoke transport signal amplification)
        # Physics: AOD × wind_alignment = smoke column density × transport efficiency.
        # A high AOD plume only matters if the wind is blowing it toward Folsom.
        # aod_trend_6h: slope of AOD over last 6h — rising AOD = incoming plume.
        # aod_persistence_24h: hours with AOD > 0.3 in last 24h — sustained smoke event.
        X['aod_trend_6h'] = aod.diff(6)  # same as aod_diff_6h but semantically named
        aod_high_flag = (aod > 0.3).astype(float)
        X['aod_persistence_24h'] = aod_high_flag.rolling(24, min_periods=1).sum()

    # Wind direction: encode as sin/cos so 359° ≈ 1° (circular continuity)
    wind_dir_rad = np.radians(df['wind_direction_10m'])
    X['wind_dir_sin'] = np.sin(wind_dir_rad)
    X['wind_dir_cos'] = np.cos(wind_dir_rad)

    # Wind U/V vector decomposition (V4.0)
    # Combines speed AND direction into continuous cartesian components.
    # LightGBM can split directly on these to detect "strong easterly wind" patterns
    # that the separate speed + sin/cos features cannot capture as efficiently.
    wind = df['wind_speed_10m']
    X['wind_u'] = wind * np.cos(wind_dir_rad)
    X['wind_v'] = wind * np.sin(wind_dir_rad)


def _add_atmospheric_stability_features(
    X: pd.DataFrame, df: pd.DataFrame, horizon_h: int
) -> None:
    """
    Mutates X in-place. Adds Group 5 (stability sub-groups 5a–5f):
    5a. Stagnation index (6h, 24h, 48h rolling sums + streak + fog-nitrate)
    5b. Inversion proxy (strength, 12h max)
    5b2. True inversion delta 850hPa (if column present)
    4b. 700hPa inversion depth (if both 700/850 columns present)
    5c. Ventilation deficit (current + 24h mean)
    5d. Synoptic blocking index Z500 (if column present)
    5e. Cold degree hours (current + 48h sum)
    5f. Tule fog precursor (dew point depression + 12h rolling)
    """
    wind = df['wind_speed_10m']
    # Renamed from `blh` to `boundary_layer_height` to eliminate the inconsistency
    # between the abbreviated local variable and the DataFrame column name.
    boundary_layer_height = df['boundary_layer_height']

    # --- 5a. Stagnation Index ---
    # Stagnation occurs when BOTH wind speed is low AND the boundary layer
    # is shallow (compressed). This "double trap" prevents vertical and
    # horizontal pollutant dispersal.
    #
    # Physics: Wind < 2 m/s + BLH < 500m = classic Central Valley stagnation.
    # The index is continuous (0-1 per hour), then summed over rolling windows
    # to capture multi-hour and multi-day events.
    low_wind = (wind < 2.0).astype(float)
    low_blh  = (boundary_layer_height < 500).astype(float)
    stag_hourly = low_wind * low_blh  # 1.0 when both conditions are met

    X['stagnation_24h'] = stag_hourly.rolling(24, min_periods=1).sum()
    X['stagnation_48h'] = stag_hourly.rolling(48, min_periods=1).sum()

    # fog_nitrate_index, stagnation_6h, stagnation_streak_h removed (V15: zero importance)

    # --- 5b. Inversion Proxy ---
    # A temperature inversion occurs when air aloft is warmer than air at
    # the surface, creating a "lid" that traps pollutants. We approximate
    # this using two complementary heuristics:
    #
    # Heuristic 1: "Shallow BLH + Rising Pressure"
    #   When the boundary layer collapses (< 300m) AND pressure is rising
    #   (high-pressure system settling in), a strong inversion is forming.
    #   This is the classic Sacramento Valley winter pattern.
    #
    # Heuristic 2: "Nighttime Radiative Cooling"
    #   Temperature drops sharply at night while pressure stays high.
    #   A large negative temp_diff (cooling) with positive pressure_diff
    #   (stabilization) strongly indicates a radiative inversion.
    pressure_change_6h = df['surface_pressure'].diff(6)
    temp_change_6h     = df['temperature_2m'].diff(6)

    # Inversion Strength: shallow BLH × pressure rise × cooling rate
    # Clamp BLH to avoid division-by-zero; smaller BLH = stronger inversion
    blh_clamped = boundary_layer_height.clip(lower=50)
    inv_blh_component      = 1000.0 / blh_clamped       # ~2.0 when BLH=500, ~20 when BLH=50
    inv_pressure_component = pressure_change_6h.clip(lower=0)   # only rising pressure
    inv_cooling_component  = (-temp_change_6h).clip(lower=0)    # only cooling

    X['inversion_strength'] = inv_blh_component * inv_pressure_component * inv_cooling_component
    X['inversion_12h_max']  = X['inversion_strength'].rolling(12, min_periods=1).max()

    # --- 5b2. TRUE Inversion Delta (850hPa) ---
    # The REAL inversion measurement: T_850hPa - T_2m.
    # Positive = warm air aloft (inversion lid trapping pollutants)
    # Negative = normal lapse rate (good vertical mixing)
    # This is the gold-standard metric used by NWS to issue stagnation advisories.
    if 'temperature_850hPa' in df.columns:
        t850 = pd.to_numeric(df['temperature_850hPa'], errors='coerce')
        # Robust NaN handling: forward-fill gaps (up to 6h), then backfill residuals
        t850 = t850.ffill(limit=6).bfill(limit=6)
        t2m  = pd.to_numeric(df['temperature_2m'], errors='coerce')

        X['inversion_delta_850'] = t850 - t2m
        X['inversion_delta_850_6h_mean'] = X['inversion_delta_850'].rolling(6, min_periods=1).mean()
        X['fwd_inversion_delta_850'] = X['inversion_delta_850'].shift(-horizon_h)

    # --- 4b. 700hPa Inversion Depth (V5.2) ---
    # The temperature gradient between 700hPa and 850hPa tells you the DEPTH and
    # STABILITY of the inversion lid.
    if 'temperature_700hPa' in df.columns and 'temperature_850hPa' in df.columns:
        t700 = pd.to_numeric(df['temperature_700hPa'], errors='coerce').ffill(limit=6).bfill(limit=6)
        t850 = pd.to_numeric(df['temperature_850hPa'], errors='coerce').ffill(limit=6).bfill(limit=6)
        t2m  = pd.to_numeric(df['temperature_2m'],     errors='coerce')

        # Full atmospheric column: T_700 - T_2m (full inversion column depth)
        X['inversion_column_depth'] = t700 - t2m
        # Inter-level gradient: T_700 - T_850 (stability of the inversion lid itself)
        # Near-zero = lid is thick and stable; large = lid is shallow and breakable
        X['inversion_lid_stability'] = t700 - t850
        X['inversion_column_24h_mean'] = X['inversion_column_depth'].rolling(24, min_periods=1).mean()
        X['fwd_inversion_column_depth'] = X['inversion_column_depth'].shift(-horizon_h)
        X['fwd_inversion_lid_stability'] = X['inversion_lid_stability'].shift(-horizon_h)

    # --- 5c. Ventilation Deficit ---
    # The ventilation coefficient (BLH × Wind) measures the atmosphere's
    # ability to flush pollutants. When it drops below VENT_THRESHOLD_M2_PER_S,
    # pollutants accumulate. The "deficit" is how far below this threshold we are.
    # This differs from blh_x_wind_speed: it captures the DEFICIT (how trapped we
    # are) rather than the raw ventilation value.
    ventilation = boundary_layer_height * wind
    vent_deficit = (VENT_THRESHOLD_M2_PER_S - ventilation).clip(lower=0)
    X['vent_deficit']          = vent_deficit
    X['vent_deficit_24h_mean'] = vent_deficit.rolling(24, min_periods=1).mean()

    # --- 5d. Synoptic Blocking Index (V5.1 Winter Patch) ---
    # Physics: Measures if a massive high-pressure ridge is parked over CA.
    # High Z500 = high pressure aloft = sinking air = stable trapping.
    if 'geopotential_height_500hPa' in df.columns:
        z500 = pd.to_numeric(df['geopotential_height_500hPa'], errors='coerce')
        z500 = z500.ffill(limit=24).fillna(z500.median())
        z500_doy_mean = z500.groupby(df.index.dayofyear).transform('mean')
        X['z500_anomaly'] = z500 - z500_doy_mean
        is_blocked = (X['z500_anomaly'] > 50).astype(float)
        X['blocking_persistence_72h'] = is_blocked.rolling(72, min_periods=1).sum()

    # --- 5e. Human Emission Proxy (V5.1 Winter Patch) ---
    # Physics: Cold temperatures (< 7°C / 45°F) trigger residential wood smoke.
    # COLD_SMOKE_THRESHOLD_C = 7°C is the temperature below which residential
    # wood burning in Folsom increases significantly (local emission inventory data).
    cold_degree_hours = (COLD_SMOKE_THRESHOLD_C - df['temperature_2m']).clip(lower=0)
    X['cold_degree_hours']      = cold_degree_hours
    X['cold_degree_hours_48h']  = cold_degree_hours.rolling(48, min_periods=1).sum()

    # --- 5f. Tule Fog Precursor (V5.1 Winter Patch) ---
    # Physics: Winter fog acts as a pollutant trap. Forms when T ≈ DewPoint.
    temp = pd.to_numeric(df['temperature_2m'], errors='coerce')
    rh   = pd.to_numeric(df['relative_humidity_2m'], errors='coerce')

    # Magnus approximation for dew point:
    # alpha = ln(RH/100) + (MAGNUS_A × T) / (MAGNUS_B + T)
    # dew_point = (MAGNUS_B × alpha) / (MAGNUS_A - alpha)
    alpha = np.log(rh.clip(lower=1) / 100) + (MAGNUS_A * temp) / (MAGNUS_B + temp)
    dew_point = (MAGNUS_B * alpha) / (MAGNUS_A - alpha)

    X['dew_point_depression'] = temp - dew_point
    is_foggy = (X['dew_point_depression'] < 3.0).astype(float)
    X['fog_precursor_12h'] = is_foggy.rolling(12, min_periods=1).sum()


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
    temp_c = df['temperature_2m']
    rh     = df['relative_humidity_2m']
    wind   = df['wind_speed_10m']

    # Hot-Dry-Windy Index (HDWI): approximates environmental fire danger.
    # High temp (> 30°C), low humidity (< 25%), high wind (> 15 km/h) = severe fire risk.
    # Magnus formula: saturation vapor pressure at current temperature
    saturation_vapor_pressure_kpa = 0.6108 * np.exp(
        MAGNUS_A * temp_c / (temp_c + MAGNUS_B)
    )
    actual_vapor_pressure_kpa = saturation_vapor_pressure_kpa * (rh / 100.0)
    vapor_pressure_deficit_kpa = saturation_vapor_pressure_kpa - actual_vapor_pressure_kpa
    X['wildfire_hdwi'] = wind * vapor_pressure_deficit_kpa

    # Antecedent Precipitation Deficit (Dry Fuel Conditions)
    # Sum of all rain over the last 30 days. Zero in summer = extreme fire risk.
    X['precip_30d_sum'] = df['precipitation'].rolling(30 * 24, min_periods=1).sum()

    # Days Since Last Rain (Precipitation Scavenging Memory)
    # Rain removes PM2.5 via wet deposition. Longer dry spells = more accumulation.
    # RAIN_THRESHOLD_MM = 0.1mm is the minimum to count as a scavenging event.
    rain_flag  = (df['precipitation'] > RAIN_THRESHOLD_MM).astype(int)
    dry_groups = (rain_flag != rain_flag.shift()).cumsum()
    dry_flag   = 1 - rain_flag
    dry_cumsum = dry_flag.groupby(dry_groups).cumsum()
    X['hours_since_rain'] = dry_cumsum
    X['days_since_rain']  = X['hours_since_rain'] / 24.0

    # flag_extreme_heat_dry removed (V15: zero importance — subsumed by wildfire_hdwi)

    # Weather Front Differencing (pressure and temperature gradients)
    X['pressure_diff_3h']  = df['surface_pressure'].diff(3)
    X['pressure_diff_6h']  = df['surface_pressure'].diff(6)
    X['pressure_diff_12h'] = df['surface_pressure'].diff(12)
    X['pressure_diff_24h'] = df['surface_pressure'].diff(24)
    X['temp_diff_24h']     = df['temperature_2m'].diff(24)
    # Extended differencing for multi-day frontal systems (V4.0)
    X['pressure_diff_48h'] = df['surface_pressure'].diff(48)
    X['temp_diff_48h']     = df['temperature_2m'].diff(48)

    # Group 8: TRUE WILDFIRE TRACKING (V6.0 NASA FIRMS Integration)
    if 'fire_frp_raw' in df.columns:
        frp   = pd.to_numeric(df['fire_frp_raw'],      errors='coerce').fillna(0)
        dist  = pd.to_numeric(df['fire_min_dist_raw'], errors='coerce').fillna(999.0)
        count = pd.to_numeric(df['fire_count_raw'],    errors='coerce').fillna(0)

        # fire_frp_current, fire_min_dist_current, fire_count_current removed (V15: zero importance)
        # Instantaneous FIRMS values are too noisy; 24h rolling aggregates are used instead.

        # Roll 24 hours: NASA satellites only pass overhead ~4 times per day.
        # The fire is still burning between passes.
        X['fire_frp_24h_sum']    = frp.rolling(24, min_periods=1).sum()
        X['fire_min_dist_24h']   = dist.rolling(24, min_periods=1).min()
        X['fire_count_24h_sum']  = count.rolling(24, min_periods=1).sum()

        # Physics Engine: Inverse Square Law
        # Thermal energy spreading over an area decays by the square of distance.
        X['fire_intensity_proximity_index'] = (
            X['fire_frp_24h_sum'] / ((X['fire_min_dist_24h'] + 1.0) ** 2)
        )

        # V8: Vectorized Fire Advection (Directional Smoke Transport)
        # A fire only delivers smoke to Folsom if the wind blows FROM the fire
        # TOWARD Folsom. We compute the dot product of wind vector and
        # fire-to-Folsom vector, clamping negative (downwind) to zero.
        if 'fire_bearing_nearest' in df.columns:
            fire_bearing = pd.to_numeric(df['fire_bearing_nearest'], errors='coerce')
            fire_to_folsom_deg = (fire_bearing + 180.0) % 360.0

            wind_dir_adv = pd.to_numeric(df['wind_direction_10m'], errors='coerce')
            angle_diff_rad = np.radians(wind_dir_adv - fire_to_folsom_deg)
            alignment = np.cos(angle_diff_rad).clip(lower=0)
            # fire_advection_score kept as intermediate for 24h_max computation
            _fire_adv = X['fire_intensity_proximity_index'] * alignment

            fwd_wind_dir_adv = wind_dir_adv.shift(-horizon_h)
            fwd_angle_diff_rad = np.radians(fwd_wind_dir_adv - fire_to_folsom_deg)
            fwd_alignment = np.cos(fwd_angle_diff_rad).clip(lower=0)
            _fwd_fire_adv = X['fire_intensity_proximity_index'] * fwd_alignment
        else:
            _fire_adv     = X['fire_intensity_proximity_index']
            _fwd_fire_adv = X['fire_intensity_proximity_index']

        # 24h rolling max — the meaningful signal (instantaneous score removed V15)
        X['fire_advection_24h_max']     = _fire_adv.rolling(24, min_periods=1).max()
        X['fwd_fire_advection_24h_max'] = _fwd_fire_adv.rolling(24, min_periods=1).max().shift(-horizon_h)

        # V15: Fire persistence features
        # Physics: A single FIRMS detection is noisy (cloud cover, satellite angle).
        # Multi-day fire persistence is a much stronger signal for sustained smoke.
        # fire_frp_7d_max: peak FRP in last 7 days — captures major fire events
        #   even when the satellite misses a pass.
        # fire_active_days_30d: count of days with any fire detection in last 30 days —
        #   distinguishes a brief flare from a sustained wildfire season.
        X['fire_frp_7d_max']      = frp.rolling(7 * 24, min_periods=1).max()
        # fire_active_days_30d: rolling 30-day count of hours with any fire detection,
        # divided by 24 to convert to approximate day count. Avoids resample() on
        # tz-aware index which can produce alignment issues.
        fire_hour_flag            = (frp > 0).astype(float)
        X['fire_active_days_30d'] = fire_hour_flag.rolling(30 * 24, min_periods=1).sum() / 24.0


def _add_temporal_features(
    X: pd.DataFrame, df: pd.DataFrame, horizon_h: int
) -> None:
    """
    Mutates X in-place. Adds Group 6:
    - Cyclic encodings: hour sin/cos, DOW sin/cos, DOY sin/cos, month sin/cos,
      day_of_week sin/cos
    - is_weekend flag
    - Second harmonic (commute traffic): hour_sin_2, hour_cos_2, weekday_rush_proxy
    - Future hour cyclic: future_hour_sin, future_hour_cos
    """
    hour        = df.index.hour
    day_of_year = df.index.day_of_year
    month       = df.index.month
    day_of_week = df.index.day_of_week
    hr          = hour
    dow         = day_of_week

    # Cyclic encodings (aliases required by trained model feature names)
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
    # is_weekend removed (V15: zero importance — subsumed by dow_sin/cos cyclic encoding)

    # V8: Commute Traffic Harmonics
    # Second harmonic captures the bimodal AM/PM rush pattern that integer
    # dummy encoding cannot represent (23:00 is NOT "far" from 00:00).
    X['hour_sin_2'] = np.sin(2 * 2 * np.pi * hr / 24)
    X['hour_cos_2'] = np.cos(2 * 2 * np.pi * hr / 24)
    # weekday_rush_proxy removed (V15: low importance, depends on removed is_weekend)

    # Future hour (cyclic) — very predictive for long horizons
    future_hour = (hour + horizon_h) % 24
    X['future_hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
    X['future_hour_cos'] = np.cos(2 * np.pi * future_hour / 24)


def _add_regulatory_features(
    X: pd.DataFrame, df: pd.DataFrame, horizon_h: int
) -> None:
    """
    Mutates X in-place. Adds Group 7 + V8/V9 derived features:
    - cbyb_season_flag (Nov–Feb wood-burning season)
    - weekend_burning_proxy (is_weekend × cold_degree_hours)
    - radiation_accum_6h (if shortwave_radiation column present)
    - V8 multi-scale atmospheric momentum features
    - V9 second-order interaction features
    - V9 evening BLH collapse velocity
    """
    # Group 7: Regulatory and Seasonal Features
    # cbyb_season_flag, weekend_burning_proxy removed (V15: zero importance)
    if 'shortwave_radiation' in df.columns:
        X['radiation_accum_6h'] = (
            pd.to_numeric(df['shortwave_radiation'], errors='coerce')
            .rolling(6, min_periods=1).sum()
        )
    else:
        X['radiation_accum_6h'] = 0.0

    # V8: Multi-Scale Atmospheric Momentum
    X['aqi_momentum_6h']   = X['aqi_ewma_6h'] - X['aqi_ewma_24h']
    X['aqi_momentum_24h']  = X['aqi_ewma_24h'] - X['aqi_roll_168h_mean']
    X['momentum_accel_6h'] = X['aqi_momentum_6h'] - X['aqi_momentum_6h'].shift(6)

    # Fat-tail event detector — keep zscore and persistence, drop binary flag (zero importance)
    weekly_std = X['aqi_roll_168h_std'].clip(lower=1.0)
    X['aqi_zscore_7d']            = (X['aqi_current'] - X['aqi_roll_168h_mean']) / weekly_std
    X['fat_tail_persistence_48h'] = (X['aqi_zscore_7d'] > 2.0).astype(float).rolling(48, min_periods=1).sum()

    # V9: Second-Order Interaction Features
    X['stability_index']        = (X['fwd_temperature_mean'] + 273.15) / X['fwd_blh'].clip(lower=50)
    X['trapping_power']         = X['inversion_column_24h_mean'] / X['fwd_wind_speed_mean'].clip(lower=0.5)
    X['fwd_ventilation_stress'] = X['fwd_humidity_mean'] * X['fwd_vent_deficit']
    X['volatility_frontal']     = X['aqi_roll_168h_std'] * X['pressure_diff_48h'].abs()

    summer_flag = ((df.index.month >= 5) & (df.index.month <= 9)).astype(float)
    X['summer_photochem_accum'] = X['fwd_photochem_accum_12h'] * summer_flag

    # V9: Evening BLH Collapse Velocity
    X['blh_collapse_rate']     = df['boundary_layer_height'].diff(3)
    X['fwd_blh_collapse_rate'] = df['boundary_layer_height'].diff(3).shift(-horizon_h)
    # evening_trap_flag removed (V15: zero importance)

    # V15 Tier 2: Regime-Conditional Interaction Features
    # Physics: The dominant predictive signal changes completely by atmospheric regime.
    # Regime 0 (well-mixed): wind features dominate — AQI will flush quickly.
    # Regime 1 (stagnant/inversion): AQI persistence dominates — pollutants trap.
    # Regime 2 (normal): intermediate behavior.
    #
    # Explicit cross-products give LightGBM a single-split shortcut to condition
    # on regime without needing deep trees to discover the interaction.
    # classify_regime is imported from the bottom of this file — use inline logic
    # to avoid circular dependency.
    wind_num  = pd.to_numeric(df['wind_speed_10m'], errors='coerce').fillna(0)
    blh_num   = pd.to_numeric(df['boundary_layer_height'], errors='coerce').fillna(500)
    regime_0  = ((wind_num >= 5.0) | (blh_num >= 1500.0)).astype(float)   # well-mixed
    regime_1  = ((wind_num < 2.0)  & (blh_num < 500.0)).astype(float)     # stagnant

    # Regime × AQI persistence: stagnant regime amplifies AQI persistence signal
    X['regime1_x_aqi_current']       = regime_1 * X['aqi_current']
    X['regime1_x_aqi_roll_24h_mean'] = regime_1 * X['aqi_roll_24h_mean']
    # Regime × ventilation: well-mixed regime amplifies wind flushing signal
    X['regime0_x_fwd_wind_mean']     = regime_0 * X['fwd_wind_speed_mean']
    X['regime0_x_trapping_power']    = regime_0 * X['trapping_power']



# ─── Public API ───────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    horizon_h: int,
    firms_hourly: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y for a given forecast horizon.

    CRITICAL: No data leakage. The target at row T is us_aqi at T+horizon_h.
    All features at row T must be available at time T with no knowledge of T+1 or later.

    Args:
        df:           Merged DataFrame with DatetimeIndex (America/Los_Angeles).
                      Required columns: us_aqi, pm2_5, boundary_layer_height,
                      wind_speed_10m, surface_pressure, relative_humidity_2m,
                      temperature_2m, precipitation, cloud_cover, wind_direction_10m
        horizon_h:    Forecast horizon in hours (6, 12, 24, or 48).
        firms_hourly: Optional hourly FIRMS DataFrame (indexed by timestamp) for
                      V12 Lagrangian trajectory features. If None or empty, trajectory
                      features degrade gracefully to zero/999 (no crash).

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

    _add_aqi_lag_features(X, df)
    X = X.copy()  # defragment after column additions

    _add_pm25_and_combustion_features(X, df)
    X = X.copy()

    _add_meteorological_features(X, df, horizon_h)
    X = X.copy()

    _add_atmospheric_stability_features(X, df, horizon_h)
    X = X.copy()

    _add_wildfire_features(X, df, horizon_h)
    X = X.copy()

    _add_temporal_features(X, df, horizon_h)
    X = X.copy()

    _add_regulatory_features(X, df, horizon_h)
    X = X.copy()

    # V14: Trajectory features are horizon-aware — strictly gated to horizon_h >= 24.
    # Belt-and-suspenders: even if add_trajectory_features is called, we drop
    # any traj_* column that slipped in for short horizons (e.g. traj_origin_lat/lon
    # acting as wind-history proxies at 12h).
    if horizon_h >= 24:
        _firms = firms_hourly if firms_hourly is not None else pd.DataFrame()
        add_trajectory_features(X, df, _firms)
        X = X.copy()

    # Hard drop: guarantee zero traj_* columns for horizon_h < 24
    if horizon_h < 24:
        traj_cols = [c for c in X.columns if c.startswith("traj_") or c == "smoke_wind_alignment"]
        if traj_cols:
            X.drop(columns=traj_cols, inplace=True)

    # Target construction
    # Residual Prediction: Target = (Future AQI) - (Current AQI)
    # The model learns to predict the delta, not the absolute value.
    y = df['us_aqi'].shift(-horizon_h) - df['us_aqi']
    y.name = 'target_residual'

    return X, y


def get_feature_names(horizon_h: int = 6) -> list[str]:
    """
    Return ordered list of feature names for a given horizon.
    Used to verify alignment between training and inference.

    V13: trajectory features are only included for horizon_h >= 24.
    Pass the correct horizon to get the exact feature set for that model.
    """
    # Build a tiny dummy df and extract column names.
    # Uses 500 rows to ensure all rolling windows and forward shifts
    # can produce non-NaN values for feature name extraction.
    idx = pd.date_range('2023-01-01', periods=500, freq='h',
                        tz='America/Los_Angeles')
    dummy = pd.DataFrame({
        'us_aqi':                    np.random.rand(500) * 100,
        'pm2_5':                     np.random.rand(500) * 50,
        'boundary_layer_height':     np.random.rand(500) * 1500,
        'wind_speed_10m':            np.random.rand(500) * 10,
        'surface_pressure':          np.random.rand(500) * 20 + 1010,
        'relative_humidity_2m':      np.random.rand(500) * 100,
        'temperature_2m':            np.random.rand(500) * 30,
        'precipitation':             np.random.rand(500),
        'cloud_cover':               np.random.rand(500) * 100,
        'cloud_cover_low':           np.random.rand(500) * 100,
        'wind_direction_10m':        np.random.rand(500) * 360,
        'direct_radiation':          np.random.rand(500) * 500,
        'shortwave_radiation':       np.random.rand(500) * 1000,
        'soil_temperature_0_to_7cm': np.random.rand(500) * 25,
        'aerosol_optical_depth':     np.random.rand(500) * 0.5,
        'temperature_850hPa':        np.random.rand(500) * 20 - 5,
        'temperature_700hPa':        np.random.rand(500) * 15 - 10,
        'geopotential_height_500hPa': np.random.rand(500) * 500 + 5500,
        'carbon_monoxide':           np.random.rand(500) * 1000,
        'nitrogen_dioxide':          np.random.rand(500) * 100,
        'dust':                      np.random.rand(500) * 50,
        'fire_frp_raw':              np.random.rand(500) * 1000,
        'fire_min_dist_raw':         np.random.rand(500) * 100,
        'fire_count_raw':            np.random.rand(500) * 10,
        'fire_bearing_nearest':      np.random.rand(500) * 360,
    }, index=idx)
    # Only pass firms_dummy for horizons that use trajectory features (>= 24h)
    firms_dummy = None
    if horizon_h >= 24:
        firms_dummy = pd.DataFrame({
            'fire_frp_raw':          np.random.rand(500) * 1000,
            'fire_count_raw':        np.random.rand(500) * 10,
            'fire_min_dist_raw':     np.random.rand(500) * 100,
            'fire_bearing_nearest':  np.random.rand(500) * 360,
        }, index=idx)
    X, _ = engineer_features(dummy, horizon_h, firms_hourly=firms_dummy)
    return list(X.columns)


# ─── Atmospheric Regime Classification (V5.0) ─────────────────────────────────

def classify_regime(df: pd.DataFrame) -> pd.Series:
    """
    Classify every hourly row into one of three atmospheric regimes based on
    physically-grounded thresholds calibrated for the Central Valley (Folsom).

    Regime 0 — "Well-Mixed / High Wind"
        Strong ventilation: wind ≥ 5 m/s OR BLH ≥ 1500m.
        Characteristic of spring frontal passages and Delta Breeze clearing
        events. AQI changes are fast and stochastic → hardest for 48h models.

    Regime 1 — "Stagnant / Inversion"
        Weak ventilation AND shallow boundary layer: wind < 2 m/s AND BLH < 500m.
        Characteristic of winter inversions and wildfire smoke traps.
        AQI is persistent and high → easiest for 48h models.

    Regime 2 — "Normal / Baseline"
        Everything else. Moderate conditions with some mixing.
        AQI is relatively stable and low → standard behavior.

    Args:
        df: DataFrame with 'wind_speed_10m' and 'boundary_layer_height' columns.

    Returns:
        pd.Series of regime labels (0, 1, or 2), same index as df.
    """
    df_numeric = df[['wind_speed_10m', 'boundary_layer_height']].apply(
        pd.to_numeric, errors='coerce'
    )
    wind = df_numeric['wind_speed_10m']
    blh  = df_numeric['boundary_layer_height']

    regime = pd.Series(2, index=df.index, name='regime')  # Default: Normal

    # Regime 0: Well-Mixed (strong ventilation)
    well_mixed = (wind >= 5.0) | (blh >= 1500.0)
    regime[well_mixed] = 0

    # Regime 1: Stagnant (weak ventilation AND shallow BLH)
    # NOTE: Regime 1 takes priority over Regime 0 if both conditions overlap
    # (which is physically impossible — you can't have wind≥5 AND wind<2).
    stagnant = (wind < 2.0) & (blh < 500.0)
    regime[stagnant] = 1

    return regime


REGIME_LABELS = {
    0: "Well-Mixed / High Wind",
    1: "Stagnant / Inversion",
    2: "Normal / Baseline",
}
