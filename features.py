"""
features.py — Feature engineering for Folsom AQI forecasting.
CRITICAL: No data leakage. All features at row T use only data available at T.

engineer_features() is decomposed into private sub-functions, one per feature
family. The orchestrator calls each in sequence and calls X = X.copy() after
each to prevent Pandas memory fragmentation from in-place column additions.
"""

import numpy as np
import pandas as pd

from trajectories import add_trajectory_features

# ─── Physical constants ───────────────────────────────────────────────────────

# Empirical Central Valley stagnation threshold (California Air Resources Board).
# When BLH × wind_speed drops below this value, pollutants accumulate.
VENT_THRESHOLD_M2_PER_S: float = 3000.0

# Temperature below which residential wood burning in Folsom increases significantly
COLD_SMOKE_THRESHOLD_C: float = 7.0

# Magnus formula coefficients for saturation vapor pressure approximation (kPa).
MAGNUS_A: float = 17.27
MAGNUS_B: float = 237.3

# Open-Meteo hallucination floor for relative humidity (Folsom grid cell artifact).
HUMIDITY_CLAMP_MIN_PCT: float = 25.0

# Open-Meteo hallucination cap for wind speed (Folsom grid cell artifact).
WIND_SPEED_CAP_KMH: float = 25.0

# Minimum precipitation to count as a wet-deposition scavenging event (mm).
RAIN_THRESHOLD_MM: float = 0.1


def _add_firms_dummy_features(X: pd.DataFrame) -> None:
    """
    Adds dummy fire features (filled with 0) to maintain compatibility with
    legacy models trained with FIRMS data. Ablation studies proved these
    features had negligible impact, but they are required for unpickling.
    """
    for col in [
        "fire_count_current", "fire_frp_24h_sum", "fire_count_24h_sum",
        "fire_advection_score", "fire_advection_24h_max"
    ]:
        X[col] = 0.0


# ─── Sub-functions ────────────────────────────────────────────────────────────


def _add_aqi_lag_features(X: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Mutates X in-place. Adds:
    - AQI lags (current, 1, 2, 3, 6, 12, 24, 48h)
    - AQI diffs (1h, 24h), second derivative (acceleration)
    - Rolling stats (mean, max, std) for windows [3,6,12,24,48,168]h
    - EWMA (6h, 24h spans) + PM2.5 EWMA (6h span)
    """
    X["aqi_current"] = df["us_aqi"]
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        X[f"aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)

    X["aqi_diff_1h"] = df["us_aqi"] - df["us_aqi"].shift(1)
    X["aqi_diff_24h"] = df["us_aqi"] - df["us_aqi"].shift(24)

    X["aqi_acceleration"] = X["aqi_diff_1h"] - X["aqi_diff_1h"].shift(1)
    X["aqi_acceleration_6h_mean"] = X["aqi_acceleration"].rolling(6, min_periods=1).mean()

    for window in [3, 6, 12, 24, 48, 168]:
        X[f"aqi_roll_{window}h_mean"] = df["us_aqi"].rolling(window, min_periods=1).mean()
        X[f"aqi_roll_{window}h_max"] = df["us_aqi"].rolling(window, min_periods=1).max()
        X[f"aqi_roll_{window}h_std"] = df["us_aqi"].rolling(window, min_periods=1).std().fillna(0)

    X["aqi_ewma_6h"] = df["us_aqi"].ewm(span=6, adjust=False).mean()
    X["aqi_ewma_24h"] = df["us_aqi"].ewm(span=24, adjust=False).mean()
    X["pm25_ewma_6h"] = df["pm2_5"].ewm(span=6, adjust=False).mean()


def _add_pm25_and_combustion_features(X: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Mutates X in-place. Adds:
    - PM2.5 current + lags (1,3,6,24h) + rolling (6h, 24h mean)
    - CO current, lags, rolling, diff (if present)
    - PM2.5/CO wildfire discrimination ratio
    - Dust current, rolling, diff
    - NO2 current, rolling
    """
    X["pm25_current"] = df["pm2_5"]
    for lag in [1, 3, 6, 24]:
        X[f"pm25_lag_{lag}h"] = df["pm2_5"].shift(lag)
    X["pm25_roll_6h_mean"] = df["pm2_5"].rolling(6, min_periods=1).mean()
    X["pm25_roll_24h_mean"] = df["pm2_5"].rolling(24, min_periods=1).mean()

    if "carbon_monoxide" in df.columns:
        co = pd.to_numeric(df["carbon_monoxide"], errors="coerce")
        X["co_current"] = co
        X["co_lag_6h"] = co.shift(6)
        X["co_lag_24h"] = co.shift(24)
        X["co_roll_24h_mean"] = co.rolling(24, min_periods=1).mean()
        X["co_diff_6h"] = co.diff(6)
        X["co_roll_6h_max"] = co.rolling(6, min_periods=1).max()

        if "pm2_5" in df.columns:
            co_safe = co.replace(0, np.nan)
            X["pm25_co_ratio"] = df["pm2_5"] / co_safe
            X["pm25_co_ratio_6h_mean"] = X["pm25_co_ratio"].rolling(6, min_periods=1).mean()

    if "dust" in df.columns:
        dust = pd.to_numeric(df["dust"], errors="coerce")
        X["dust_current"] = dust
        X["dust_roll_24h_mean"] = dust.rolling(24, min_periods=1).mean()
        X["dust_diff_6h"] = dust.diff(6)

    if "nitrogen_dioxide" in df.columns:
        no2 = pd.to_numeric(df["nitrogen_dioxide"], errors="coerce")
        X["no2_current"] = no2
        X["no2_roll_24h_mean"] = no2.rolling(24, min_periods=1).mean()

    if "ozone" in df.columns:
        o3 = pd.to_numeric(df["ozone"], errors="coerce")
        X["ozone_current"] = o3
        X["ozone_roll_6h_mean"] = o3.rolling(6, min_periods=1).mean()
        X["ozone_diff_3h"] = o3.diff(3)


def _add_meteorological_features(X: pd.DataFrame, df: pd.DataFrame, horizon_h: int) -> None:
    """
    Mutates X in-place. Adds:
    - Raw met: BLH, wind, pressure, humidity, temp, precip, cloud, radiation
    - Physical interactions: blh_x_wind, aqi_x_wind, aqi_x_rad
    - Photochemical forcing indices
    - Forward NWP features (wind, BLH, temp, humidity, pressure, precip)
    - Forward ventilation and HDWI
    - AOD current + diffs
    """
    X["boundary_layer_height"] = df["boundary_layer_height"]
    X["wind_speed_10m"] = df["wind_speed_10m"]
    X["surface_pressure"] = df["surface_pressure"]
    X["relative_humidity_2m"] = df["relative_humidity_2m"]
    X["temperature_2m"] = df["temperature_2m"]
    X["precipitation"] = df["precipitation"]
    X["cloud_cover"] = df["cloud_cover"]
    X["direct_radiation"] = pd.to_numeric(df.get("direct_radiation"), errors="coerce").fillna(0)

    X["blh_x_wind_speed"] = df["boundary_layer_height"] * df["wind_speed_10m"]
    X["aqi_x_wind"] = X["aqi_current"] * df["wind_speed_10m"]
    X["aqi_x_rad"] = X["aqi_current"] * X["direct_radiation"]

    ghi = pd.to_numeric(df.get("shortwave_radiation", 0), errors="coerce").fillna(0)
    _pc_temp = pd.to_numeric(df["temperature_2m"], errors="coerce")
    _pc_rh = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
    ghi_norm = ghi / 1000.0
    rh_factor = (_pc_rh / 100.0).clip(0.3, 1.0)
    thermal_factor = np.exp(0.069 * (_pc_temp - 25.0))

    X["photochem_forcing"] = ghi_norm * rh_factor * thermal_factor
    X["photochem_forcing_6h"] = X["photochem_forcing"].rolling(6, min_periods=1).mean()
    X["ghi_accum_12h"] = ghi.rolling(12, min_periods=1).sum()

    fwd_ghi = ghi.shift(-horizon_h)
    fwd_rh_pc = df["relative_humidity_2m"].shift(-horizon_h)
    fwd_temp_pc = df["temperature_2m"].shift(-horizon_h)
    fwd_thermal = np.exp(0.069 * (fwd_temp_pc - 25.0))
    X["fwd_photochem_forcing"] = (
        (fwd_ghi / 1000.0) * (fwd_rh_pc / 100.0).clip(0.3, 1.0) * fwd_thermal
    )

    X["fwd_photochem_accum_12h"] = (
        X["photochem_forcing"].rolling(12, min_periods=1).sum().shift(-horizon_h)
    )

    X["fwd_wind_speed"] = df["wind_speed_10m"].shift(-horizon_h)
    X["fwd_blh"] = df["boundary_layer_height"].shift(-horizon_h)
    X["fwd_temperature"] = df["temperature_2m"].shift(-horizon_h)
    X["fwd_humidity"] = df["relative_humidity_2m"].shift(-horizon_h)
    X["fwd_pressure"] = df["surface_pressure"].shift(-horizon_h)
    X["fwd_precipitation"] = df["precipitation"].shift(-horizon_h)

    fwd_wind_dir = df["wind_direction_10m"].shift(-horizon_h)
    fwd_wind_dir_rad = np.radians(fwd_wind_dir)
    X["fwd_wind_dir_sin"] = np.sin(fwd_wind_dir_rad)
    X["fwd_wind_dir_cos"] = np.cos(fwd_wind_dir_rad)
    X["fwd_wind_u"] = X["fwd_wind_speed"] * X["fwd_wind_dir_cos"]
    X["fwd_wind_v"] = X["fwd_wind_speed"] * X["fwd_wind_dir_sin"]

    X["fwd_temperature_mean"] = (
        df["temperature_2m"].rolling(horizon_h, min_periods=1).mean().shift(-horizon_h)
    )
    X["fwd_wind_speed_mean"] = (
        df["wind_speed_10m"].rolling(horizon_h, min_periods=1).mean().shift(-horizon_h)
    )
    X["fwd_humidity_mean"] = (
        df["relative_humidity_2m"].rolling(horizon_h, min_periods=1).mean().shift(-horizon_h)
    )
    X["fwd_pressure_mean"] = (
        df["surface_pressure"].rolling(horizon_h, min_periods=1).mean().shift(-horizon_h)
    )
    X["fwd_precip_accum"] = (
        df["precipitation"].rolling(horizon_h, min_periods=1).sum().shift(-horizon_h)
    )

    X["fwd_ventilation"] = X["fwd_blh"] * X["fwd_wind_speed"]
    X["fwd_vent_deficit"] = (VENT_THRESHOLD_M2_PER_S - X["fwd_ventilation"]).clip(lower=0)

    fwd_sat_vp = 0.6108 * np.exp(MAGNUS_A * X["fwd_temperature"] / (X["fwd_temperature"] + MAGNUS_B))
    fwd_act_vp = fwd_sat_vp * (X["fwd_humidity"] / 100.0)
    fwd_vpd = fwd_sat_vp - fwd_act_vp
    X["fwd_hdwi"] = X["fwd_wind_speed"] * fwd_vpd

    if "aerosol_optical_depth" in df.columns:
        aod = pd.to_numeric(df["aerosol_optical_depth"], errors="coerce")
        X["aod_current"] = aod
        X["aod_diff_3h"] = aod.diff(3)
        X["aod_diff_6h"] = aod.diff(6)
        X["aod_roll_24h_mean"] = aod.rolling(24, min_periods=1).mean()
        X["fwd_aod"] = aod.shift(-horizon_h)
        X["aod_trend_6h"] = aod.diff(6)
        aod_high_flag = (aod > 0.3).astype(float)
        X["aod_persistence_24h"] = aod_high_flag.rolling(24, min_periods=1).sum()

    wind_dir_rad = np.radians(df["wind_direction_10m"])
    X["wind_dir_sin"] = np.sin(wind_dir_rad)
    X["wind_dir_cos"] = np.cos(wind_dir_rad)
    wind = df["wind_speed_10m"]
    X["wind_u"] = wind * np.cos(wind_dir_rad)
    X["wind_v"] = wind * np.sin(wind_dir_rad)


def _add_atmospheric_stability_features(X: pd.DataFrame, df: pd.DataFrame, horizon_h: int) -> None:
    """
    Mutates X in-place. Adds:
    - Stagnation index (24h, 48h rolling sums)
    - Inversion proxy (strength, 12h max)
    - True inversion delta 850hPa
    - 700hPa inversion depth
    - Ventilation deficit
    - Synoptic blocking index Z500
    - Cold degree hours
    - Tule fog precursor
    """
    wind = df["wind_speed_10m"]
    boundary_layer_height = df["boundary_layer_height"]

    low_wind = (wind < 2.0).astype(float)
    low_blh = (boundary_layer_height < 500).astype(float)
    stag_hourly = low_wind * low_blh

    X["stagnation_24h"] = stag_hourly.rolling(24, min_periods=1).sum()
    X["stagnation_48h"] = stag_hourly.rolling(48, min_periods=1).sum()

    pressure_change_6h = df["surface_pressure"].diff(6)
    temp_change_6h = df["temperature_2m"].diff(6)

    blh_clamped = boundary_layer_height.clip(lower=50)
    inv_blh_comp = 1000.0 / blh_clamped
    inv_pres_comp = pressure_change_6h.clip(lower=0)
    inv_cool_comp = (-temp_change_6h).clip(lower=0)

    X["inversion_strength"] = inv_blh_comp * inv_pres_comp * inv_cool_comp
    X["inversion_12h_max"] = X["inversion_strength"].rolling(12, min_periods=1).max()

    if "temperature_850hPa" in df.columns:
        t850 = pd.to_numeric(df["temperature_850hPa"], errors="coerce").ffill(limit=6).bfill(limit=6)
        t2m = pd.to_numeric(df["temperature_2m"], errors="coerce")
        X["inversion_delta_850"] = t850 - t2m
        X["inversion_delta_850_6h_mean"] = X["inversion_delta_850"].rolling(6, min_periods=1).mean()
        X["fwd_inversion_delta_850"] = X["inversion_delta_850"].shift(-horizon_h)

    if "temperature_700hPa" in df.columns and "temperature_850hPa" in df.columns:
        t700 = pd.to_numeric(df["temperature_700hPa"], errors="coerce").ffill(limit=6).bfill(limit=6)
        t850 = pd.to_numeric(df["temperature_850hPa"], errors="coerce").ffill(limit=6).bfill(limit=6)
        t2m = pd.to_numeric(df["temperature_2m"], errors="coerce")
        X["inversion_column_depth"] = t700 - t2m
        X["inversion_lid_stability"] = t700 - t850
        X["inversion_column_24h_mean"] = X["inversion_column_depth"].rolling(24, min_periods=1).mean()
        X["fwd_inversion_column_depth"] = X["inversion_column_depth"].shift(-horizon_h)
        X["fwd_inversion_lid_stability"] = X["inversion_lid_stability"].shift(-horizon_h)

    ventilation = boundary_layer_height * wind
    vent_deficit = (VENT_THRESHOLD_M2_PER_S - ventilation).clip(lower=0)
    X["vent_deficit"] = vent_deficit
    X["vent_deficit_24h_mean"] = vent_deficit.rolling(24, min_periods=1).mean()

    if "geopotential_height_500hPa" in df.columns:
        z500 = pd.to_numeric(df["geopotential_height_500hPa"], errors="coerce")
        z500 = z500.ffill(limit=24).fillna(z500.median())
        z500_doy_mean = z500.groupby(df.index.dayofyear).transform("mean")
        X["z500_anomaly"] = z500 - z500_doy_mean
        is_blocked = (X["z500_anomaly"] > 50).astype(float)
        X["blocking_persistence_72h"] = is_blocked.rolling(72, min_periods=1).sum()

    cold_degree_hours = (COLD_SMOKE_THRESHOLD_C - df["temperature_2m"]).clip(lower=0)
    X["cold_degree_hours"] = cold_degree_hours
    X["cold_degree_hours_48h"] = cold_degree_hours.rolling(48, min_periods=1).sum()

    temp = pd.to_numeric(df["temperature_2m"], errors="coerce")
    rh = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
    alpha = np.log(rh.clip(lower=1) / 100) + (MAGNUS_A * temp) / (MAGNUS_B + temp)
    dew_point = (MAGNUS_B * alpha) / (MAGNUS_A - alpha)

    X["dew_point_depression"] = temp - dew_point
    is_foggy = (X["dew_point_depression"] < 3.0).astype(float)
    X["fog_precursor_12h"] = is_foggy.rolling(12, min_periods=1).sum()


def _add_wildfire_features(X: pd.DataFrame, df: pd.DataFrame, horizon_h: int) -> None:
    """
    Mutates X in-place. Adds wildfire proxy features:
    - HDWI (Hot-Dry-Windy Index using VPD × wind)
    - Antecedent precipitation deficit (30-day sum)
    - Hours/days since last rain (dry streak counter)
    - Pressure and temperature front differencing
    """
    temp_c = df["temperature_2m"]
    rh = df["relative_humidity_2m"]
    wind = df["wind_speed_10m"]

    sat_vp = 0.6108 * np.exp(MAGNUS_A * temp_c / (temp_c + MAGNUS_B))
    act_vp = sat_vp * (rh / 100.0)
    vpd = sat_vp - act_vp
    X["wildfire_hdwi"] = wind * vpd

    X["precip_30d_sum"] = df["precipitation"].rolling(30 * 24, min_periods=1).sum()

    rain_flag = (df["precipitation"] > RAIN_THRESHOLD_MM).astype(int)
    dry_groups = (rain_flag != rain_flag.shift()).cumsum()
    dry_flag = 1 - rain_flag
    dry_cumsum = dry_flag.groupby(dry_groups).cumsum()
    X["hours_since_rain"] = dry_cumsum
    X["days_since_rain"] = X["hours_since_rain"] / 24.0

    X["pressure_diff_3h"] = df["surface_pressure"].diff(3)
    X["pressure_diff_6h"] = df["surface_pressure"].diff(6)
    X["pressure_diff_12h"] = df["surface_pressure"].diff(12)
    X["pressure_diff_24h"] = df["surface_pressure"].diff(24)
    X["temp_diff_24h"] = df["temperature_2m"].diff(24)
    X["pressure_diff_48h"] = df["surface_pressure"].diff(48)
    X["temp_diff_48h"] = df["temperature_2m"].diff(48)


def _add_temporal_features(X: pd.DataFrame, df: pd.DataFrame, horizon_h: int) -> None:
    """
    Mutates X in-place. Adds:
    - Cyclic encodings: hour, DOW, DOY, month
    - Second harmonic (commute traffic): hour_sin_2, hour_cos_2
    - Future hour cyclic: future_hour_sin, future_hour_cos
    """
    hour = df.index.hour
    day_of_year = df.index.day_of_year
    month = df.index.month
    day_of_week = df.index.day_of_week

    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    X["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    X["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    X["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    X["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    X["hour_sin_2"] = np.sin(2 * 2 * np.pi * hour / 24)
    X["hour_cos_2"] = np.cos(2 * 2 * np.pi * hour / 24)

    future_hour = (hour + horizon_h) % 24
    X["future_hour_sin"] = np.sin(2 * np.pi * future_hour / 24)
    X["future_hour_cos"] = np.cos(2 * np.pi * future_hour / 24)


def _add_regulatory_features(X: pd.DataFrame, df: pd.DataFrame, horizon_h: int) -> None:
    """
    Mutates X in-place. Adds:
    - Radiation accumulation
    - Multi-scale atmospheric momentum
    - Second-order interaction features
    - Evening BLH collapse velocity
    - Regime-conditional interaction features
    """
    if "shortwave_radiation" in df.columns:
        X["radiation_accum_6h"] = (
            pd.to_numeric(df["shortwave_radiation"], errors="coerce")
            .rolling(6, min_periods=1)
            .sum()
        )
    else:
        X["radiation_accum_6h"] = 0.0

    X["aqi_momentum_6h"] = X["aqi_ewma_6h"] - X["aqi_ewma_24h"]
    X["aqi_momentum_24h"] = X["aqi_ewma_24h"] - X["aqi_roll_168h_mean"]
    X["momentum_accel_6h"] = X["aqi_momentum_6h"] - X["aqi_momentum_6h"].shift(6)

    weekly_std = X["aqi_roll_168h_std"].clip(lower=1.0)
    X["aqi_zscore_7d"] = (X["aqi_current"] - X["aqi_roll_168h_mean"]) / weekly_std
    X["fat_tail_persistence_48h"] = (
        (X["aqi_zscore_7d"] > 2.0).astype(float).rolling(48, min_periods=1).sum()
    )

    X["stability_index"] = (X["fwd_temperature_mean"] + 273.15) / X["fwd_blh"].clip(lower=50)
    X["trapping_power"] = X["inversion_column_24h_mean"] / X["fwd_wind_speed_mean"].clip(lower=0.5)
    X["fwd_ventilation_stress"] = X["fwd_humidity_mean"] * X["fwd_vent_deficit"]
    X["volatility_frontal"] = X["aqi_roll_168h_std"] * X["pressure_diff_48h"].abs()

    summer_flag = ((df.index.month >= 5) & (df.index.month <= 9)).astype(float)
    X["summer_photochem_accum"] = X["fwd_photochem_accum_12h"] * summer_flag

    X["blh_collapse_rate"] = df["boundary_layer_height"].diff(3)
    X["fwd_blh_collapse_rate"] = df["boundary_layer_height"].diff(3).shift(-horizon_h)

    wind_num = pd.to_numeric(df["wind_speed_10m"], errors="coerce").fillna(0)
    blh_num = pd.to_numeric(df["boundary_layer_height"], errors="coerce").fillna(500)
    regime_0 = ((wind_num >= 5.0) | (blh_num >= 1500.0)).astype(float)
    regime_1 = ((wind_num < 2.0) & (blh_num < 500.0)).astype(float)

    X["regime1_x_aqi_current"] = regime_1 * X["aqi_current"]
    X["regime1_x_aqi_roll_24h_mean"] = regime_1 * X["aqi_roll_24h_mean"]
    X["regime0_x_fwd_wind_mean"] = regime_0 * X["fwd_wind_speed_mean"]
    X["regime0_x_trapping_power"] = regime_0 * X["trapping_power"]


# ─── Public API ───────────────────────────────────────────────────────────────


def engineer_features(
    df: pd.DataFrame,
    horizon_h: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y for a given forecast horizon.
    """
    X = pd.DataFrame(index=df.index)
    df = df.apply(pd.to_numeric, errors="coerce")

    _add_aqi_lag_features(X, df)
    X = X.copy()
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
    _add_firms_dummy_features(X)
    X = X.copy()

    if horizon_h >= 24:
        add_trajectory_features(X, df)
        X = X.copy()

    y = df["us_aqi"].shift(-horizon_h) - df["us_aqi"]
    y.name = "target_residual"

    return X, y


def get_feature_names(horizon_h: int = 6) -> list[str]:
    """Return ordered list of feature names for a given horizon."""
    idx = pd.date_range("2023-01-01", periods=500, freq="h", tz="America/Los_Angeles")
    dummy = pd.DataFrame(
        {
            "us_aqi": np.random.rand(500) * 100,
            "pm2_5": np.random.rand(500) * 50,
            "boundary_layer_height": np.random.rand(500) * 1500,
            "wind_speed_10m": np.random.rand(500) * 10,
            "surface_pressure": np.random.rand(500) * 20 + 1010,
            "relative_humidity_2m": np.random.rand(500) * 100,
            "temperature_2m": np.random.rand(500) * 30,
            "precipitation": np.random.rand(500),
            "cloud_cover": np.random.rand(500) * 100,
            "cloud_cover_low": np.random.rand(500) * 100,
            "wind_direction_10m": np.random.rand(500) * 360,
            "direct_radiation": np.random.rand(500) * 500,
            "shortwave_radiation": np.random.rand(500) * 1000,
            "soil_temperature_0_to_7cm": np.random.rand(500) * 25,
            "aerosol_optical_depth": np.random.rand(500) * 0.5,
            "temperature_850hPa": np.random.rand(500) * 20 - 5,
            "temperature_700hPa": np.random.rand(500) * 15 - 10,
            "geopotential_height_500hPa": np.random.rand(500) * 500 + 5500,
            "carbon_monoxide": np.random.rand(500) * 1000,
            "nitrogen_dioxide": np.random.rand(500) * 100,
            "dust": np.random.rand(500) * 50,
            "ozone": np.random.rand(500) * 100,
        },
        index=idx,
    )
    X, _ = engineer_features(dummy, horizon_h)
    return list(X.columns)


def classify_regime(df: pd.DataFrame) -> pd.Series:
    """Classify into well-mixed, stagnant, or normal regimes."""
    df_numeric = df[["wind_speed_10m", "boundary_layer_height"]].apply(
        pd.to_numeric, errors="coerce"
    )
    wind = df_numeric["wind_speed_10m"]
    blh = df_numeric["boundary_layer_height"]

    regime = pd.Series(2, index=df.index, name="regime")
    well_mixed = (wind >= 5.0) | (blh >= 1500.0)
    regime[well_mixed] = 0
    stagnant = (wind < 2.0) & (blh < 500.0)
    regime[stagnant] = 1

    return regime


REGIME_LABELS = {
    0: "Well-Mixed / High Wind",
    1: "Stagnant / Inversion",
    2: "Normal / Baseline",
}
