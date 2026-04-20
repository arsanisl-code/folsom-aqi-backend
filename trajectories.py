"""
trajectories.py — Lagrangian Back-Trajectory Kinematics for Folsom AQI.

Computes the estimated origin coordinates of air parcels arriving at Folsom
using a simple Euler step-back on ERA5 wind fields. No satellite fire data.

Physics (Euler step-back):
    U = -wind_speed × sin(wind_dir_rad)   [eastward transport, m/s]
    V = -wind_speed × cos(wind_dir_rad)   [northward transport, m/s]

    U_mean, V_mean = rolling mean over lookback window h (causal)

    Δlat = -(V_mean × h × 3600) / (KM_PER_DEG_LAT × 1000)
    Δlon = -(U_mean × h × 3600) / (KM_PER_DEG_LON × 1000)

    origin_lat = LAT_FOLSOM + Δlat
    origin_lon = LON_FOLSOM + Δlon

Features added (per horizon h in TRAJ_LOOKBACKS):
    traj_origin_lat_{h}h  — estimated parcel origin latitude
    traj_origin_lon_{h}h  — estimated parcel origin longitude

These encode the integrated wind transport direction over the lookback window,
providing genuine atmospheric physics signal without any satellite dependency.
"""

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

LAT_FOLSOM: float = 38.6780
LON_FOLSOM: float = -121.1761

# Lookback windows for trajectory calculation (hours)
TRAJ_LOOKBACKS: list[int] = [6, 12, 24, 48]

# Degrees per km (approximate, valid for mid-latitudes)
KM_PER_DEG_LAT: float = 111.32
KM_PER_DEG_LON: float = 111.32 * np.cos(np.radians(LAT_FOLSOM))


# ─── Core trajectory feature builder ─────────────────────────────────────────

def add_trajectory_features(
    X: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """
    Compute Lagrangian back-trajectory origin coordinates and add to X in-place.

    Args:
        X:   Feature DataFrame being built (same index as df).
        df:  Raw merged DataFrame with wind_speed_10m, wind_direction_10m.

    Adds traj_origin_lat_{h}h and traj_origin_lon_{h}h for each h in TRAJ_LOOKBACKS.
    """
    timestamps = df.index

    # Wind U/V transport components (m/s)
    # Meteorological convention: wind_dir = direction FROM which wind blows.
    # Negate to get the transport direction (where the parcel came FROM).
    wind_speed   = pd.to_numeric(df["wind_speed_10m"],     errors="coerce").fillna(0.0).values
    wind_dir     = pd.to_numeric(df["wind_direction_10m"], errors="coerce").fillna(0.0).values
    wind_dir_rad = np.radians(wind_dir)

    U = -wind_speed * np.sin(wind_dir_rad)  # eastward transport (m/s)
    V = -wind_speed * np.cos(wind_dir_rad)  # northward transport (m/s)

    for h in TRAJ_LOOKBACKS:
        dt_s = h * 3600  # seconds in lookback window

        # Rolling mean wind over the lookback window (causal — no future data)
        U_mean = pd.Series(U, index=timestamps).rolling(h, min_periods=1).mean().values
        V_mean = pd.Series(V, index=timestamps).rolling(h, min_periods=1).mean().values

        # Euler step-back: origin = Folsom - mean_wind × dt
        # Convert m/s × seconds → km → degrees
        delta_lat = -(V_mean * dt_s / 1000.0) / KM_PER_DEG_LAT
        delta_lon = -(U_mean * dt_s / 1000.0) / KM_PER_DEG_LON

        X[f"traj_origin_lat_{h}h"] = LAT_FOLSOM + delta_lat
        X[f"traj_origin_lon_{h}h"] = LON_FOLSOM + delta_lon
