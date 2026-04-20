"""
trajectories.py — V12 Lagrangian Back-Trajectory & Smoke Advection Features.

Physics:
    For a forecast at time T, we estimate where the air parcel arriving at
    Folsom originated by stepping backward in time using the ERA5 wind field.

    Euler step-back (per lookback window h):
        U = wind_speed × cos(wind_dir_rad)   [m/s, eastward]
        V = wind_speed × sin(wind_dir_rad)   [m/s, northward]

        Δlon = -(U_mean × h × 3600) / (111_320 × cos(lat_rad))   [degrees]
        Δlat = -(V_mean × h × 3600) / 111_320                    [degrees]

        origin_lon = LON_FOLSOM + Δlon
        origin_lat = LAT_FOLSOM + Δlat

    We then query the FIRMS fire pixel cache for fires within SEARCH_RADIUS_KM
    of the trajectory origin that were active within ±WINDOW_H hours of T-h.

Leakage guardrail:
    For a forecast at time T, we only use FIRMS data with timestamp ≤ T.
    The lookback window upper bound is clamped to T, so no future fire data
    is ever used.

    Window: [T - h - WINDOW_H,  min(T - h + WINDOW_H, T)]
    This is always ≤ T, so strictly causal.

Performance:
    Fire detections are sorted by timestamp. For each unique lookback window
    we use np.searchsorted to find the relevant fire slice in O(log N), then
    do a vectorized haversine distance check across all rows sharing that
    window. No Python loops over the 63k-row dataset.

New features added to X (per horizon h in TRAJ_LOOKBACKS):
    traj_origin_lat_{h}h        — estimated parcel origin latitude
    traj_origin_lon_{h}h        — estimated parcel origin longitude
    traj_fire_frp_sum_{h}h      — sum of FRP within SEARCH_RADIUS_KM of origin
    traj_fire_count_{h}h        — count of fire pixels near origin
    traj_fire_min_dist_{h}h     — min distance from origin to any fire pixel (km)

Smoke alignment feature (horizon-independent):
    smoke_wind_alignment        — cosine similarity between current wind vector
                                  and vector from Folsom toward the largest
                                  active fire within SMOKE_ALIGN_RADIUS_KM
                                  in the past 24h window.
                                  1.0 = perfectly downwind, 0.0 = perpendicular.
"""

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

LAT_FOLSOM: float = 38.6780
LON_FOLSOM: float = -121.1761

# Lookback windows for trajectory calculation (hours)
TRAJ_LOOKBACKS: list[int] = [6, 12, 24, 48]

# ±hours around T-h to search for fire detections.
# MODIS/VIIRS pass overhead ~4x/day (~6h cadence), so ±3h catches the
# nearest satellite pass to the parcel's estimated position time.
WINDOW_H: int = 3

# Search corridor radius around trajectory origin (km)
# Set to 150km to capture the realistic fire distribution around Folsom.
# Fires at 50-150km are the primary smoke contributors (Caldor, Mosquito,
# Creek fires were all 60-120km away). At 50km only ~11% of fire-hours
# are reachable; at 150km that rises to ~84%.
SEARCH_RADIUS_KM: float = 150.0

# Radius for smoke alignment feature (km)
SMOKE_ALIGN_RADIUS_KM: float = 200.0

# Lookback window for smoke alignment (hours) — use recent 24h fire activity
SMOKE_ALIGN_WINDOW_H: int = 24

# Earth radius (km)
EARTH_RADIUS_KM: float = 6371.0

# Degrees per km (approximate, valid for mid-latitudes)
KM_PER_DEG_LAT: float = 111.32
KM_PER_DEG_LON: float = 111.32 * np.cos(np.radians(LAT_FOLSOM))

# Nanoseconds per hour (for int64 timestamp arithmetic)
NS_PER_HOUR: np.int64 = np.int64(3_600_000_000_000)


# ─── Vectorized haversine ─────────────────────────────────────────────────────

def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized haversine distance (km).
    Supports broadcasting: lat1/lon1 can be (N,1) and lat2/lon2 (1,M)
    to produce an (N,M) distance matrix.
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─── FIRMS window query ───────────────────────────────────────────────────────

def _build_fire_arrays(firms_hourly: pd.DataFrame):
    """
    Pre-process firms_hourly into sorted numpy arrays for fast window queries.

    Returns tuple of (fire_times_ns, fire_lat, fire_lon, fire_frp, fire_count)
    all sorted by fire_times_ns ascending.
    """
    fire_times_ns = firms_hourly.index.asi8.copy()

    fire_bearing = pd.to_numeric(
        firms_hourly.get("fire_bearing_nearest", pd.Series(0.0, index=firms_hourly.index)),
        errors="coerce",
    ).fillna(0.0).values

    fire_dist_km = pd.to_numeric(
        firms_hourly.get("fire_min_dist_raw", pd.Series(999.0, index=firms_hourly.index)),
        errors="coerce",
    ).fillna(999.0).values

    bearing_rad = np.radians(fire_bearing)
    fire_lat = LAT_FOLSOM + (fire_dist_km / KM_PER_DEG_LAT) * np.cos(bearing_rad)
    fire_lon = LON_FOLSOM + (fire_dist_km / KM_PER_DEG_LON) * np.sin(bearing_rad)

    fire_frp = pd.to_numeric(
        firms_hourly["fire_frp_raw"], errors="coerce"
    ).fillna(0.0).values

    fire_count = pd.to_numeric(
        firms_hourly.get("fire_count_raw", pd.Series(0.0, index=firms_hourly.index)),
        errors="coerce",
    ).fillna(0.0).values

    # Sort by time for searchsorted
    sort_idx = np.argsort(fire_times_ns)
    return (
        fire_times_ns[sort_idx],
        fire_lat[sort_idx],
        fire_lon[sort_idx],
        fire_frp[sort_idx],
        fire_count[sort_idx],
    )


def _query_window(
    center_ns: np.int64,
    window_ns: np.int64,
    cap_ns: np.int64,
    fire_times_ns: np.ndarray,
    fire_lat: np.ndarray,
    fire_lon: np.ndarray,
    fire_frp: np.ndarray,
    fire_count: np.ndarray,
):
    """
    Return fire arrays for detections within [center_ns - window_ns,
    min(center_ns + window_ns, cap_ns)].

    Uses searchsorted for O(log N) slice — no Python loop over fire rows.
    """
    lo = center_ns - window_ns
    hi = min(center_ns + window_ns, cap_ns)
    if lo > hi:
        return None

    i0 = np.searchsorted(fire_times_ns, lo,  side="left")
    i1 = np.searchsorted(fire_times_ns, hi + 1, side="left")

    if i0 >= i1:
        return None

    return (
        fire_lat[i0:i1],
        fire_lon[i0:i1],
        fire_frp[i0:i1],
        fire_count[i0:i1],
    )


# ─── Core trajectory feature builder ─────────────────────────────────────────

def add_trajectory_features(
    X: pd.DataFrame,
    df: pd.DataFrame,
    firms_hourly: pd.DataFrame,
) -> None:
    """
    Compute Lagrangian back-trajectory features and add them to X in-place.

    Args:
        X:            Feature DataFrame being built (same index as df).
        df:           Raw merged DataFrame with wind_speed_10m, wind_direction_10m.
        firms_hourly: Hourly FIRMS DataFrame indexed by timestamp with columns
                      fire_frp_raw, fire_count_raw, fire_min_dist_raw,
                      fire_bearing_nearest (as produced by data_fetcher).
                      May be empty — features gracefully degrade to zero/999.
    """
    n = len(df)
    timestamps = df.index  # DatetimeIndex, tz-aware
    ts_ns = timestamps.asi8  # int64 nanoseconds, shape (n,)

    # ── Wind U/V components (m/s) ─────────────────────────────────────────
    wind_speed   = pd.to_numeric(df["wind_speed_10m"],    errors="coerce").fillna(0.0).values
    wind_dir     = pd.to_numeric(df["wind_direction_10m"], errors="coerce").fillna(0.0).values
    wind_dir_rad = np.radians(wind_dir)

    # Meteorological convention: wind_dir = direction FROM which wind blows.
    # Negate to get the transport direction (where the parcel came FROM).
    U = -wind_speed * np.sin(wind_dir_rad)  # eastward transport (m/s)
    V = -wind_speed * np.cos(wind_dir_rad)  # northward transport (m/s)

    # ── Pre-process FIRMS ─────────────────────────────────────────────────
    has_firms = (
        firms_hourly is not None
        and not firms_hourly.empty
        and "fire_frp_raw" in firms_hourly.columns
    )

    if has_firms:
        fire_times_ns, fire_lat, fire_lon, fire_frp, fire_count = _build_fire_arrays(firms_hourly)
        window_ns = np.int64(WINDOW_H) * NS_PER_HOUR
        align_window_ns = np.int64(SMOKE_ALIGN_WINDOW_H) * NS_PER_HOUR
    else:
        fire_times_ns = fire_lat = fire_lon = fire_frp = fire_count = None
        window_ns = align_window_ns = None

    # ── Per-lookback trajectory features ─────────────────────────────────
    for h in TRAJ_LOOKBACKS:
        dt_s = h * 3600  # seconds in lookback window

        # Rolling mean wind over the lookback window (causal — no future data)
        U_mean = pd.Series(U, index=timestamps).rolling(h, min_periods=1).mean().values
        V_mean = pd.Series(V, index=timestamps).rolling(h, min_periods=1).mean().values

        # Euler step-back: origin = Folsom - mean_wind × dt
        delta_lat = -(V_mean * dt_s / 1000.0) / KM_PER_DEG_LAT
        delta_lon = -(U_mean * dt_s / 1000.0) / KM_PER_DEG_LON

        origin_lat = LAT_FOLSOM + delta_lat
        origin_lon = LON_FOLSOM + delta_lon

        X[f"traj_origin_lat_{h}h"] = origin_lat
        X[f"traj_origin_lon_{h}h"] = origin_lon

        if not has_firms:
            X[f"traj_fire_frp_sum_{h}h"]  = 0.0
            X[f"traj_fire_count_{h}h"]    = 0.0
            X[f"traj_fire_min_dist_{h}h"] = 999.0
            continue

        # ── Spatial intersection with ±WINDOW_H time window ───────────────
        # For each row i at time T_i:
        #   center = T_i - h  (when the parcel was at origin_i)
        #   window = [center - WINDOW_H, min(center + WINDOW_H, T_i)]
        #   query FIRMS for fires in that window, check distance to origin_i
        #
        # We group rows by their unique lookback center (T_i - h) to avoid
        # redundant searchsorted calls for rows sharing the same center.

        lookback_ns = np.int64(h) * NS_PER_HOUR
        center_ns_arr = ts_ns - lookback_ns  # shape (n,)

        frp_sum_arr  = np.zeros(n, dtype=np.float32)
        count_arr    = np.zeros(n, dtype=np.float32)
        min_dist_arr = np.full(n, 999.0, dtype=np.float32)

        # Group rows by unique center timestamp
        unique_centers, center_inverse = np.unique(center_ns_arr, return_inverse=True)

        for ci, center_val in enumerate(unique_centers):
            row_idxs = np.where(center_inverse == ci)[0]

            # cap_ns = T_i (use the first row's T — all rows in this group
            # share the same center, so T = center + h is also the same)
            cap_val = center_val + lookback_ns  # = T_i

            result = _query_window(
                center_val, window_ns, cap_val,
                fire_times_ns, fire_lat, fire_lon, fire_frp, fire_count,
            )
            if result is None:
                continue

            f_lat_w, f_lon_w, f_frp_w, f_cnt_w = result

            # Vectorized distance: (R, F) matrix
            o_lat = origin_lat[row_idxs, np.newaxis]   # (R, 1)
            o_lon = origin_lon[row_idxs, np.newaxis]   # (R, 1)
            f_lat_b = f_lat_w[np.newaxis, :]            # (1, F)
            f_lon_b = f_lon_w[np.newaxis, :]            # (1, F)

            dist_mat = _haversine_km(o_lat, o_lon, f_lat_b, f_lon_b)  # (R, F)
            within   = dist_mat <= SEARCH_RADIUS_KM                    # (R, F)

            # Aggregate per row — vectorized where possible
            any_within = within.any(axis=1)  # (R,)
            hit_rows = row_idxs[any_within]

            if len(hit_rows) == 0:
                continue

            hit_within  = within[any_within]          # (H, F)
            hit_dist    = dist_mat[any_within]         # (H, F)

            # FRP sum: dot product of within mask and frp values
            frp_sum_arr[hit_rows]  = (hit_within * f_frp_w[np.newaxis, :]).sum(axis=1)
            count_arr[hit_rows]    = (hit_within * f_cnt_w[np.newaxis, :]).sum(axis=1)
            # Min distance: mask non-within with large value, then take min
            masked_dist = np.where(hit_within, hit_dist, 9999.0)
            min_dist_arr[hit_rows] = masked_dist.min(axis=1)

        X[f"traj_fire_frp_sum_{h}h"]  = frp_sum_arr
        X[f"traj_fire_count_{h}h"]    = count_arr
        X[f"traj_fire_min_dist_{h}h"] = min_dist_arr

    # ── Smoke wind alignment feature ──────────────────────────────────────
    # For each timestamp T, find the largest fire within SMOKE_ALIGN_RADIUS_KM
    # that was active in the past SMOKE_ALIGN_WINDOW_H hours (causal: window
    # upper bound = T, so no future data).
    # Then compute cosine similarity between current wind and Folsom→fire vector.
    if has_firms:
        smoke_align = np.zeros(n, dtype=np.float32)

        # Group rows by unique T (all unique since hourly index, but handle
        # duplicates gracefully)
        unique_ts, ts_inverse = np.unique(ts_ns, return_inverse=True)

        for ti, ts_val in enumerate(unique_ts):
            row_idxs = np.where(ts_inverse == ti)[0]

            # Window: [T - SMOKE_ALIGN_WINDOW_H, T]  — strictly causal
            result = _query_window(
                ts_val, align_window_ns, ts_val,
                fire_times_ns, fire_lat, fire_lon, fire_frp, fire_count,
            )
            if result is None:
                continue

            f_lat_w, f_lon_w, f_frp_w, _ = result

            # Distance from Folsom to each fire in window
            dist_folsom = _haversine_km(
                np.float32(LAT_FOLSOM), np.float32(LON_FOLSOM),
                f_lat_w, f_lon_w,
            )
            nearby = dist_folsom <= SMOKE_ALIGN_RADIUS_KM
            if not nearby.any():
                continue

            # Largest fire within radius (by FRP)
            masked_frp = np.where(nearby, f_frp_w, -1.0)
            best = int(np.argmax(masked_frp))
            if not nearby[best]:
                continue

            # Unit vector from Folsom toward the fire
            dlat = (f_lat_w[best] - LAT_FOLSOM) * KM_PER_DEG_LAT
            dlon = (f_lon_w[best] - LON_FOLSOM) * KM_PER_DEG_LON
            fire_norm = np.sqrt(dlat**2 + dlon**2) + 1e-9
            fire_unit_n = dlat / fire_norm   # northward component
            fire_unit_e = dlon / fire_norm   # eastward component

            # Cosine similarity for each row in this batch (vectorized)
            u_batch = U[row_idxs]
            v_batch = V[row_idxs]
            wind_norms = np.sqrt(u_batch**2 + v_batch**2) + 1e-9
            # dot(wind_unit, fire_unit): U=east, V=north
            cos_sim = (u_batch * fire_unit_e + v_batch * fire_unit_n) / wind_norms
            smoke_align[row_idxs] = np.maximum(0.0, cos_sim).astype(np.float32)

        X["smoke_wind_alignment"] = smoke_align
    else:
        X["smoke_wind_alignment"] = 0.0
