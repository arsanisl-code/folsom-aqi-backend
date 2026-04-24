import numpy as np
import pandas as pd

from trajectories import LAT_FOLSOM, LON_FOLSOM, TRAJ_LOOKBACKS, add_trajectory_features


def test_add_trajectory_features_west_wind():
    """
    Test scenario: Steady wind FROM the West (270 degrees).
    Transport is to the East.
    Therefore, the parcel origin should be to the West of Folsom.
    (Longitude < Folsom Longitude, Latitude ≈ Folsom Latitude)
    """
    idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="America/Los_Angeles")
    df = pd.DataFrame(
        {
            "wind_speed_10m": [10.0] * 10,  # 10 m/s
            "wind_direction_10m": [270.0] * 10,  # From West
        },
        index=idx,
    )

    X = pd.DataFrame(index=idx)

    add_trajectory_features(X, df)

    for h in TRAJ_LOOKBACKS:
        lat_col = f"traj_origin_lat_{h}h"
        lon_col = f"traj_origin_lon_{h}h"

        assert lat_col in X.columns
        assert lon_col in X.columns

        last_lon = X[lon_col].iloc[-1]
        last_lat = X[lat_col].iloc[-1]

        # Origin should be west (more negative longitude)
        assert last_lon < LON_FOLSOM
        # Latitude shouldn't change much for pure westerly wind
        np.testing.assert_almost_equal(last_lat, LAT_FOLSOM, decimal=2)


def test_add_trajectory_features_south_wind():
    """
    Test scenario: Steady wind FROM the South (180 degrees).
    Transport is to the North.
    Therefore, the parcel origin should be to the South of Folsom.
    (Latitude < Folsom Latitude, Longitude ≈ Folsom Longitude)
    """
    idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="America/Los_Angeles")
    df = pd.DataFrame({"wind_speed_10m": [5.0] * 10, "wind_direction_10m": [180.0] * 10}, index=idx)

    X = pd.DataFrame(index=idx)
    add_trajectory_features(X, df)

    last_lat = X["traj_origin_lat_6h"].iloc[-1]
    last_lon = X["traj_origin_lon_6h"].iloc[-1]

    assert last_lat < LAT_FOLSOM
    np.testing.assert_almost_equal(last_lon, LON_FOLSOM, decimal=2)


def test_add_trajectory_features_calm():
    """
    Test scenario: Calm wind (0 m/s).
    Parcel origin should be exactly at Folsom.
    """
    idx = pd.date_range("2023-01-01", periods=5, freq="h", tz="America/Los_Angeles")
    df = pd.DataFrame({"wind_speed_10m": [0.0] * 5, "wind_direction_10m": [0.0] * 5}, index=idx)

    X = pd.DataFrame(index=idx)
    add_trajectory_features(X, df)

    last_lat = X["traj_origin_lat_6h"].iloc[-1]
    last_lon = X["traj_origin_lon_6h"].iloc[-1]

    assert last_lat == LAT_FOLSOM
    assert last_lon == LON_FOLSOM
