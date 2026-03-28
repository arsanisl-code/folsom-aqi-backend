import glob
import os
import pandas as pd
import numpy as np
import sys

# Folsom coordinates
FOLSOM_LAT, FOLSOM_LON = 38.6780, -121.1761

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth surface."""
    R = 6371.0 # Radius of the earth in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def pad_time(t):
    """Pads integer times like 945 to 0945"""
    return str(t).zfill(4)

def main():
    print("[firms] Loading archived CSVs...")
    files = glob.glob("data/firms_archive/*.csv")
    if not files:
        print("[firms] ERROR: No files found in data/firms_archive/")
        sys.exit(1)

    df_list = []
    for file in files:
        print(f"  - Loading {os.path.basename(file)}...")
        df_list.append(pd.read_csv(file))

    fires = pd.concat(df_list, ignore_index=True)
    print(f"[firms] Total hotspots loaded: {len(fires)}")

    # 1. Parse UTC Datetimes
    fires['utc_str'] = fires['acq_date'] + ' ' + fires['acq_time'].apply(pad_time)
    fires['timestamp_utc'] = pd.to_datetime(fires['utc_str'], format='%Y-%m-%d %H%M')
    fires['timestamp_utc'] = fires['timestamp_utc'].dt.tz_localize('UTC')
    
    # 2. Floor to the start of the hour in UTC (safe from DST shifts)
    fires['timestamp_utc'] = fires['timestamp_utc'].dt.floor('h')
    
    # 3. Convert to Folsom Local Time
    fires['timestamp'] = fires['timestamp_utc'].dt.tz_convert('America/Los_Angeles')

    # 4. Calculate Distance to Folsom
    fires['distance_km'] = haversine(fires['latitude'], fires['longitude'], FOLSOM_LAT, FOLSOM_LON)

    # 5. Aggregate by local hour
    print("[firms] Aggregating by local hour...")
    hourly_fires = fires.groupby('timestamp').agg(
        fire_frp_raw=('frp', 'sum'),
        fire_count_raw=('frp', 'count'),
        fire_min_dist_raw=('distance_km', 'min')
    ).reset_index()

    hourly_fires = hourly_fires.set_index('timestamp')

    # 6. Load existing historical data
    hist_path = 'data/historical.parquet'
    print(f"[firms] Loading {hist_path}...")
    hist = pd.read_parquet(hist_path)

    # 7. Merge the new columns
    if 'fire_frp_raw' in hist.columns:
        print("[firms] FIRMS columns already exist. Dropping old ones for a clean rewrite.")
        hist = hist.drop(columns=['fire_frp_raw', 'fire_count_raw', 'fire_min_dist_raw'], errors='ignore')

    print("[firms] Merging fire data into historical index...")
    hist = hist.join(hourly_fires, how='left')

    # 8. Fill missing values (hours with no satellite passing, or no fires)
    hist['fire_frp_raw'] = hist['fire_frp_raw'].fillna(0)
    hist['fire_count_raw'] = hist['fire_count_raw'].fillna(0)
    hist['fire_min_dist_raw'] = hist['fire_min_dist_raw'].fillna(999.0)

    # Sanity Check display
    print("\n--- FIRMS Raw Aggregation Summary ---")
    print(hist[['fire_frp_raw', 'fire_count_raw', 'fire_min_dist_raw']].describe())

    # 9. Save it back
    print(f"\n[firms] Saving updated historical array back to {hist_path}...")
    hist.to_parquet(hist_path)
    print("[firms] Phase 1 (The Historical Dig) completed successfully! ✅")

if __name__ == "__main__":
    main()
