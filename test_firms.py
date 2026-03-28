import requests
import os
import sys
import io
import pandas as pd

# Coordinates for Folsom area (100-mile bounding box)
AREA = "-122.0,38.0,-120.5,39.0"
KEY = os.getenv("FIRMS_MAP_KEY", "YOUR_KEY_HERE")

def test_firms():
    print(f"Testing FIRMS API for area: {AREA}...")
    
    # We will fetch both MODIS and VIIRS for complete coverage
    sensors = ["MODIS_NRT", "VIIRS_SNPP_NRT"]
    
    for sensor in sensors:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{sensor}/{AREA}/1"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                print(f"✓ {sensor}: {len(df)} hotspots detected.")
                if len(df) > 0:
                    print(df[['latitude', 'longitude', 'frp', 'acq_date', 'acq_time']].head())
            else:
                print(f"✗ {sensor} failed: {resp.status_code} - {resp.text}")
        except Exception as exc:
            print(f"✗ {sensor} error: {exc}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        KEY = sys.argv[1]
    test_firms()
