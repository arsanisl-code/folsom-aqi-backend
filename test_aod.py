import requests

def test_aod_logic():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 38.6780,
        "longitude": -121.1761,
        "hourly": "pm2_5,aerosol_optical_depth",
        "start_date": "2023-08-01",
        "end_date": "2023-08-02",
        "timezone": "America/Los_Angeles",
    }
    try:
        resp = requests.get(url, params=params)
        print(resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            print("AOD length:", len(data["hourly"].get("aerosol_optical_depth", [])))
        else:
            print(resp.text)
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_aod_logic()
