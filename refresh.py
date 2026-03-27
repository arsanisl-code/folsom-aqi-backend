"""
refresh.py — Called by cron every hour to update data/latest.json.
Also called at the end of deploy.sh to prime the cache after deployment.

Usage:
    python refresh.py
    # or via cron:
    0 * * * * /opt/folsom-aqi/venv/bin/python /opt/folsom-aqi/refresh.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from inference import predict_now


def main():
    start = datetime.now()
    print(f"[refresh] Starting at {start.isoformat()}", flush=True)

    try:
        result = predict_now()
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[refresh] Success in {elapsed:.1f}s", flush=True)
        print(f"[refresh] Generated at: {result['generated_at']}", flush=True)
        print(f"[refresh] Data freshness: {result['data_freshness_minutes']} min", flush=True)
        print(f"[refresh] Current AQI: {result['current']['aqi']} ({result['current']['category']})", flush=True)
        for h, fc in result['forecasts'].items():
            print(f"[refresh]   {h}: {fc['aqi']} AQI ({fc['category']})", flush=True)

    except Exception as exc:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[refresh] FAILED after {elapsed:.1f}s: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
