"""
refresh.py — Called by GitHub Actions every hour to update data/latest.json.
Also called at the end of deploy.sh to prime the cache after deployment.

Usage:
    python refresh.py
    # or via cron:
    0 * * * * /opt/folsom-aqi/venv/bin/python /opt/folsom-aqi/refresh.py
"""

import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from inference import predict_now
from logger import get_logger

log = get_logger(__name__)


class _NNLSMeta:
    """Dummy class for joblib unpickling in the __main__ context."""
    pass


def main():
    start = datetime.now()
    log.info("Starting at %s", start.isoformat())

    try:
        result = predict_now()
        elapsed = (datetime.now() - start).total_seconds()
        log.info("Success in %.1fs", elapsed)
        log.info("Generated at: %s", result["generated_at"])
        log.info("Data freshness: %s min", result["data_freshness_minutes"])
        log.info(
            "Current AQI: %s (%s)",
            result["current"]["aqi"],
            result["current"]["category"],
        )
        for h, fc in result["forecasts"].items():
            log.info("  %s: %s AQI (%s)", h, fc["aqi"], fc["category"])

    except Exception as exc:
        elapsed = (datetime.now() - start).total_seconds()
        log.critical("FAILED after %.1fs: %s", elapsed, exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
