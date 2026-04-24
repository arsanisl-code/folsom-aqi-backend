"""
baselines.py — Module 1: Naive Baseline Evaluation for Academic Validation.

Evaluates two naive forecasting models on the 2025 holdout set:

1. Persistence Baseline
   Prediction: pred[T+h] = AQI[T]  (current value, no model)
   This is the hardest naive baseline to beat at short horizons.
   Skill score = 1 - MAE_model / MAE_persistence measures improvement
   over "do nothing".

2. Climatological Baseline
   Prediction: pred[T+h] = mean AQI for (day_of_year, hour) in training data
   Built from 2019-2024 training data ONLY — no holdout leakage.
   Captures diurnal and seasonal patterns without any dynamic information.

Both baselines use the same train/holdout split as V15:
  Train:   2019-01-01 → 2024-12-31
  Holdout: 2025-01-01 → 2025-12-31

Output: models_v6/naive_baselines.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from data_fetcher import fetch_full_history
from logger import get_logger

log = get_logger(__name__)

MODELS_DIR = Path("models_v6")
OUTPUT_PATH = MODELS_DIR / "naive_baselines.json"
TRAIN_CUTOFF = datetime(2024, 12, 31, 23, 59, 59)
HOLDOUT_YEAR = 2025
HORIZONS = [6, 12, 24, 48]


def run_baselines():
    log.info("=" * 65)
    log.info("  Academic Validation — Naive Baselines")
    log.info("  Train: 2019-01-01 → 2024-12-31")
    log.info("  Holdout: 2025-01-01 → 2025-12-31")
    log.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("Loading historical data...")
    df = fetch_full_history()
    tz = df.index.tz

    train_cutoff_ts = pd.Timestamp(TRAIN_CUTOFF, tz=tz)
    df_train = df[df.index <= train_cutoff_ts].copy()
    df_hold = df[df.index.year == HOLDOUT_YEAR].copy()

    log.info("  Train rows: %s  |  Holdout rows: %s", f"{len(df_train):,}", f"{len(df_hold):,}")

    if len(df_hold) == 0:
        log.error("No 2025 holdout data found.")
        sys.exit(1)

    # ── AQI series ────────────────────────────────────────────────────────
    # Use us_aqi column directly — no feature engineering needed for baselines.
    aqi_train = pd.to_numeric(df_train["us_aqi"], errors="coerce")
    aqi_hold = pd.to_numeric(df_hold["us_aqi"], errors="coerce")

    # ── Climatological lookup table ───────────────────────────────────────
    # Built from training data only. Key: (day_of_year, hour).
    # Handles missing (doy, hour) combinations with global mean fallback.
    log.info("Building climatological lookup table from training data...")
    clim_df = pd.DataFrame(
        {
            "aqi": aqi_train.values,
            "day_of_year": aqi_train.index.day_of_year,
            "hour": aqi_train.index.hour,
        }
    ).dropna()

    clim_table = clim_df.groupby(["day_of_year", "hour"])["aqi"].mean().to_dict()
    global_mean = float(clim_df["aqi"].mean())
    log.info("  Climatology entries: %d  |  Global mean: %.1f AQI", len(clim_table), global_mean)

    results: dict[str, list] = {"persistence": [], "climatology": []}

    for h in HORIZONS:
        log.info("─" * 65)
        log.info("  Horizon: %sh", h)

        # Align holdout: for each row at time T, we need AQI at T and T+h.
        # Use the raw holdout series shifted by h hours.
        aqi_now = aqi_hold.copy()  # AQI at T
        aqi_future = aqi_hold.shift(-h)  # AQI at T+h (target)

        # Drop rows where either is NaN
        valid = aqi_now.notna() & aqi_future.notna()
        y_true = aqi_future[valid].values
        y_now = aqi_now[valid].values
        idx = aqi_hold.index[valid]

        n = len(y_true)
        log.info("  Valid holdout pairs: %d", n)

        # ── Persistence ───────────────────────────────────────────────────
        mae_p = mean_absolute_error(y_true, y_now)
        r2_p = r2_score(y_true, y_now)
        log.info("  Persistence:    MAE=%.2f  R²=%.3f", mae_p, r2_p)

        results["persistence"].append(
            {
                "horizon_h": h,
                "mae": round(mae_p, 2),
                "r2": round(r2_p, 3),
                "n": n,
            }
        )

        # ── Climatology ───────────────────────────────────────────────────
        doy_arr = idx.day_of_year
        hour_arr = idx.hour
        clim_pred = np.array(
            [clim_table.get((int(d), int(hr)), global_mean) for d, hr in zip(doy_arr, hour_arr)]
        )

        mae_c = mean_absolute_error(y_true, clim_pred)
        r2_c = r2_score(y_true, clim_pred)
        log.info("  Climatology:    MAE=%.2f  R²=%.3f", mae_c, r2_c)

        results["climatology"].append(
            {
                "horizon_h": h,
                "mae": round(mae_c, 2),
                "r2": round(r2_c, 3),
                "n": n,
            }
        )

    # ── Sanity checks (Reviewer protocol) ────────────────────────────────
    for h_idx, h in enumerate(HORIZONS):
        p_mae = results["persistence"][h_idx]["mae"]
        c_mae = results["climatology"][h_idx]["mae"]
        log.info("  [CHECK] %sh: persistence MAE=%.2f  climatology MAE=%.2f", h, p_mae, c_mae)

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF),
        "holdout_year": HOLDOUT_YEAR,
        "baselines": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    log.info("=" * 65)
    log.info("  Naive baselines saved → %s", OUTPUT_PATH)
    log.info("Baselines complete.")


if __name__ == "__main__":
    run_baselines()
