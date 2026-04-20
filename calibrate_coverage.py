"""
calibrate_coverage.py — V15 Season-Aware Split Conformal Prediction Calibration.

Problem with V14 calibration:
    The calibration window (T-120d to T-60d) fell in winter, where raw coverage
    was already 99%+, giving q=0. But summer (May-Sep) coverage on the 2025
    holdout was only 89-93%. The conformal guarantee requires the calibration
    distribution to match the test distribution — a winter-only window violates
    this for summer predictions.

Fix (V15):
    Use a full-year calibration window (T-365d to T-60d) and compute separate
    q scalars for:
        summer: May-Sep (months 5-9)
        winter: Oct-Apr (months 10-4)
    At inference time, apply the seasonal q based on the current month.

Math (Split Conformal Prediction):
    Nonconformity score: s_i = max(lo_i - y_i, y_i - hi_i)
    q = quantile at level ceil((1-alpha)(1 + 1/n)) / n of sorted scores.
    Expanding [lo - q, hi + q] guarantees >= (1-alpha) coverage.

    Reference: Angelopoulos & Bates (2021).

Artifacts:
    models_v6/conformal_scales.json — {
        "6":  {"summer": q_s, "winter": q_w},
        "12": {"summer": q_s, "winter": q_w},
        ...
    }
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from data_fetcher import fetch_full_history
from features_v6 import engineer_features
from logger import get_logger

log = get_logger(__name__)

MODELS_DIR      = Path("models_v6")
OUTPUT_PATH     = MODELS_DIR / "conformal_scales.json"
HORIZONS        = [6, 12, 24, 48]
ALPHA           = 0.05   # 95% coverage target
CAL_DAYS_END    = 60     # exclude last 60 days (val set)
CAL_DAYS_START  = 365    # use full year for calibration

SUMMER_MONTHS = {5, 6, 7, 8, 9}   # May-Sep
WINTER_MONTHS = {10, 11, 12, 1, 2, 3, 4}  # Oct-Apr


def _extract_firms(df: pd.DataFrame) -> pd.DataFrame:
    firms_cols = ["fire_frp_raw", "fire_count_raw", "fire_min_dist_raw", "fire_bearing_nearest"]
    avail = [c for c in firms_cols if c in df.columns]
    if not avail or "fire_frp_raw" not in df.columns:
        return pd.DataFrame()
    firms = df[avail].copy()
    return firms[firms["fire_frp_raw"] > 0]


def compute_conformal_scale(
    y_true: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    alpha: float = ALPHA,
) -> float:
    """
    Compute conformal correction scalar q for a calibration set.
    Returns 0.0 if already over-covered (no expansion needed).
    """
    n = len(y_true)
    if n < 10:
        log.warning("  Too few calibration samples (%d) — returning q=0", n)
        return 0.0

    scores = np.maximum(y_lo - y_true, y_true - y_hi)
    level  = min(np.ceil((1 - alpha) * (1 + 1 / n)) / n, 1.0)
    q      = float(np.quantile(scores, level))
    return max(q, 0.0)


def run_calibration():
    log.info("=" * 65)
    log.info("  V15 Season-Aware Conformal Calibration")
    log.info("  alpha=%.2f → %.0f%% coverage target", ALPHA, (1 - ALPHA) * 100)
    log.info("  Cal window: T-%dd to T-%dd (full year)", CAL_DAYS_START, CAL_DAYS_END)
    log.info("=" * 65)

    log.info("Loading historical data...")
    df = fetch_full_history()
    tz  = df.index.tz
    now = pd.Timestamp.now(tz=tz)

    cal_end   = now - timedelta(days=CAL_DAYS_END)
    cal_start = now - timedelta(days=CAL_DAYS_START)
    log.info("  Cal window: %s → %s", cal_start.date(), cal_end.date())

    firms_all = _extract_firms(df)
    scales_out: dict = {}
    report: list[dict] = []

    for h in HORIZONS:
        log.info("─" * 65)
        log.info("  Horizon: %sh", h)

        # Load models
        for p in [MODELS_DIR / f"lgbm_{k}_{h}h.pkl" for k in ["point", "q05", "q95"]] + \
                 [MODELS_DIR / f"imputer_{h}h.pkl"]:
            if not p.exists():
                log.error("  Missing: %s — run train.py first", p)
                sys.exit(1)

        point_model = joblib.load(MODELS_DIR / f"lgbm_point_{h}h.pkl")
        q05_model   = joblib.load(MODELS_DIR / f"lgbm_q05_{h}h.pkl")
        q95_model   = joblib.load(MODELS_DIR / f"lgbm_q95_{h}h.pkl")
        imputer     = joblib.load(MODELS_DIR / f"imputer_{h}h.pkl")

        # Build features for full calibration window
        X_full, y_full = engineer_features(df, h, firms_hourly=firms_all)
        mask = (
            y_full.notna()
            & (X_full.index >= cal_start)
            & (X_full.index <= cal_end)
        )
        X_cal = X_full[mask]
        y_cal = y_full[mask]
        log.info("  Total cal rows: %d", len(X_cal))

        if len(X_cal) < 50:
            log.warning("  Insufficient data — q=0 for all seasons")
            scales_out[str(h)] = {"summer": 0.0, "winter": 0.0}
            continue

        # Impute using training imputer (no refit — no leakage)
        X_cal_imp = imputer.transform(X_cal)
        X_cal_df  = pd.DataFrame(X_cal_imp, columns=X_cal.columns, index=X_cal.index)

        # Predict
        base_aqi = X_cal_df["aqi_current"].values
        pred_res = point_model.predict(X_cal_df[point_model.feature_name_])
        pred_q05 = q05_model.predict(X_cal_df[q05_model.feature_name_])
        pred_q95 = q95_model.predict(X_cal_df[q95_model.feature_name_])

        pred_abs = np.clip(pred_res + base_aqi, 0, 500)
        pred_lo  = np.clip(np.minimum(pred_q05, pred_q95) + base_aqi, 0, 500)
        pred_hi  = np.clip(np.maximum(pred_q05, pred_q95) + base_aqi, 0, 500)
        true_abs = y_cal.values + base_aqi

        months = X_cal_df.index.month

        h_scales: dict[str, float] = {}
        for season, month_set in [("summer", SUMMER_MONTHS), ("winter", WINTER_MONTHS)]:
            mask_s = np.array([m in month_set for m in months])
            n_s    = mask_s.sum()

            if n_s < 10:
                log.warning("  %s: only %d rows — q=0", season, n_s)
                h_scales[season] = 0.0
                continue

            raw_cov = float(np.mean(
                (true_abs[mask_s] >= pred_lo[mask_s]) &
                (true_abs[mask_s] <= pred_hi[mask_s])
            ))
            q = compute_conformal_scale(
                true_abs[mask_s], pred_lo[mask_s], pred_hi[mask_s], ALPHA
            )
            cal_cov = float(np.mean(
                (true_abs[mask_s] >= pred_lo[mask_s] - q) &
                (true_abs[mask_s] <= pred_hi[mask_s] + q)
            ))
            avg_w_raw = float(np.mean(pred_hi[mask_s] - pred_lo[mask_s]))

            log.info("  %s (n=%d): raw_cov=%.1f%%  q=%.3f  cal_cov=%.1f%%  width %.1f→%.1f",
                     season, n_s, raw_cov*100, q, cal_cov*100, avg_w_raw, avg_w_raw + 2*q)

            h_scales[season] = round(q, 4)
            report.append({
                "horizon_h": h, "season": season, "n": int(n_s),
                "raw_coverage": round(raw_cov*100, 1),
                "cal_coverage": round(cal_cov*100, 1),
                "q_scalar": round(q, 4),
                "avg_width_raw": round(avg_w_raw, 1),
                "avg_width_cal": round(avg_w_raw + 2*q, 1),
            })

        scales_out[str(h)] = h_scales

    output = {
        "generated_at":    datetime.now().isoformat(),
        "alpha":           ALPHA,
        "target_coverage": (1 - ALPHA) * 100,
        "cal_window_days": [CAL_DAYS_START, CAL_DAYS_END],
        "summer_months":   sorted(SUMMER_MONTHS),
        "winter_months":   sorted(WINTER_MONTHS),
        "scales":          scales_out,
        "report":          report,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    log.info("=" * 65)
    log.info("  Conformal scales saved → %s", OUTPUT_PATH)
    log.info("Calibration complete.")


if __name__ == "__main__":
    run_calibration()
