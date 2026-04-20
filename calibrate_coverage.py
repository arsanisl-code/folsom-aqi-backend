"""
calibrate_coverage.py — V14 Split Conformal Prediction Calibration.

Math (Split Conformal Prediction):
    Given a calibration set of n samples with true values y_i and predicted
    intervals [lo_i, hi_i], define the nonconformity score for each sample as:

        s_i = max(lo_i - y_i,  y_i - hi_i)   (signed interval residual)
            = 0 if y_i is inside [lo_i, hi_i]
            > 0 if y_i is outside (by how much)

    The conformal correction scalar q is the ceil((1-α)(1 + 1/n)) quantile
    of {s_1, ..., s_n}.  Expanding the interval by q guarantees:

        P(y_{n+1} ∈ [lo - q, hi + q]) ≥ 1 - α

    for any exchangeable test point, with finite-sample validity.

    Reference: Angelopoulos & Bates (2021), "A Gentle Introduction to
    Conformal Prediction and Distribution-Free Uncertainty Quantification."

Usage:
    python calibrate_coverage.py

Artifacts:
    models_v6/conformal_scales.json  — {horizon_h: q_scalar} for each horizon
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

MODELS_DIR   = Path("models_v6")
OUTPUT_PATH  = MODELS_DIR / "conformal_scales.json"
HORIZONS     = [6, 12, 24, 48]
ALPHA        = 0.05   # target miscoverage rate → 95% coverage
# Calibration window: 60–120 days before today (separate from val set)
# Using a dedicated calibration split avoids contaminating the val metrics.
CAL_DAYS_END   = 60
CAL_DAYS_START = 120


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
    Compute the conformal correction scalar q for a calibration set.

    Nonconformity score: s_i = max(lo_i - y_i, y_i - hi_i)
      = 0 when y_i is inside the interval
      > 0 when y_i is outside (distance to nearest boundary)

    q = quantile at level ceil((1-alpha)(1 + 1/n)) / n of sorted scores.
    Expanding [lo - q, hi + q] guarantees >= (1-alpha) coverage.

    Args:
        y_true: Ground-truth absolute AQI values, shape (n,)
        y_lo:   Lower bound predictions, shape (n,)
        y_hi:   Upper bound predictions, shape (n,)
        alpha:  Miscoverage rate (0.05 → 95% coverage)

    Returns:
        q: Non-negative scalar correction (AQI units)
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty calibration set — cannot compute conformal scale.")

    # Nonconformity scores: distance outside interval (0 if inside)
    scores = np.maximum(y_lo - y_true, y_true - y_hi)

    # Conformal quantile level with finite-sample correction
    level = np.ceil((1 - alpha) * (1 + 1 / n)) / n
    level = min(level, 1.0)  # clamp to [0,1]

    q = float(np.quantile(scores, level))
    q = max(q, 0.0)  # non-negative: never shrink intervals
    return q


def run_calibration():
    log.info("=" * 65)
    log.info("  V14 Conformal Calibration  (alpha=%.2f → %.0f%% coverage)", ALPHA, (1-ALPHA)*100)
    log.info("  Calibration window: T-120d to T-60d")
    log.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("Loading historical data...")
    df = fetch_full_history()
    tz = df.index.tz
    now = pd.Timestamp.now(tz=tz)

    cal_end   = now - timedelta(days=CAL_DAYS_END)
    cal_start = now - timedelta(days=CAL_DAYS_START)

    log.info("  Cal window: %s → %s", cal_start.date(), cal_end.date())

    firms_all = _extract_firms(df)
    scales: dict[str, float] = {}
    coverage_report: list[dict] = []

    for h in HORIZONS:
        log.info("─" * 65)
        log.info("  Horizon: %sh", h)

        # ── Load trained models ───────────────────────────────────────────
        point_path = MODELS_DIR / f"lgbm_point_{h}h.pkl"
        q05_path   = MODELS_DIR / f"lgbm_q05_{h}h.pkl"
        q95_path   = MODELS_DIR / f"lgbm_q95_{h}h.pkl"
        imp_path   = MODELS_DIR / f"imputer_{h}h.pkl"

        for p in [point_path, q05_path, q95_path, imp_path]:
            if not p.exists():
                log.error("  Missing: %s — run train.py first", p)
                sys.exit(1)

        point_model = joblib.load(point_path)
        q05_model   = joblib.load(q05_path)
        q95_model   = joblib.load(q95_path)
        imputer     = joblib.load(imp_path)

        # ── Build features for calibration window ─────────────────────────
        # Use full df for rolling warm-up, then slice to cal window.
        # firms_all is causal per-row via trajectories.py cap_ns.
        X_full, y_full = engineer_features(df, h, firms_hourly=firms_all)
        mask = (
            y_full.notna()
            & (X_full.index >= cal_start)
            & (X_full.index <= cal_end)
        )
        X_cal = X_full[mask]
        y_cal = y_full[mask]

        log.info("  Calibration rows: %d", len(X_cal))
        if len(X_cal) < 50:
            log.warning("  Too few calibration rows — using q=0 (no correction)")
            scales[str(h)] = 0.0
            continue

        # ── Impute using the TRAINING imputer (no refit) ──────────────────
        X_cal_imp = imputer.transform(X_cal)
        X_cal_df  = pd.DataFrame(X_cal_imp, columns=X_cal.columns, index=X_cal.index)

        # ── Predict on calibration set ────────────────────────────────────
        base_aqi = X_cal_df["aqi_current"].values

        pred_res = point_model.predict(X_cal_df[point_model.feature_name_])
        pred_q05 = q05_model.predict(X_cal_df[q05_model.feature_name_])
        pred_q95 = q95_model.predict(X_cal_df[q95_model.feature_name_])

        pred_abs = np.clip(pred_res + base_aqi, 0, 500)
        pred_lo  = np.clip(pred_q05 + base_aqi, 0, 500)
        pred_hi  = np.clip(pred_q95 + base_aqi, 0, 500)
        true_abs = y_cal.values + base_aqi

        # Fix quantile crossing
        pred_lo_sorted = np.minimum(pred_lo, pred_hi)
        pred_hi_sorted = np.maximum(pred_lo, pred_hi)

        # ── Raw coverage before calibration ───────────────────────────────
        raw_cov = float(np.mean((true_abs >= pred_lo_sorted) & (true_abs <= pred_hi_sorted)))
        log.info("  Raw coverage: %.1f%%  (target: %.0f%%)", raw_cov * 100, (1-ALPHA)*100)

        # ── Compute conformal scale ───────────────────────────────────────
        q = compute_conformal_scale(true_abs, pred_lo_sorted, pred_hi_sorted, ALPHA)
        log.info("  Conformal q: %.3f AQI", q)

        # ── Verify calibrated coverage ────────────────────────────────────
        cal_cov = float(np.mean(
            (true_abs >= pred_lo_sorted - q) & (true_abs <= pred_hi_sorted + q)
        ))
        log.info("  Calibrated coverage: %.1f%%", cal_cov * 100)

        avg_width_raw = float(np.mean(pred_hi_sorted - pred_lo_sorted))
        avg_width_cal = float(np.mean((pred_hi_sorted + q) - (pred_lo_sorted - q)))
        log.info("  Avg width: %.1f → %.1f AQI (+%.1f)", avg_width_raw, avg_width_cal, 2*q)

        scales[str(h)] = round(q, 4)
        coverage_report.append({
            "horizon_h":      h,
            "n_cal":          int(len(X_cal)),
            "raw_coverage":   round(raw_cov * 100, 1),
            "cal_coverage":   round(cal_cov * 100, 1),
            "q_scalar":       round(q, 4),
            "avg_width_raw":  round(avg_width_raw, 1),
            "avg_width_cal":  round(avg_width_cal, 1),
        })

    # ── Save scales ───────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now().isoformat(),
        "alpha":        ALPHA,
        "target_coverage": (1 - ALPHA) * 100,
        "cal_window_days": [CAL_DAYS_START, CAL_DAYS_END],
        "scales":       scales,
        "report":       coverage_report,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    log.info("=" * 65)
    log.info("  Conformal scales saved → %s", OUTPUT_PATH)
    log.info("  %-6s  %-8s  %-12s  %-12s  %-8s", "Horiz", "q (AQI)", "Raw Cov", "Cal Cov", "Width+")
    log.info("  " + "-" * 55)
    for r in coverage_report:
        log.info("  %-6s  %-8.3f  %-12s  %-12s  +%.1f",
                 f"{r['horizon_h']}h",
                 r["q_scalar"],
                 f"{r['raw_coverage']:.1f}%",
                 f"{r['cal_coverage']:.1f}%",
                 2 * r["q_scalar"])
    log.info("Calibration complete.")


if __name__ == "__main__":
    run_calibration()
