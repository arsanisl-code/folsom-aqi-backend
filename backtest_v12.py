"""
backtest.py — 2025 Holdout Backtest.

Methodology:
  - Train on data from 2019-01-01 through 2024-12-31
  - Evaluate on 2025-01-01 through 2025-12-31 (month-by-month)
  - Reports MAE, R², Coverage per horizon per month + annual averages

Leakage prevention:
  1. Imputer fit ONLY on training rows — never sees holdout statistics.
  2. Feature engineering runs on the full df for rolling-window warm-up,
     but training/holdout rows are split by index after construction.
  3. fwd_* features use shift(-horizon_h) which is legitimate NWP proxy.

Usage:
    python backtest_v12.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from data_fetcher import fetch_full_history
from features_v6 import engineer_features
from train import _point_params, _quantile_params
from logger import get_logger

log = get_logger(__name__)

TRAIN_CUTOFF = datetime(2024, 12, 31, 23, 59, 59)
HOLDOUT_YEAR = 2025
HORIZONS     = [6, 12, 24, 48]
OUTPUT_CSV   = Path("backtest_2025_report.csv")
OUTPUT_JSON  = Path("backtest_2025_report.json")


def run_backtest():
    log.info("=" * 65)
    log.info("  2025 Holdout Backtest")
    log.info("  Train: 2019-01-01 → 2024-12-31")
    log.info("  Holdout: 2025-01-01 → 2025-12-31")
    log.info("=" * 65)

    log.info("Loading historical data...")
    df = fetch_full_history()
    log.info("  Rows: %s  |  Range: %s → %s",
             f"{len(df):,}", df.index.min().date(), df.index.max().date())

    tz = df.index.tz
    train_cutoff_ts = pd.Timestamp(TRAIN_CUTOFF, tz=tz)

    df_hold_check = df[df.index.year == HOLDOUT_YEAR]
    if len(df_hold_check) == 0:
        log.error("No 2025 holdout data found. Check data range.")
        sys.exit(1)

    log.info("  Train rows: %s  |  Holdout rows: %s",
             f"{len(df[df.index <= train_cutoff_ts]):,}",
             f"{len(df_hold_check):,}")

    results = []

    for h in HORIZONS:
        log.info("─" * 65)
        log.info("  Horizon: %sh", h)

        log.info("  Building training features...")
        X_tr_full, y_tr_full = engineer_features(df, h)
        mask_tr = y_tr_full.notna() & (X_tr_full.index <= train_cutoff_ts)
        X_tr = X_tr_full[mask_tr]
        y_tr = y_tr_full[mask_tr]

        log.info("  Building holdout features...")
        X_ho_full, y_ho_full = engineer_features(df, h)
        mask_ho = y_ho_full.notna() & (X_ho_full.index.year == HOLDOUT_YEAR)
        X_ho = X_ho_full[mask_ho]
        y_ho = y_ho_full[mask_ho]

        log.info("  Train: %s rows  |  Holdout: %s rows",
                 f"{len(X_tr):,}", f"{len(X_ho):,}")

        if len(X_tr) < 500 or len(X_ho) == 0:
            log.warning("  Skipping %sh — insufficient data", h)
            continue

        # ── Impute — fit ONLY on training rows ────────────────────────────
        # Critical: imputer statistics must not see any holdout data.
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_ho_imp = imputer.transform(X_ho)   # transform only — no fit

        X_tr_df = pd.DataFrame(X_tr_imp, columns=X_tr.columns, index=X_tr.index)
        X_ho_df = pd.DataFrame(X_ho_imp, columns=X_ho.columns, index=X_ho.index)

        # ── Train point model ─────────────────────────────────────────────
        # ES split: last 30 days of training window (within train only)
        es_cutoff = train_cutoff_ts - pd.Timedelta(days=30)
        es_mask   = X_tr_df.index >= es_cutoff
        X_fit, y_fit = X_tr_df[~es_mask], y_tr[~es_mask]
        X_es,  y_es  = X_tr_df[es_mask],  y_tr[es_mask]

        log.info("  Training point model (%s fit, %s ES)...",
                 f"{len(X_fit):,}", f"{len(X_es):,}")

        point_model = lgb.LGBMRegressor(**_point_params(h))
        point_model.fit(
            X_fit, y_fit,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        # ── Train quantile models ─────────────────────────────────────────
        log.info("  Training quantile models...")
        q05_model = lgb.LGBMRegressor(**_quantile_params(0.005, h))
        q05_model.fit(X_tr_df, y_tr, callbacks=[lgb.log_evaluation(-1)])

        q95_model = lgb.LGBMRegressor(**_quantile_params(0.995, h))
        q95_model.fit(X_tr_df, y_tr, callbacks=[lgb.log_evaluation(-1)])

        # ── Evaluate on holdout — month by month ─────────────────────────
        base_aqi_ho = X_ho_df["aqi_current"].values

        pred_res = point_model.predict(X_ho_df)
        pred_q05 = q05_model.predict(X_ho_df)
        pred_q95 = q95_model.predict(X_ho_df)

        # Invert residual prediction back to absolute AQI
        pred_abs = np.clip(pred_res + base_aqi_ho, 0, 500)
        pred_lo  = np.clip(pred_q05 + base_aqi_ho, 0, 500)
        pred_hi  = np.clip(pred_q95 + base_aqi_ho, 0, 500)
        true_abs = y_ho.values + base_aqi_ho

        months = X_ho_df.index.month

        for m in range(1, 13):
            mask_m = months == m
            if mask_m.sum() == 0:
                continue

            t_m  = true_abs[mask_m]
            p_m  = pred_abs[mask_m]
            lo_m = pred_lo[mask_m]
            hi_m = pred_hi[mask_m]

            mae      = mean_absolute_error(t_m, p_m)
            r2       = r2_score(t_m, p_m) if len(t_m) > 1 else float("nan")
            coverage = float(np.mean((t_m >= lo_m) & (t_m <= hi_m))) * 100

            month_name = datetime(2025, m, 1).strftime("%b")
            results.append({
                "horizon_h":  h,
                "month":      m,
                "month_name": month_name,
                "season":     "summer" if m in {5,6,7,8,9} else "winter",
                "mae":        round(mae, 2),
                "r2":         round(r2, 3),
                "coverage":   round(coverage, 1),
                "n":          int(mask_m.sum()),
            })
            log.info("    %s %2sh  MAE=%.2f  R²=%.3f  Cov=%.1f%%  (n=%d)",
                     month_name, h, mae, r2, coverage, mask_m.sum())

    # ── Annual averages ───────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("  ANNUAL AVERAGES (2025 Full Year)")
    log.info("=" * 65)

    df_res = pd.DataFrame(results)

    # Load V6 baseline for comparison
    baseline_path = Path("backtest_v6_2025_report.csv")
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)
        df_base.columns = [c.lower() for c in df_base.columns]
        has_baseline = True
    else:
        has_baseline = False

    log.info("  %-6s  %-8s  %-8s  %-8s  %-10s  %-10s",
             "Horiz", "MAE", "MAE_prev", "Δ MAE", "R²", "Coverage")
    log.info("  " + "-" * 60)

    annual = []
    for h in HORIZONS:
        h_rows = df_res[df_res["horizon_h"] == h]
        if h_rows.empty:
            continue

        weights = h_rows["n"].values
        mae_v12 = float(np.average(h_rows["mae"].values, weights=weights))
        r2_v12  = float(np.average(h_rows["r2"].values,  weights=weights))
        cov_v12 = float(np.average(h_rows["coverage"].values, weights=weights))

        mae_v6 = float("nan")
        if has_baseline:
            b_rows = df_base[df_base["horizon"] == f"{h}h"]
            if not b_rows.empty:
                mae_v6 = b_rows["mae"].mean()

        delta     = mae_v12 - mae_v6 if not np.isnan(mae_v6) else float("nan")
        delta_str = f"{delta:+.2f}" if not np.isnan(delta) else "N/A"
        v6_str    = f"{mae_v6:.2f}" if not np.isnan(mae_v6) else "N/A"

        log.info("  %-6s  %-8.2f  %-8s  %-8s  %-10.3f  %.1f%%",
                 f"{h}h", mae_v12, v6_str, delta_str, r2_v12, cov_v12)

        # Season-stratified breakdown
        for season in ["summer", "winter"]:
            s_rows = h_rows[h_rows["season"] == season] if "season" in h_rows.columns else pd.DataFrame()
            if not s_rows.empty:
                s_w = s_rows["n"].values
                s_mae = float(np.average(s_rows["mae"].values, weights=s_w))
                s_cov = float(np.average(s_rows["coverage"].values, weights=s_w))
                log.info("         %-8s MAE=%.2f  Cov=%.1f%%", season, s_mae, s_cov)

        annual.append({
            "horizon_h": h,
            "mae_v12":   round(mae_v12, 2),
            "mae_v6":    round(mae_v6, 2) if not np.isnan(mae_v6) else None,
            "delta_mae": round(delta, 2) if not np.isnan(delta) else None,
            "r2_v12":    round(r2_v12, 3),
            "coverage":  round(cov_v12, 1),
        })

    # ── Save outputs ──────────────────────────────────────────────────────
    df_res.to_csv(OUTPUT_CSV, index=False)
    log.info("  Monthly results → %s", OUTPUT_CSV)

    report = {
        "generated_at": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF),
        "holdout_year": HOLDOUT_YEAR,
        "annual":       annual,
        "monthly":      results,
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2))
    log.info("  Full report → %s", OUTPUT_JSON)
    log.info("Backtest complete.")


if __name__ == "__main__":
    run_backtest()
