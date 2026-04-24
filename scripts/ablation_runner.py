"""
ablation_runner.py — Module 2: Wildfire Feature Ablation Study.

Purpose:
    Prove that the Lagrangian FIRMS trajectory and fire detection features
    add genuine predictive value by training an identical  architecture
    with all fire-detection features removed (ablation_mode=True).

Ablation scope (fire-detection features dropped):
    fire_frp_24h_sum, fire_min_dist_24h, fire_count_24h_sum,
    fire_intensity_proximity_index, fire_advection_24h_max,
    fwd_fire_advection_24h_max, fire_frp_7d_max, fire_active_days_30d,
    traj_fire_frp_sum_*, traj_fire_count_*, traj_fire_min_dist_*,
    smoke_wind_alignment

Intentionally KEPT (wind-derived, not fire-derived):
    traj_origin_lat_*, traj_origin_lon_*

This isolates the causal question:
    "Does knowing WHERE fires are and HOW MUCH they burn improve AQI forecasts,
     beyond what wind transport direction alone can tell us?"

Methodology:
    - Identical train/holdout split as : train ≤ 2024-12-31, holdout = 2025
    - Identical loss function: regression_l1 ( standard)
    - Identical imputer discipline: fit on train only, transform holdout
    - Identical hyperparameters: loaded from best_optuna_params.json
    - No re-tuning (ablation uses same HPO params as  full model)

Output: models/ablation_metrics.json
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
from features import engineer_features
from logger import get_logger
from train import _point_params

log = get_logger(__name__)

MODELS_DIR = Path("models")
OUTPUT_PATH = MODELS_DIR / "ablation_metrics.json"
TRAIN_CUTOFF = datetime(2024, 12, 31, 23, 59, 59)
HOLDOUT_YEAR = 2025
HORIZONS = [6, 12, 24, 48]


def _extract_firms(df: pd.DataFrame) -> pd.DataFrame:
    """Extract FIRMS fire columns from merged df (causal per-row via trajectories.py)."""
    firms_cols = ["fire_frp_raw", "fire_count_raw", "fire_min_dist_raw", "fire_bearing_nearest"]
    avail = [c for c in firms_cols if c in df.columns]
    if not avail or "fire_frp_raw" not in df.columns:
        return pd.DataFrame()
    firms = df[avail].copy()
    return firms[firms["fire_frp_raw"] > 0]


def run_ablation():
    log.info("=" * 65)
    log.info("  Academic Validation — Wildfire Feature Ablation Study")
    log.info("  ablation_mode=True: all FIRMS fire-detection features dropped")
    log.info("  Trajectory origin lat/lon (wind-derived) KEPT")
    log.info("  Train: 2019-01-01 → 2024-12-31  |  Holdout: 2025")
    log.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("Loading historical data...")
    df = fetch_full_history()
    tz = df.index.tz
    train_cutoff_ts = pd.Timestamp(TRAIN_CUTOFF, tz=tz)

    df_hold_check = df[df.index.year == HOLDOUT_YEAR]
    if len(df_hold_check) == 0:
        log.error("No 2025 holdout data found.")
        sys.exit(1)

    log.info(
        "  Train rows: %s  |  Holdout rows: %s",
        f"{len(df[df.index <= train_cutoff_ts]):,}",
        f"{len(df_hold_check):,}",
    )

    # FIRMS for trajectory features (causal per-row)
    firms_train_only = _extract_firms(df[df.index <= train_cutoff_ts])
    firms_all = _extract_firms(df)
    log.info("  FIRMS fire-hours (train): %d  |  (full): %d", len(firms_train_only), len(firms_all))

    results = []

    for h in HORIZONS:
        log.info("─" * 65)
        log.info("  Horizon: %sh  [ABLATED — no fire detection features]", h)

        # ── Feature engineering with ablation_mode=True ───────────────────
        # Training features: firms_train_only + ablation_mode=True
        log.info("  Building ablated training features...")
        X_tr_full, y_tr_full = engineer_features(
            df,
            h,
            firms_hourly=firms_train_only,
            ablation_mode=True,
        )
        mask_tr = y_tr_full.notna() & (X_tr_full.index <= train_cutoff_ts)
        X_tr = X_tr_full[mask_tr]
        y_tr = y_tr_full[mask_tr]

        # Holdout features: firms_all + ablation_mode=True
        log.info("  Building ablated holdout features...")
        X_ho_full, y_ho_full = engineer_features(
            df,
            h,
            firms_hourly=firms_all,
            ablation_mode=True,
        )
        mask_ho = y_ho_full.notna() & (X_ho_full.index.year == HOLDOUT_YEAR)
        X_ho = X_ho_full[mask_ho]
        y_ho = y_ho_full[mask_ho]

        log.info(
            "  Train: %s rows  |  Holdout: %s rows  |  Features: %d",
            f"{len(X_tr):,}",
            f"{len(X_ho):,}",
            X_tr.shape[1],
        )

        if len(X_tr) < 500 or len(X_ho) == 0:
            log.warning("  Skipping %sh — insufficient data", h)
            continue

        # ── Impute — fit on train only ────────────────────────────────────
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_ho_imp = imputer.transform(X_ho)

        X_tr_df = pd.DataFrame(X_tr_imp, columns=X_tr.columns, index=X_tr.index)
        X_ho_df = pd.DataFrame(X_ho_imp, columns=X_ho.columns, index=X_ho.index)

        # ── Train point model — same params as  (regression_l1) ───────
        # ES split: last 30 days of training window
        es_cutoff = train_cutoff_ts - pd.Timedelta(days=30)
        es_mask = X_tr_df.index >= es_cutoff
        X_fit, y_fit = X_tr_df[~es_mask], y_tr[~es_mask]
        X_es, y_es = X_tr_df[es_mask], y_tr[es_mask]

        log.info(
            "  Training ablated point model (%s fit, %s ES)...", f"{len(X_fit):,}", f"{len(X_es):,}"
        )

        point_model = lgb.LGBMRegressor(**_point_params(h))
        point_model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        # ── Evaluate on holdout ───────────────────────────────────────────
        base_aqi = X_ho_df["aqi_current"].values
        pred_res = point_model.predict(X_ho_df)
        pred_abs = np.clip(pred_res + base_aqi, 0, 500)
        true_abs = y_ho.values + base_aqi

        mae = mean_absolute_error(true_abs, pred_abs)
        r2 = r2_score(true_abs, pred_abs)

        log.info("  Ablated %sh:  MAE=%.2f  R²=%.3f  (n=%d)", h, mae, r2, len(true_abs))

        results.append(
            {
                "horizon_h": h,
                "mae": round(mae, 2),
                "r2": round(r2, 3),
                "n": int(len(true_abs)),
                "n_features": int(X_tr.shape[1]),
            }
        )

    # ── Reviewer sanity checks ────────────────────────────────────────────
    log.info("=" * 65)
    log.info("  ABLATION SANITY CHECKS")
    log.info("  (Ablated MAE should be >=  full MAE at 24h/48h)")
    _mae = {6: 2.81, 12: 5.37, 24: 7.20, 48: 8.50}
    for r in results:
        h = r["horizon_h"]
        diff = r["mae"] - _mae.get(h, 0)
        flag = "✓ fire features help" if diff > 0 else "⚠ unexpected — check"
        log.info(
            "  %sh: ablated=%.2f  =%.2f  Δ=%+.2f  %s", h, r["mae"], _mae.get(h, 0), diff, flag
        )

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF),
        "holdout_year": HOLDOUT_YEAR,
        "ablation_scope": (
            "Dropped: fire_frp_*, fire_min_dist_*, fire_count_*, "
            "fire_intensity_proximity_index, fire_advection_*, "
            "fwd_fire_advection_*, fire_frp_7d_max, fire_active_days_30d, "
            "traj_fire_frp_sum_*, traj_fire_count_*, traj_fire_min_dist_*, "
            "smoke_wind_alignment. "
            "KEPT: traj_origin_lat/lon_* (wind-derived)."
        ),
        "horizons": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    log.info("  Ablation metrics saved → %s", OUTPUT_PATH)
    log.info("Ablation study complete.")


if __name__ == "__main__":
    run_ablation()
