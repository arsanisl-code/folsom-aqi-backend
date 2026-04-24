"""
train_ensemble.py — V11 Feature-Diversity Stacked Ensemble for Folsom AQI Navigator.

Architecture (V11 changes from V10):
    Base models (3, not 4):
      1. LGBM-Full    — MAE loss, all 173 features, Optuna-tuned
      2. XGBoost      — reg:absoluteerror, all 173 features, Optuna-tuned
      3. LGBM-Physics — MAE loss, ~35 physics-only features (ZERO AQI lag/rolling),
                        provides genuine feature diversity, not just algorithm diversity

    Meta-learner:
      NNLS (scipy) with bias column → normalized convex combination + intercept

    Key V11 fixes vs V10:
      1. CatBoost + Ridge removed → replaced by LGBM-Physics (feature diversity)
      2. LGBM loss: huber → regression_l1 (aligns training with MAE eval metric)
      3. NNLS bias: np.c_[OOF, ones] gives meta-learner a free intercept
      4. HPO eval set: fold 5 of TimeSeriesSplit (aligned with OOF distribution)
      5. ES window: 30 → 60 days (more stable early stopping)
      6. Physics HPO: 75 trials, num_leaves 8–64, max_depth 3–8

Persistence & Safety:
    - Optuna studies in SQLite (optuna_ensemble.db), load_if_exists=True
    - Horizon checkpoint: meta_learner_{h}h.pkl as sentinel
    - Incremental report: tournament_report.json updated after every horizon

Usage:
    python train_ensemble.py                  # all 4 horizons
    python train_ensemble.py --horizon 6      # single horizon chunk
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from data_fetcher import fetch_full_history
from features_v6 import engineer_features, get_feature_names
from logger import get_logger

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = get_logger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models_v10_ensemble")
MODELS_V9_DIR = Path("models_v6")
DATA_DIR = Path("data")
OPTUNA_DB = "sqlite:///optuna_ensemble.db"
REPORT_PATH = MODELS_DIR / "tournament_report.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [6, 12, 24, 48]

# ─── Thread config — Ryzen 7 9800X3D (8C/16T Zen 5) ─────────────────────────
N_JOBS = 14

# ─── Physics feature list — ZERO AQI lag/rolling features ────────────────────
# LGBM-Physics trains exclusively on these atmospheric state features.
# Every feature here is available at prediction time T with no knowledge
# of past or future AQI values. This forces the model to learn from
# atmospheric physics alone, providing genuine feature diversity vs LGBM-Full.
PHYSICS_FEATURES = [
    # Boundary layer / atmospheric stability
    "boundary_layer_height",
    "fwd_blh",
    "blh_collapse_rate",
    "fwd_blh_collapse_rate",
    # Wind — current
    "wind_speed_10m",
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_u",
    "wind_v",
    # Wind — forward NWP
    "fwd_wind_speed",
    "fwd_wind_speed_mean",
    "fwd_wind_dir_sin",
    "fwd_wind_dir_cos",
    "fwd_wind_u",
    "fwd_wind_v",
    # Temperature / humidity / pressure — current
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    # Temperature / humidity / pressure — forward NWP
    "fwd_temperature",
    "fwd_humidity",
    "fwd_pressure",
    "fwd_temperature_mean",
    "fwd_humidity_mean",
    "fwd_pressure_mean",
    # Ventilation
    "fwd_ventilation",
    "fwd_vent_deficit",
    "vent_deficit",
    "vent_deficit_24h_mean",
    # Inversion
    "inversion_strength",
    "inversion_12h_max",
    # Stagnation
    "stagnation_6h",
    "stagnation_24h",
    "stagnation_48h",
    # Fire danger
    "fwd_hdwi",
    "wildfire_hdwi",
    # Precipitation
    "precipitation",
    "fwd_precipitation",
    "fwd_precip_accum",
    # Temporal cyclic — no AQI content
    "hour_sin",
    "hour_cos",
    "future_hour_sin",
    "future_hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]


def _get_physics_cols(X: pd.DataFrame) -> list[str]:
    """Return physics feature columns that actually exist in X."""
    return [c for c in PHYSICS_FEATURES if c in X.columns]


# ─── V9 warm-start / baseline loaders ────────────────────────────────────────


def _load_v9_warmstart(horizon_h: int) -> dict:
    """Load V9 Optuna best params for LGBM-Full warm-start."""
    for fname in ("best_optuna_params_v9.json", "best_optuna_params.json"):
        path = MODELS_V9_DIR / fname
        if path.exists():
            try:
                data = json.loads(path.read_text())
                params = data.get(f"{horizon_h}h", {}).get("point", {}).get("best_params", {})
                if params:
                    log.info("  V9 warm-start: %s  (%d keys)", fname, len(params))
                    return params
            except Exception as exc:
                log.warning("  Could not parse %s: %s", fname, exc)
    return {}


def _load_v9_baseline(horizon_h: int) -> dict:
    path = MODELS_V9_DIR / "training_metrics_v6.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        for h in data.get("horizons", []):
            if h["horizon_h"] == horizon_h:
                return {"mae": h.get("val_mae"), "r2": h.get("val_r2")}
    except Exception:
        pass
    return {}


# ─── Incremental report ───────────────────────────────────────────────────────


def _read_report() -> dict:
    if REPORT_PATH.exists():
        try:
            return json.loads(REPORT_PATH.read_text())
        except Exception:
            pass
    return {
        "generated_at": datetime.now().isoformat(),
        "architecture": "V11 Feature-Diversity Ensemble (LGBM-Full + XGB + LGBM-Physics + NNLS-Bias)",
        "meta_learner": "NNLS(scipy) with bias column, weights normalized to sum=1",
        "validation": "TimeSeriesSplit(n_splits=5, gap=horizon_h)",
        "hpo_budget": {"lgbm_full": 50, "xgb": 75, "lgbm_physics": 75},
        "n_jobs": N_JOBS,
        "optuna_db": OPTUNA_DB,
        "base_models": [
            "LGBM-Full (regression_l1, all 173 features)",
            "XGBoost (reg:absoluteerror, all 173 features)",
            "LGBM-Physics (regression_l1, ~40 physics-only features, zero AQI lags)",
        ],
        "physics_features": PHYSICS_FEATURES,
        "total_run_s": None,
        "horizons": [],
    }


def _write_report(report: dict) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log.info("  Report updated → %s", REPORT_PATH)


def _upsert_horizon_result(result: dict) -> None:
    report = _read_report()
    h = result["horizon_h"]
    existing = [r for r in report["horizons"] if r.get("horizon_h") != h]
    existing.append(result)
    existing.sort(key=lambda r: r.get("horizon_h", 0))
    report["horizons"] = existing
    report["updated_at"] = datetime.now().isoformat()
    _write_report(report)


def _horizon_is_complete(horizon_h: int) -> bool:
    return (MODELS_DIR / f"meta_learner_{horizon_h}h.pkl").exists()


# ─── Trial timer ──────────────────────────────────────────────────────────────


class _TrialTimer:
    def __init__(self, label: str, n_trials: int):
        self.label = label
        self.n_trials = n_trials
        self._trial_starts: dict[int, float] = {}
        self._times: list[float] = []
        self._study_start = time.perf_counter()

    def record_start(self, trial_number: int) -> None:
        self._trial_starts[trial_number] = time.perf_counter()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.duration is not None:
            elapsed = trial.duration.total_seconds()
        elif trial.number in self._trial_starts:
            elapsed = time.perf_counter() - self._trial_starts[trial.number]
        else:
            elapsed = 0.0
        self._times.append(elapsed)
        log.info(
            "    [%s] trial %3d/%-3d  MAE=%.4f  time=%5.1fs  mean=%5.1fs  best=%.4f",
            self.label,
            trial.number + 1,
            self.n_trials,
            trial.value if trial.value is not None else float("nan"),
            elapsed,
            float(np.mean(self._times)) if self._times else 0.0,
            study.best_value,
        )

    def summary(self) -> dict:
        if not self._times:
            return {}
        total = time.perf_counter() - self._study_start
        arr = np.array(self._times)
        return {
            "label": self.label,
            "n_trials": len(arr),
            "total_s": round(float(total), 2),
            "mean_s": round(float(arr.mean()), 2),
            "median_s": round(float(np.median(arr)), 2),
            "min_s": round(float(arr.min()), 2),
            "max_s": round(float(arr.max()), 2),
            "p95_s": round(float(np.percentile(arr, 95)), 2),
            "trials_per_min": round(float(len(arr) / max(total / 60, 1e-6)), 2),
        }


# ─── Optuna objective factories ───────────────────────────────────────────────


def _lgbm_full_objective(trial, X_tr, y_tr, X_va, y_va, warmstart: dict, timer: _TrialTimer):
    """
    LGBM-Full objective — regression_l1 (MAE) loss, all features.
    V11 change: huber → regression_l1 to align training loss with eval metric.
    """
    timer.record_start(trial.number)
    ws = warmstart
    params = {
        "objective": "regression_l1",
        "n_estimators": 2000,
        "n_jobs": N_JOBS,
        "verbosity": -1,
        "random_state": 42,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int(
            "num_leaves",
            max(16, int(ws.get("num_leaves", 31) * 0.5)),
            min(255, int(ws.get("num_leaves", 63) * 2)),
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            max(3, int(ws.get("max_depth", 6) - 2)),
            min(12, int(ws.get("max_depth", 8) + 2)),
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            max(10, int(ws.get("min_child_samples", 20) - 10)),
            min(100, int(ws.get("min_child_samples", 40) + 20)),
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction",
            max(0.4, float(ws.get("colsample_bytree", 0.7)) - 0.2),
            min(1.0, float(ws.get("colsample_bytree", 0.8)) + 0.1),
        ),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction",
            max(0.4, float(ws.get("subsample", 0.7)) - 0.2),
            min(1.0, float(ws.get("subsample", 0.8)) + 0.1),
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 2.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return mean_absolute_error(y_va, model.predict(X_va))


def _xgb_objective(trial, X_tr, y_tr, X_va, y_va, timer: _TrialTimer):
    """XGBoost objective — reg:absoluteerror loss, all features."""
    timer.record_start(trial.number)
    params = {
        "objective": "reg:absoluteerror",
        "n_estimators": 2000,
        "n_jobs": N_JOBS,
        "verbosity": 0,
        "random_state": 42,
        "tree_method": "hist",
        "eta": trial.suggest_float("eta", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
    }
    model = XGBRegressor(**params, early_stopping_rounds=30, eval_metric="mae")
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return mean_absolute_error(y_va, model.predict(X_va))


def _lgbm_physics_objective(trial, X_tr_phys, y_tr, X_va_phys, y_va, timer: _TrialTimer):
    """
    LGBM-Physics objective — regression_l1 loss, physics features only.
    Narrower search space: num_leaves 8–64, max_depth 3–8 (small feature set).
    Different random_state (99) to ensure independent exploration from LGBM-Full.
    """
    timer.record_start(trial.number)
    params = {
        "objective": "regression_l1",
        "n_estimators": 2000,
        "n_jobs": N_JOBS,
        "verbosity": -1,
        "random_state": 99,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 2.0),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr_phys,
        y_tr,
        eval_set=[(X_va_phys, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return mean_absolute_error(y_va, model.predict(X_va_phys))


# ─── HPO runners ─────────────────────────────────────────────────────────────


def run_lgbm_full_hpo(
    X_tr, y_tr, X_va, y_va, horizon_h: int, n_trials: int = 50
) -> tuple[dict, dict]:
    warmstart = _load_v9_warmstart(horizon_h)
    study_name = f"lgbm_full_{horizon_h}h"
    log.info(
        "  [LGBM-Full HPO] study=%s  trials=%s  warmstart=%d keys",
        study_name,
        n_trials,
        len(warmstart),
    )

    timer = _TrialTimer(f"LGBM-Full-{horizon_h}h", n_trials=n_trials)
    study = optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_DB,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
    )
    remaining = max(0, n_trials - len(study.trials))
    log.info("  [LGBM-Full HPO] %d done, running %d more", n_trials - remaining, remaining)

    t0 = time.perf_counter()
    study.optimize(
        lambda t: _lgbm_full_objective(t, X_tr, y_tr, X_va, y_va, warmstart, timer),
        n_trials=remaining,
        callbacks=[timer],
        show_progress_bar=False,
    )
    wall = time.perf_counter() - t0
    timing = timer.summary()
    log.info(
        "  [LGBM-Full HPO] done  best=%.4f  wall=%.1fs  %.1f trials/min",
        study.best_value,
        wall,
        timing.get("trials_per_min", 0),
    )
    return study.best_params, timing


def run_xgb_hpo(X_tr, y_tr, X_va, y_va, horizon_h: int, n_trials: int = 75) -> tuple[dict, dict]:
    study_name = f"xgb_{horizon_h}h"
    log.info("  [XGB HPO] study=%s  trials=%s", study_name, n_trials)

    timer = _TrialTimer(f"XGB-{horizon_h}h", n_trials=n_trials)
    study = optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_DB,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=43, n_startup_trials=20),
    )
    remaining = max(0, n_trials - len(study.trials))
    log.info("  [XGB HPO] %d done, running %d more", n_trials - remaining, remaining)

    t0 = time.perf_counter()
    study.optimize(
        lambda t: _xgb_objective(t, X_tr, y_tr, X_va, y_va, timer),
        n_trials=remaining,
        callbacks=[timer],
        show_progress_bar=False,
    )
    wall = time.perf_counter() - t0
    timing = timer.summary()
    log.info(
        "  [XGB HPO] done  best=%.4f  wall=%.1fs  %.1f trials/min",
        study.best_value,
        wall,
        timing.get("trials_per_min", 0),
    )
    return study.best_params, timing


def run_lgbm_physics_hpo(
    X_tr_phys, y_tr, X_va_phys, y_va, horizon_h: int, n_trials: int = 75
) -> tuple[dict, dict]:
    study_name = f"lgbm_physics_{horizon_h}h"
    log.info(
        "  [LGBM-Physics HPO] study=%s  trials=%s  features=%d",
        study_name,
        n_trials,
        X_tr_phys.shape[1],
    )

    timer = _TrialTimer(f"LGBM-Physics-{horizon_h}h", n_trials=n_trials)
    study = optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_DB,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=99, n_startup_trials=15),
    )
    remaining = max(0, n_trials - len(study.trials))
    log.info("  [LGBM-Physics HPO] %d done, running %d more", n_trials - remaining, remaining)

    t0 = time.perf_counter()
    study.optimize(
        lambda t: _lgbm_physics_objective(t, X_tr_phys, y_tr, X_va_phys, y_va, timer),
        n_trials=remaining,
        callbacks=[timer],
        show_progress_bar=False,
    )
    wall = time.perf_counter() - t0
    timing = timer.summary()
    log.info(
        "  [LGBM-Physics HPO] done  best=%.4f  wall=%.1fs  %.1f trials/min",
        study.best_value,
        wall,
        timing.get("trials_per_min", 0),
    )
    return study.best_params, timing


# ─── NNLS Meta-Learner ────────────────────────────────────────────────────────


class _NNLSMeta:
    """
    NNLS meta-learner with bias column and normalized weights.

    V11: OOF matrix is augmented with a column of ones before solving:
        [oof_lgbm | oof_xgb | oof_physics | 1] @ w ≈ y_meta

    This gives the meta-learner a free intercept to correct systematic bias
    in the base models (e.g., consistent underprediction of high-AQI events)
    without violating the non-negativity constraint.

    After solving, weights are normalized over the model columns only
    (not the bias), so the blend percentages remain interpretable.
    """

    def __init__(self):
        self.coef_: np.ndarray = np.array([])  # model weights (normalized)
        self.bias_: float = 0.0  # raw bias weight
        self._raw_w: np.ndarray = np.array([])

    def fit(self, X_oof: np.ndarray, y: np.ndarray) -> "_NNLSMeta":
        """
        X_oof: shape (n, n_models) — raw unscaled OOF residuals
        Bias column appended internally.
        """
        X_aug = np.c_[X_oof, np.ones(len(X_oof))]
        raw_w, _ = nnls(X_aug, y)

        # Normalize model weights (all columns except last) to sum=1
        model_w = raw_w[:-1]
        bias_w = raw_w[-1]
        total = model_w.sum()
        if total < 1e-10:
            model_w = np.ones(X_oof.shape[1]) / X_oof.shape[1]
            total = 1.0
        self._raw_w = raw_w
        self.coef_ = model_w / total
        self.bias_ = float(bias_w)
        return self

    def predict(self, X_oof: np.ndarray) -> np.ndarray:
        """X_oof: shape (n, n_models) — bias appended internally."""
        return self.predict_normalized(X_oof)

    def predict_normalized(self, X_oof: np.ndarray) -> np.ndarray:
        """
        Predict using normalized model weights + raw bias.
        This is the correct inference path: normalized blend + bias correction.
        """
        return X_oof @ self.coef_ + self.bias_


# ─── Final model builders ─────────────────────────────────────────────────────


def build_lgbm_full(best_params: dict, X_fit, y_fit, X_es, y_es) -> lgb.LGBMRegressor:
    """LGBM-Full: regression_l1, all features, 4000 trees."""
    params = {
        "objective": "regression_l1",
        "n_estimators": 4000,
        "n_jobs": N_JOBS,
        "verbosity": -1,
        "random_state": 42,
        "bagging_freq": 1,
        **best_params,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_es, y_es)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    log.info("    LGBM-Full final: best_iter=%s", model.best_iteration_)
    return model


def build_xgb(best_params: dict, X_fit, y_fit, X_es, y_es) -> XGBRegressor:
    """XGBoost: reg:absoluteerror, all features, 4000 trees."""
    params = {
        "objective": "reg:absoluteerror",
        "n_estimators": 4000,
        "n_jobs": N_JOBS,
        "verbosity": 0,
        "random_state": 42,
        "tree_method": "hist",
        **best_params,
    }
    model = XGBRegressor(**params, early_stopping_rounds=50, eval_metric="mae")
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    log.info("    XGB final: best_iter=%s", model.best_iteration)
    return model


def build_lgbm_physics(best_params: dict, X_fit_phys, y_fit, X_es_phys, y_es) -> lgb.LGBMRegressor:
    """LGBM-Physics: regression_l1, physics features only, 4000 trees."""
    params = {
        "objective": "regression_l1",
        "n_estimators": 4000,
        "n_jobs": N_JOBS,
        "verbosity": -1,
        "random_state": 99,
        "bagging_freq": 1,
        **best_params,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_fit_phys,
        y_fit,
        eval_set=[(X_es_phys, y_es)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    log.info("    LGBM-Physics final: best_iter=%s", model.best_iteration_)
    return model


# ─── OOF generation ───────────────────────────────────────────────────────────


def generate_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_h: int,
    lgbm_full_params: dict,
    xgb_params: dict,
    lgbm_physics_params: dict,
    physics_cols: list,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Out-of-Fold predictions for all 3 base models.

    Validation: TimeSeriesSplit(n_splits=5, gap=horizon_h)
    Returns: (oof_lgbm_full, oof_xgb, oof_lgbm_physics) — shape (n_samples,)
    NaN where the sample fell in the initial gap.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon_h)

    oof_lgbm_full = np.full(len(X), np.nan)
    oof_xgb = np.full(len(X), np.nan)
    oof_lgbm_physics = np.full(len(X), np.nan)

    X_arr = X.values
    y_arr = y.values

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
        log.info(
            "    Fold %d/%d  train=%d  val=%d", fold_idx + 1, n_splits, len(train_idx), len(val_idx)
        )

        X_tr_raw, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_va_raw, y_va = X_arr[val_idx], y_arr[val_idx]

        # Impute within fold — fit on train only
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr_raw)
        X_va_imp = imp.transform(X_va_raw)

        X_tr_df = pd.DataFrame(X_tr_imp, columns=X.columns)
        X_va_df = pd.DataFrame(X_va_imp, columns=X.columns)

        # Early stopping split: last 10% of training fold
        es_split = int(len(X_tr_df) * 0.9)
        X_fit_f, y_fit_f = X_tr_df.iloc[:es_split], y_tr[:es_split]
        X_es_f, y_es_f = X_tr_df.iloc[es_split:], y_tr[es_split:]

        # ── LGBM-Full fold ──
        lgbm_f = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=2000,
            n_jobs=N_JOBS,
            verbosity=-1,
            random_state=42,
            bagging_freq=1,
            **lgbm_full_params,
        )
        lgbm_f.fit(
            X_fit_f,
            y_fit_f,
            eval_set=[(X_es_f, y_es_f)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_lgbm_full[val_idx] = lgbm_f.predict(X_va_df)

        # ── XGBoost fold ──
        xgb_f = XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=2000,
            n_jobs=N_JOBS,
            verbosity=0,
            random_state=42,
            tree_method="hist",
            early_stopping_rounds=30,
            eval_metric="mae",
            **xgb_params,
        )
        xgb_f.fit(X_fit_f, y_fit_f, eval_set=[(X_es_f, y_es_f)], verbose=False)
        oof_xgb[val_idx] = xgb_f.predict(X_va_df)

        # ── LGBM-Physics fold — physics features only ──
        avail_phys = [c for c in physics_cols if c in X.columns]
        X_tr_phys_f = X_tr_df[avail_phys]
        X_va_phys_f = X_va_df[avail_phys]
        X_fit_phys_f = X_fit_f[avail_phys]
        X_es_phys_f = X_es_f[avail_phys]

        phys_f = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=2000,
            n_jobs=N_JOBS,
            verbosity=-1,
            random_state=99,
            bagging_freq=1,
            **lgbm_physics_params,
        )
        phys_f.fit(
            X_fit_phys_f,
            y_fit_f,
            eval_set=[(X_es_phys_f, y_es_f)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_lgbm_physics[val_idx] = phys_f.predict(X_va_phys_f)

    return oof_lgbm_full, oof_xgb, oof_lgbm_physics


# ─── Per-horizon ensemble trainer ─────────────────────────────────────────────


def train_ensemble_horizon(df: pd.DataFrame, horizon_h: int, val_cutoff: datetime) -> dict:
    """Full V11 ensemble pipeline for one horizon."""
    log.info("=" * 65)
    log.info("  HORIZON: %sh", horizon_h)
    log.info("=" * 65)

    # ── Checkpoint ────────────────────────────────────────────────────────
    if _horizon_is_complete(horizon_h):
        log.info("  ✓ Checkpoint found — skipping %sh.", horizon_h)
        report = _read_report()
        for r in report.get("horizons", []):
            if r.get("horizon_h") == horizon_h:
                return r
        return {"horizon_h": horizon_h, "skipped": True}

    t_start = time.perf_counter()

    # ── 1. Feature engineering ────────────────────────────────────────────
    log.info("  Step 1/9: Building features...")
    t0 = time.perf_counter()
    X_full, y_full = engineer_features(df, horizon_h)
    mask = y_full.notna()
    X_full, y_full = X_full[mask], y_full[mask]
    log.info(
        "  Features: %s rows × %s cols  (%.1fs)",
        f"{len(X_full):,}",
        len(X_full.columns),
        time.perf_counter() - t0,
    )

    # ── 2. Physics column selection ───────────────────────────────────────
    physics_cols = _get_physics_cols(X_full)
    log.info(
        "  Physics features available: %d / %d requested", len(physics_cols), len(PHYSICS_FEATURES)
    )
    if len(physics_cols) < 10:
        raise RuntimeError(
            f"Too few physics features ({len(physics_cols)}). "
            "Check PHYSICS_FEATURES list against feature names."
        )

    # ── 3. Temporal train / val split ─────────────────────────────────────
    train_mask = X_full.index < val_cutoff
    val_mask = X_full.index >= val_cutoff
    X_train, y_train = X_full[train_mask], y_full[train_mask]
    X_val, y_val = X_full[val_mask], y_full[val_mask]
    log.info("  Train: %s  |  Val: %s", f"{len(X_train):,}", f"{len(X_val):,}")

    if len(X_train) < 500:
        raise RuntimeError(f"Too few training rows ({len(X_train)}).")

    # ── 4. Impute (fit on train only) ─────────────────────────────────────
    log.info("  Step 2/9: Imputing...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    joblib.dump(imputer, MODELS_DIR / f"imputer_{horizon_h}h.pkl")

    X_train_df = pd.DataFrame(X_train_imp, columns=X_train.columns, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_imp, columns=X_val.columns, index=X_val.index)

    # ── 5. HPO eval set — fold 5 of TimeSeriesSplit (V11 fix) ────────────
    # Use the SAME split as OOF generation so HPO params are optimal for
    # the same data distribution the meta-learner will see.
    log.info("  Step 3/9: Computing HPO eval set (fold 5 of TimeSeriesSplit)...")
    tscv_hpo = TimeSeriesSplit(n_splits=5, gap=horizon_h)
    folds = list(tscv_hpo.split(X_train_df))
    hpo_tr_idx, hpo_va_idx = folds[-1]  # fold 5 = last fold

    X_hpo_tr = X_train_df.iloc[hpo_tr_idx]
    y_hpo_tr = y_train.values[hpo_tr_idx]
    X_hpo_va = X_train_df.iloc[hpo_va_idx]
    y_hpo_va = y_train.values[hpo_va_idx]

    # Physics subset for physics HPO
    X_hpo_tr_phys = X_hpo_tr[physics_cols]
    X_hpo_va_phys = X_hpo_va[physics_cols]

    log.info(
        "  HPO fold 5: train=%d  val=%d  physics_features=%d",
        len(X_hpo_tr),
        len(X_hpo_va),
        len(physics_cols),
    )

    # ── 6. Optuna HPO ─────────────────────────────────────────────────────
    # V15 Tier 2: Increased HPO budget (50→150, 75→200, 75→150).
    # With load_if_exists=True, incremental runs add to existing trials.
    log.info("  Step 4/9: LGBM-Full HPO (150 trials)...")
    lgbm_full_params, lgbm_full_timing = run_lgbm_full_hpo(
        X_hpo_tr, y_hpo_tr, X_hpo_va, y_hpo_va, horizon_h, n_trials=150
    )

    log.info("  Step 5/9: XGBoost HPO (200 trials)...")
    xgb_params, xgb_timing = run_xgb_hpo(
        X_hpo_tr, y_hpo_tr, X_hpo_va, y_hpo_va, horizon_h, n_trials=200
    )

    log.info("  Step 6/9: LGBM-Physics HPO (150 trials)...")
    lgbm_physics_params, lgbm_physics_timing = run_lgbm_physics_hpo(
        X_hpo_tr_phys, y_hpo_tr, X_hpo_va_phys, y_hpo_va, horizon_h, n_trials=150
    )

    hpo_results = {
        "lgbm_full": lgbm_full_params,
        "xgb": xgb_params,
        "lgbm_physics": lgbm_physics_params,
    }
    (MODELS_DIR / f"hpo_params_{horizon_h}h.json").write_text(json.dumps(hpo_results, indent=2))

    # ── 7. OOF predictions ────────────────────────────────────────────────
    log.info("  Step 7/9: Generating OOF predictions (5-fold walk-forward)...")
    t_oof = time.perf_counter()
    oof_lgbm_full, oof_xgb, oof_lgbm_physics = generate_oof_predictions(
        X_train_df,
        y_train,
        horizon_h=horizon_h,
        lgbm_full_params=lgbm_full_params,
        xgb_params=xgb_params,
        lgbm_physics_params=lgbm_physics_params,
        physics_cols=physics_cols,
        n_splits=5,
    )
    oof_wall_s = round(time.perf_counter() - t_oof, 2)
    log.info("  OOF complete  wall=%.1fs", oof_wall_s)

    # ── 8. NNLS meta-learner with bias column ─────────────────────────────
    log.info("  Step 8/9: Training NNLS meta-learner (with bias)...")
    oof_stack = np.column_stack([oof_lgbm_full, oof_xgb, oof_lgbm_physics])
    valid_mask = ~np.any(np.isnan(oof_stack), axis=1)
    oof_clean = oof_stack[valid_mask]
    y_meta = y_train.values[valid_mask]

    log.info("  OOF valid rows: %d / %d", valid_mask.sum(), len(y_train))

    oof_corr = np.corrcoef(oof_clean.T)
    log.info("  OOF correlation matrix:\n%s", np.round(oof_corr, 3))

    meta = _NNLSMeta()
    meta.fit(oof_clean, y_meta)

    log.info(
        "  NNLS weights: LGBM-Full=%.4f  XGB=%.4f  LGBM-Physics=%.4f  bias=%.4f  (sum=%.4f)",
        meta.coef_[0],
        meta.coef_[1],
        meta.coef_[2],
        meta.bias_,
        meta.coef_.sum(),
    )

    # ── 9. Final base model training (ES_DAYS=60, V11 fix) ───────────────
    log.info("  Step 9/9: Training final base models (4000 trees, ES=60d)...")
    ES_DAYS = 60
    es_cutoff = val_cutoff - timedelta(days=ES_DAYS)
    es_mask = X_train_df.index >= es_cutoff

    X_fit_df = X_train_df[~es_mask]
    y_fit = y_train[~es_mask]
    X_es_df = X_train_df[es_mask]
    y_es_arr = y_train[es_mask]
    log.info("  Fit rows: %d  |  ES rows: %d", len(X_fit_df), len(X_es_df))

    t1 = time.perf_counter()
    lgbm_full_final = build_lgbm_full(lgbm_full_params, X_fit_df, y_fit, X_es_df, y_es_arr)
    t_lgbm_s = round(time.perf_counter() - t1, 2)

    t2 = time.perf_counter()
    xgb_final = build_xgb(xgb_params, X_fit_df, y_fit, X_es_df, y_es_arr)
    t_xgb_s = round(time.perf_counter() - t2, 2)

    t3 = time.perf_counter()
    lgbm_physics_final = build_lgbm_physics(
        lgbm_physics_params,
        X_fit_df[physics_cols],
        y_fit,
        X_es_df[physics_cols],
        y_es_arr,
    )
    t_phys_s = round(time.perf_counter() - t3, 2)

    log.info(
        "  Build times — LGBM-Full=%.1fs  XGB=%.1fs  LGBM-Physics=%.1fs",
        t_lgbm_s,
        t_xgb_s,
        t_phys_s,
    )

    # ── Save artifacts ────────────────────────────────────────────────────
    joblib.dump(lgbm_full_final, MODELS_DIR / f"lgbm_full_{horizon_h}h.pkl")
    joblib.dump(xgb_final, MODELS_DIR / f"xgb_{horizon_h}h.pkl")
    joblib.dump(lgbm_physics_final, MODELS_DIR / f"lgbm_physics_{horizon_h}h.pkl")
    (MODELS_DIR / f"physics_cols_{horizon_h}h.json").write_text(json.dumps(physics_cols, indent=2))
    # Sentinel LAST
    joblib.dump(meta, MODELS_DIR / f"meta_learner_{horizon_h}h.pkl")
    log.info("  ✓ Artifacts saved  (sentinel: meta_learner_%sh.pkl)", horizon_h)

    # ── Validation metrics ────────────────────────────────────────────────
    if len(X_val_df) == 0:
        log.warning("  No validation data.")
        result = {"horizon_h": horizon_h}
        _upsert_horizon_result(result)
        return result

    log.info("  Computing validation metrics...")
    base_aqi_val = X_val["aqi_current"].values

    pred_lgbm_full = np.clip(lgbm_full_final.predict(X_val_df) + base_aqi_val, 0, 500)
    pred_xgb = np.clip(xgb_final.predict(X_val_df) + base_aqi_val, 0, 500)
    pred_lgbm_physics = np.clip(
        lgbm_physics_final.predict(X_val_df[physics_cols]) + base_aqi_val, 0, 500
    )

    # Ensemble: raw residuals → meta-learner (bias applied internally)
    val_oof_stack = np.column_stack(
        [
            pred_lgbm_full - base_aqi_val,
            pred_xgb - base_aqi_val,
            pred_lgbm_physics - base_aqi_val,
        ]
    )
    pred_meta = np.clip(meta.predict_normalized(val_oof_stack) + base_aqi_val, 0, 500)
    y_val_abs = y_val.values + base_aqi_val

    def _m(name, preds):
        mae = mean_absolute_error(y_val_abs, preds)
        r2 = r2_score(y_val_abs, preds)
        log.info("  %-26s  MAE=%6.3f  R²=%6.4f", name, mae, r2)
        return {"mae": round(float(mae), 3), "r2": round(float(r2), 4)}

    horizon_wall_s = round(time.perf_counter() - t_start, 2)

    result = {
        "horizon_h": horizon_h,
        "horizon_wall_s": horizon_wall_s,
        "baseline_lgbm_v9": _load_v9_baseline(horizon_h),
        "lgbm_full_v11": _m("LGBM-Full V11", pred_lgbm_full),
        "xgb_v11": _m("XGBoost V11", pred_xgb),
        "lgbm_physics_v11": _m("LGBM-Physics V11", pred_lgbm_physics),
        "ensemble_v11": _m("★ ENSEMBLE V11", pred_meta),
        "meta_coefficients": {
            "lgbm_full": round(float(meta.coef_[0]), 4),
            "xgb": round(float(meta.coef_[1]), 4),
            "lgbm_physics": round(float(meta.coef_[2]), 4),
            "bias": round(float(meta.bias_), 4),
            "sum": round(float(meta.coef_.sum()), 6),
        },
        "oof_diversity": {
            "lgbm_full_xgb_corr": round(float(oof_corr[0, 1]), 4),
            "lgbm_full_physics_corr": round(float(oof_corr[0, 2]), 4),
            "xgb_physics_corr": round(float(oof_corr[1, 2]), 4),
        },
        "physics_features_used": len(physics_cols),
        "hpo_params": hpo_results,
        "benchmark_9800x3d": {
            "lgbm_full_hpo": lgbm_full_timing,
            "xgb_hpo": xgb_timing,
            "lgbm_physics_hpo": lgbm_physics_timing,
            "oof_generation_s": oof_wall_s,
            "final_build_s": {
                "lgbm_full": t_lgbm_s,
                "xgb": t_xgb_s,
                "lgbm_physics": t_phys_s,
            },
        },
    }

    _upsert_horizon_result(result)
    log.info(
        "  Horizon %sh complete  wall=%.1fs (%.1f min)",
        horizon_h,
        horizon_wall_s,
        horizon_wall_s / 60,
    )
    return result


# ─── Tournament report printer ────────────────────────────────────────────────


def _print_tournament_table(report: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  V11 FEATURE-DIVERSITY ENSEMBLE — TOURNAMENT REPORT")
    log.info("=" * 80)
    log.info("  %-8s %-28s %-10s %-10s", "Horizon", "Model", "MAE", "R²")
    log.info("  " + "-" * 60)

    for hr in report.get("horizons", []):
        h = hr["horizon_h"]
        rows = [
            ("Baseline LGBM V9", hr.get("baseline_lgbm_v9", {})),
            ("LGBM-Full V11", hr.get("lgbm_full_v11", {})),
            ("XGBoost V11", hr.get("xgb_v11", {})),
            ("LGBM-Physics V11", hr.get("lgbm_physics_v11", {})),
            ("★ ENSEMBLE V11", hr.get("ensemble_v11", {})),
        ]
        for name, m in rows:
            mae = f"{m['mae']:.3f}" if m.get("mae") is not None else "N/A"
            r2 = f"{m['r2']:.4f}" if m.get("r2") is not None else "N/A"
            log.info("  %-8s %-28s %-10s %-10s", f"{h}h", name, mae, r2)
        log.info("  " + "-" * 60)

    log.info("")
    log.info("  META-LEARNER WEIGHTS (NNLS + bias, normalized):")
    for hr in report.get("horizons", []):
        h = hr["horizon_h"]
        coef = hr.get("meta_coefficients", {})
        log.info(
            "  %sh  LGBM-Full=%.4f  XGB=%.4f  LGBM-Physics=%.4f  bias=%.4f  (sum=%.4f)",
            h,
            coef.get("lgbm_full", 0),
            coef.get("xgb", 0),
            coef.get("lgbm_physics", 0),
            coef.get("bias", 0),
            coef.get("sum", 0),
        )

    log.info("")
    log.info("  OOF DIVERSITY (Pearson r — lower = more diverse):")
    for hr in report.get("horizons", []):
        h = hr["horizon_h"]
        div = hr.get("oof_diversity", {})
        log.info(
            "  %sh  Full↔XGB=%.3f  Full↔Physics=%.3f  XGB↔Physics=%.3f",
            h,
            div.get("lgbm_full_xgb_corr", 0),
            div.get("lgbm_full_physics_corr", 0),
            div.get("xgb_physics_corr", 0),
        )

    log.info("")
    log.info("  9800X3D BENCHMARK:")
    log.info(
        "  %-8s %-26s %-14s %-10s %-10s", "Horizon", "Study", "Trials/min", "Mean(s)", "Total(s)"
    )
    log.info("  " + "-" * 72)
    for hr in report.get("horizons", []):
        h = hr["horizon_h"]
        bm = hr.get("benchmark_9800x3d", {})
        for key, label in [
            ("lgbm_full_hpo", "LGBM-Full HPO"),
            ("xgb_hpo", "XGBoost HPO"),
            ("lgbm_physics_hpo", "LGBM-Physics HPO"),
        ]:
            t = bm.get(key, {})
            if not t:
                continue
            log.info(
                "  %-8s %-26s %-14.1f %-10.1f %-10.1f",
                f"{h}h",
                label,
                t.get("trials_per_min", 0),
                t.get("mean_s", 0),
                t.get("total_s", 0),
            )
        fb = bm.get("final_build_s", {})
        if fb:
            log.info(
                "  %-8s %-26s  OOF=%.1fs  Full=%.1fs  XGB=%.1fs  Physics=%.1fs",
                f"{h}h",
                "Final Training",
                bm.get("oof_generation_s", 0),
                fb.get("lgbm_full", 0),
                fb.get("xgb", 0),
                fb.get("lgbm_physics", 0),
            )
        log.info("  " + "-" * 72)

    total = report.get("total_run_s")
    if total:
        log.info("  Total wall time: %.1fs (%.1f min)", total, total / 60)
    log.info("=" * 80)


# ─── Main ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V11 Feature-Diversity Ensemble")
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[6, 12, 24, 48],
        default=None,
        metavar="{6,12,24,48}",
        help="Run only this horizon (default: all 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    horizons_to_run = [args.horizon] if args.horizon else HORIZONS

    log.info("=" * 65)
    log.info("  Folsom AQI Navigator — V11 Feature-Diversity Ensemble")
    log.info("  Horizons: %s", horizons_to_run)
    log.info("  Optuna DB: %s", OPTUNA_DB)
    log.info("  n_jobs: %s  |  Output: %s", N_JOBS, MODELS_DIR)
    log.info("  Physics features: %d", len(PHYSICS_FEATURES))
    log.info("=" * 65)

    _run_start = time.perf_counter()

    # Dependency check
    for pkg in ("xgboost", "optuna"):
        try:
            __import__(pkg)
        except ImportError:
            log.error("Missing package: %s  →  pip install %s", pkg, pkg)
            sys.exit(1)

    # Fetch data
    log.info("Step 1: Fetching historical data...")
    df = fetch_full_history()
    df.to_parquet(DATA_DIR / "historical.parquet")
    log.info(
        "  Rows: %s  |  Range: %s → %s",
        f"{len(df):,}",
        df.index.min().date(),
        df.index.max().date(),
    )

    val_cutoff = datetime.now(tz=df.index.tz) - timedelta(days=60)
    log.info("  Val cutoff: %s", val_cutoff.strftime("%Y-%m-%d"))

    # Train
    log.info("Step 2: Training V11 ensemble: %s", horizons_to_run)
    for h in horizons_to_run:
        train_ensemble_horizon(df, h, val_cutoff)

    # Feature names
    feature_names = get_feature_names(6)
    (MODELS_DIR / "feature_names_v11.json").write_text(json.dumps(feature_names, indent=2))
    log.info("  Feature names saved (%d features)", len(feature_names))

    # Finalise report
    total_run_s = round(time.perf_counter() - _run_start, 2)
    report = _read_report()
    report["total_run_s"] = total_run_s
    report["completed_at"] = datetime.now().isoformat()
    _write_report(report)

    _print_tournament_table(report)

    log.info("V11 complete.  Total wall time: %.1fs (%.1f min)", total_run_s, total_run_s / 60)
    log.info("Artifacts: %s", MODELS_DIR)


if __name__ == "__main__":
    main()
