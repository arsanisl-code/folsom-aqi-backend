import pandas as pd, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("backtest_v6_2025_report.csv")

v82 = {"6h": (3.57, 0.868), "12h": (5.31, 0.777), "24h": (7.51, 0.607), "48h": (8.50, 0.494)}
print("=== V8.2 vs V9 COMPARISON (2025 Holdout) ===")
print(f"  {'Horizon':<8} {'V8.2 MAE':>8} {'V9 MAE':>7} {'dMAE':>7}   {'V8.2 R2':>8} {'V9 R2':>6} {'dR2':>7}")
print(f"  {'-'*8} {'-'*8} {'-'*7} {'-'*7}   {'-'*8} {'-'*6} {'-'*7}")
for h in ["6h","12h","24h","48h"]:
    sub = df[df.Horizon==h]
    mae = sub.MAE.mean()
    r2  = sub.R2.mean()
    old_mae, old_r2 = v82[h]
    d_mae = mae - old_mae
    d_r2  = r2 - old_r2
    arrow_mae = "UP" if d_mae < 0 else "DN"
    arrow_r2  = "UP" if d_r2 > 0 else "DN"
    print(f"  {h:>8} {old_mae:>8.2f} {mae:>7.2f} {d_mae:>+7.2f} {arrow_mae}   {old_r2:>8.3f} {r2:>6.3f} {d_r2:>+7.3f} {arrow_r2}")

print()
print("=== SPRING PERFORMANCE ===")
for h in ["24h","48h"]:
    sub_v9 = df[(df.Horizon==h) & df.Month.isin([3,4,5])]
    print(f"  {h} Spring: MAE={sub_v9.MAE.mean():.2f} R2={sub_v9.R2.mean():.3f} Skill={sub_v9.Skill.mean():.3f}")
    sub_v9_all = df[(df.Horizon==h)]
    print(f"  {h} Annual: MAE={sub_v9_all.MAE.mean():.2f} R2={sub_v9_all.R2.mean():.3f} Skill={sub_v9_all.Skill.mean():.3f}")

print()
print("=== TOP 30 FEATURES (48h V9) ===")
m48 = joblib.load("models_v6/lgbm_point_48h.pkl")
cols = m48.feature_name_
imps = m48.feature_importances_
new_features = {"stability_index","trapping_power","fwd_ventilation_stress","volatility_frontal",
                "summer_photochem_accum","blh_collapse_rate","fwd_blh_collapse_rate","evening_trap_flag"}
top = sorted(zip(cols, imps), key=lambda x: -x[1])[:35]
for i,(name,imp) in enumerate(top):
    tag = " <<< NEW V9" if name in new_features else ""
    print(f"  {i+1:2d}. {name:42s} {imp:6d}{tag}")
