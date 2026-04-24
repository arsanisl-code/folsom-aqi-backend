"""
generate_paper_tables.py — Module 3: Academic Metric Synthesizer.

Reads three JSON artifacts and produces a publication-ready comparison table:
  - models_v6/training_metrics_v6.json   → V15 Full model (val set)
  - backtest_v12_2025_report.json        → V15 Full model (2025 holdout)
  - models_v6/ablation_metrics.json      → V15 Ablated (no fire features)
  - models_v6/naive_baselines.json       → Persistence + Climatology

Outputs:
  - paper_table_1.md   — Markdown table for direct inclusion in paper
  - paper_table_1.csv  — CSV for LaTeX/Excel import

Metrics reported:
  - MAE (Mean Absolute Error, AQI units)
  - R²  (Coefficient of Determination)
  - Skill Score = 1 - MAE_model / MAE_persistence  (% improvement over naive)

No training, no data loading — pure synthesis.
"""

import csv
import json
import sys
from pathlib import Path

MODELS_DIR = Path("models_v6")

# Input paths
HOLDOUT_PATH = Path("backtest_v12_2025_report.json")
ABLATION_PATH = MODELS_DIR / "ablation_metrics.json"
BASELINES_PATH = MODELS_DIR / "naive_baselines.json"

# Output paths
MD_PATH = Path("paper_table_1.md")
CSV_PATH = Path("paper_table_1.csv")

HORIZONS = [6, 12, 24, 48]


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: Missing required file: {path}")
        sys.exit(1)
    return json.loads(path.read_text())


def _skill(mae_model: float, mae_persistence: float) -> float:
    """Skill score: fraction of persistence error eliminated. Higher = better."""
    if mae_persistence == 0:
        return 0.0
    return round((1 - mae_model / mae_persistence) * 100, 1)


def generate():
    print("=" * 65)
    print("  Academic Synthesizer — Paper Table 1")
    print("=" * 65)

    holdout = _load(HOLDOUT_PATH)
    ablation = _load(ABLATION_PATH)
    baselines = _load(BASELINES_PATH)

    # ── Index all data by horizon ─────────────────────────────────────────
    v15_full: dict[int, dict] = {r["horizon_h"]: r for r in holdout["annual"]}
    ablated: dict[int, dict] = {r["horizon_h"]: r for r in ablation["horizons"]}
    persistence: dict[int, dict] = {
        r["horizon_h"]: r for r in baselines["baselines"]["persistence"]
    }
    climatology: dict[int, dict] = {
        r["horizon_h"]: r for r in baselines["baselines"]["climatology"]
    }

    # ── Verify all horizons present ───────────────────────────────────────
    for h in HORIZONS:
        for name, d in [
            ("V15 Full", v15_full),
            ("Ablated", ablated),
            ("Persistence", persistence),
            ("Climatology", climatology),
        ]:
            if h not in d:
                print(f"ERROR: Missing horizon {h}h in {name}")
                sys.exit(1)

    # ── Build rows ────────────────────────────────────────────────────────
    rows = []
    for h in HORIZONS:
        p_mae = persistence[h]["mae"]
        rows.append(
            {
                "horizon": f"{h}h",
                "model": "Persistence",
                "mae": p_mae,
                "r2": persistence[h]["r2"],
                "skill": 0.0,
            }
        )
        rows.append(
            {
                "horizon": f"{h}h",
                "model": "Climatology",
                "mae": climatology[h]["mae"],
                "r2": climatology[h]["r2"],
                "skill": _skill(climatology[h]["mae"], p_mae),
            }
        )
        rows.append(
            {
                "horizon": f"{h}h",
                "model": "V15 Ablated (no fire)",
                "mae": ablated[h]["mae"],
                "r2": ablated[h]["r2"],
                "skill": _skill(ablated[h]["mae"], p_mae),
            }
        )
        rows.append(
            {
                "horizon": f"{h}h",
                "model": "V15 Full (ours)",
                "mae": v15_full[h]["mae_v12"],
                "r2": v15_full[h]["r2_v12"],
                "skill": _skill(v15_full[h]["mae_v12"], p_mae),
            }
        )

    # ── Markdown table ────────────────────────────────────────────────────
    md_lines = [
        "# Table 1: AQI Forecast Performance — 2025 Holdout (Folsom, CA)",
        "",
        "**Evaluation period:** 2025-01-01 to 2025-12-31 (8,760 hourly observations)",
        "**Training period:** 2019-01-01 to 2024-12-31",
        "**Skill Score** = 1 − MAE_model / MAE_persistence (higher is better)",
        "",
        "| Horizon | Model | MAE (AQI) | R² | Skill Score |",
        "|---------|-------|----------:|---:|------------:|",
    ]

    prev_horizon = None
    for r in rows:
        sep = "| | | | | |" if r["horizon"] == prev_horizon else None
        if sep and r["model"] == "Climatology":
            pass  # no separator needed within same horizon
        horizon_cell = r["horizon"] if r["horizon"] != prev_horizon else ""
        bold_open = "**" if r["model"] == "V15 Full (ours)" else ""
        bold_close = "**" if r["model"] == "V15 Full (ours)" else ""
        skill_str = f"{r['skill']:+.1f}%" if r["model"] != "Persistence" else "---"
        md_lines.append(
            f"| {bold_open}{horizon_cell}{bold_close} "
            f"| {bold_open}{r['model']}{bold_close} "
            f"| {bold_open}{r['mae']:.2f}{bold_close} "
            f"| {bold_open}{r['r2']:.3f}{bold_close} "
            f"| {bold_open}{skill_str}{bold_close} |"
        )
        prev_horizon = r["horizon"]

    md_lines += [
        "",
        "> **Bold** = our proposed V15 Full model.",
        "> Skill Score measures improvement over the Persistence baseline.",
        "> V15 Ablated removes all FIRMS fire-detection features while retaining",
        "> wind-derived trajectory origin coordinates.",
    ]

    MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Markdown table → {MD_PATH}")

    # ── CSV ───────────────────────────────────────────────────────────────
    fieldnames = ["horizon", "model", "mae", "r2", "skill_score_pct"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "horizon": r["horizon"],
                    "model": r["model"],
                    "mae": r["mae"],
                    "r2": r["r2"],
                    "skill_score_pct": r["skill"] if r["model"] != "Persistence" else "",
                }
            )
    print(f"  CSV table → {CSV_PATH}")

    # ── Console summary ───────────────────────────────────────────────────
    print()
    print(f"{'Horizon':<8} {'Model':<25} {'MAE':>8} {'R²':>7} {'Skill':>8}")
    print("-" * 60)
    for r in rows:
        skill_str = f"{r['skill']:+.1f}%" if r["model"] != "Persistence" else "---"
        marker = " ◄" if r["model"] == "V15 Full (ours)" else ""
        print(
            f"{r['horizon']:<8} {r['model']:<25} {r['mae']:>8.2f} "
            f"{r['r2']:>7.3f} {skill_str:>8}{marker}"
        )
        if r["model"] == "V15 Full (ours)" and r["horizon"] != "48h":
            print()

    # ── Reviewer assertions ───────────────────────────────────────────────
    print()
    print("REVIEWER CHECKS:")
    all_pass = True
    for h in HORIZONS:
        v15_mae = v15_full[h]["mae_v12"]
        p_mae = persistence[h]["mae"]
        abl_mae = ablated[h]["mae"]
        v15_r2 = v15_full[h]["r2_v12"]
        p_r2 = persistence[h]["r2"]

        checks = [
            (v15_mae < p_mae, f"{h}h: V15 MAE ({v15_mae}) < Persistence MAE ({p_mae})"),
            (v15_r2 > p_r2, f"{h}h: V15 R² ({v15_r2}) > Persistence R² ({p_r2})"),
        ]
        for passed, msg in checks:
            status = "✓" if passed else "✗ FAIL"
            print(f"  [{status}] {msg}")
            if not passed:
                all_pass = False

    if all_pass:
        print("  All checks passed.")
    else:
        print("  WARNING: Some checks failed — review results before publication.")

    print()
    print("Paper Table 1 generation complete.")


if __name__ == "__main__":
    generate()
