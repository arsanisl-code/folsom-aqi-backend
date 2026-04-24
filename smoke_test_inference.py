"""
test_inference.py — Quick smoke test: loads models, runs one prediction, prints JSON.
Run this before deploying to verify the inference pipeline works end-to-end.

Expected output: a JSON blob printed to stdout, matching the /forecast schema.
Runtime: ~10-30 seconds (includes API calls).

Usage:
    python test_inference.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 60)
    print("  Folsom AQI — Inference Smoke Test")
    print("=" * 60)

    # Verify models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        print("\n[ERROR] models/ directory not found.", file=sys.stderr)
        print("Run train.py first, then re-run this test.", file=sys.stderr)
        sys.exit(1)

    expected_files = []
    for h in [6, 12, 24, 48]:
        for kind in ["point", "q05", "q95"]:
            expected_files.append(f"models/lgbm_{kind}_{h}h.pkl")
        expected_files.append(f"models/imputer_{h}h.pkl")

    missing = [f for f in expected_files if not Path(f).exists()]
    if missing:
        print(f"\n[ERROR] Missing {len(missing)} model files:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        print("\nRun train.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Found all {len(expected_files)} model files.")

    # Load models
    print("\nLoading models into memory...")
    from inference import load_all_models

    models = load_all_models()
    print(f"✓ Loaded models for horizons: {list(models.keys())}")

    # Run inference
    print("\nFetching data and running inference...")
    print("(This calls Open-Meteo and AirNow APIs — needs internet access)\n")
    from inference import predict_now

    result = predict_now()

    # Print schema-compliant JSON
    print("\n" + "=" * 60)
    print("  FORECAST OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))

    # Validate key fields
    print("\n" + "=" * 60)
    print("  VALIDATION CHECKS")
    print("=" * 60)

    checks = [
        ("generated_at present", bool(result.get("generated_at"))),
        ("location.name correct", result.get("location", {}).get("name") == "Folsom, CA"),
        ("current.aqi is int", isinstance(result.get("current", {}).get("aqi"), int)),
        ("current.source present", bool(result.get("current", {}).get("source"))),
        ("forecast 6h present", "6h" in result.get("forecasts", {})),
        ("forecast 12h present", "12h" in result.get("forecasts", {})),
        ("forecast 24h present", "24h" in result.get("forecasts", {})),
        ("forecast 48h present", "48h" in result.get("forecasts", {})),
        ("history_72h is list", isinstance(result.get("history_72h"), list)),
        ("history_72h ≤ 72 items", len(result.get("history_72h", [])) <= 72),
        ("model_version present", bool(result.get("model_version"))),
    ]

    # Check CI bounds
    for h_key, fc in result.get("forecasts", {}).items():
        checks.append((f"{h_key} ci_lo ≤ aqi", fc.get("ci_lo", 999) <= fc.get("aqi", 0)))
        checks.append((f"{h_key} aqi ≤ ci_hi", fc.get("aqi", 999) <= fc.get("ci_hi", 0)))
        checks.append((f"{h_key} aqi in [0,500]", 0 <= fc.get("aqi", -1) <= 500))

    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All checks passed. Inference pipeline is working correctly.")
        print(f"\nCurrent AQI: {result['current']['aqi']} — {result['current']['category']}")
        print(f"Source:      {result['current']['source']}")
        for h_key, fc in result["forecasts"].items():
            print(
                f"Forecast {h_key}: {fc['aqi']} AQI [{fc['ci_lo']}–{fc['ci_hi']}] — {fc['category']}"
            )
        print("\nNext: run uvicorn api:app --reload  and open http://localhost:8000/forecast")
    else:
        print("\n✗ Some checks failed. Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
