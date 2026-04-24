# Table 1: AQI Forecast Performance — 2025 Holdout (Folsom, CA)

**Evaluation period:** 2025-01-01 to 2025-12-31 (8,760 hourly observations)
**Training period:** 2019-01-01 to 2024-12-31
**Skill Score** = 1 − MAE_model / MAE_persistence (higher is better)

| Horizon | Model | MAE (AQI) | R² | Skill Score |
|---------|-------|----------:|---:|------------:|
| 6h | Persistence | 10.61 | 0.077 | --- |
|  | Climatology | 18.16 | -0.674 | -71.2% |
|  | V15 Ablated (no fire) | 2.80 | 0.950 | +73.6% |
| **** | **V15 Full (ours)** | **2.81** | **0.935** | **+73.5%** |
| 12h | Persistence | 14.30 | -0.362 | --- |
|  | Climatology | 19.47 | -0.920 | -36.2% |
|  | V15 Ablated (no fire) | 5.32 | 0.808 | +62.8% |
| **** | **V15 Full (ours)** | **5.37** | **0.764** | **+62.4%** |
| 24h | Persistence | 9.97 | 0.468 | --- |
|  | Climatology | 15.14 | -0.176 | -51.9% |
|  | V15 Ablated (no fire) | 7.21 | 0.721 | +27.7% |
| **** | **V15 Full (ours)** | **7.20** | **0.626** | **+27.8%** |
| 48h | Persistence | 12.84 | 0.159 | --- |
|  | Climatology | 14.84 | -0.140 | -15.6% |
|  | V15 Ablated (no fire) | 8.45 | 0.633 | +34.2% |
| **** | **V15 Full (ours)** | **8.50** | **0.513** | **+33.8%** |

> **Bold** = our proposed V15 Full model.
> Skill Score measures improvement over the Persistence baseline.
> V15 Ablated removes all FIRMS fire-detection features while retaining
> wind-derived trajectory origin coordinates.