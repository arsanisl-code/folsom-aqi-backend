"""
app.py — Folsom AQI Monitor · FLC Los Rios STEM Fair 2026
Live AQI forecast dashboard backed by FastAPI + LightGBM on Render.

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → Streamlit Community Cloud
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── V6 Model Metadata (loaded once at import) ────────────────────────────────

_V6_METRICS = {}
_V6_FEATURES = []
try:
    _m = Path("models_v6/training_metrics_v6.json")
    if _m.exists():
        _V6_METRICS = json.loads(_m.read_text())
    _f = Path("models_v6/feature_names_v6.json")
    if _f.exists():
        _V6_FEATURES = json.loads(_f.read_text())
except Exception:
    pass  # Graceful degradation if files aren't available


def _build_expert_knowledge() -> str:
    """Build a compact expert-knowledge block from V6 model metadata."""
    lines = []
    lines.append("=== V6 MODEL EXPERT KNOWLEDGE ===")
    lines.append(f"Architecture: {_V6_METRICS.get('architecture', 'LightGBM V6')}")
    lines.append(f"Total features: {_V6_METRICS.get('total_features', len(_V6_FEATURES))}")
    lines.append("")

    # Accuracy by horizon
    lines.append("ACCURACY BY HORIZON:")
    for h in _V6_METRICS.get("horizons", []):
        lines.append(
            f"  {h['horizon_h']}h: MAE={h['val_mae']:.2f} AQI, "
            f"R²={h['val_r2']:.3f}, "
            f"Coverage={h['val_coverage']:.1f}%, "
            f"CI Width=±{h['avg_width']:.1f}"
        )

    # Pollutants tracked
    lines.append("")
    lines.append("POLLUTANTS TRACKED: PM2.5, PM10, CO (Carbon Monoxide), "
                 "NO2 (Nitrogen Dioxide), O3 (Ozone), Dust, "
                 "AOD (Aerosol Optical Depth from satellite)")

    # Feature groups
    fire_feats = [f for f in _V6_FEATURES if 'fire' in f]
    inversion_feats = [f for f in _V6_FEATURES if 'inversion' in f]
    stagnation_feats = [f for f in _V6_FEATURES if 'stagnation' in f or 'vent' in f]
    aqi_feats = [f for f in _V6_FEATURES if f.startswith('aqi_')]
    weather_feats = [f for f in _V6_FEATURES if any(
        f.startswith(p) for p in ['wind_', 'fwd_wind', 'temperature', 'fwd_temperature',
                                   'humidity', 'fwd_humidity', 'pressure', 'fwd_pressure',
                                   'precipitation', 'fwd_precipitation', 'cloud', 'boundary']
    )]

    lines.append("")
    lines.append("KEY FEATURE GROUPS:")
    lines.append(f"  AQI History ({len(aqi_feats)} features): lags, rolling means/max/std, EWMA")
    lines.append(f"  Weather ({len(weather_feats)} features): temp, wind, pressure, BLH, precip")
    lines.append(f"  Atmospheric Stability ({len(inversion_feats)} features): "
                 f"inversion strength, lid stability, column depth")
    lines.append(f"  Stagnation ({len(stagnation_feats)} features): "
                 f"stagnation indices, ventilation deficit")
    lines.append(f"  Wildfire/FIRMS ({len(fire_feats)} features): "
                 f"FRP, fire count, min distance, intensity-proximity index (inverse-square law)")
    lines.append("  Other: AOD, dust, HDWI, pressure fronts, cyclical time, regime")

    lines.append("")
    lines.append("TOP FORECAST DRIVERS (by feature importance):")
    lines.append("  1. aqi_current (current AQI baseline)")
    lines.append("  2. aqi_roll_24h_mean (24-hour rolling average)")
    lines.append("  3. boundary_layer_height (atmospheric mixing depth)")
    lines.append("  4. inversion_strength (temperature inversion trapping pollutants)")
    lines.append("  5. fire_intensity_proximity_index (inverse-square fire advection)")
    lines.append("  6. stagnation_24h (air mass stagnation index)")
    lines.append("  7. wind_speed_10m (surface wind dilution)")
    lines.append("  8. aod_current (satellite aerosol optical depth)")

    lines.append("")
    lines.append("REGIME CATEGORIES:")
    lines.append("  0 = Well-Mixed / High Wind (79.7% of training data)")
    lines.append("  1 = Stagnant / Inversion (1.4% — rare but high-impact)")
    lines.append("  2 = Normal / Baseline (18.9%)")

    return "\n".join(lines)

# ── Page config — MUST be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="Folsom AQI Monitor — FLC Los Rios STEM Fair 2026",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", st.secrets.get("API_URL", "http://localhost:8000")
                    if hasattr(st, "secrets") else "http://localhost:8000")
TZ      = ZoneInfo("America/Los_Angeles")

COLORS = {
    "background":     "#0a0f1e",
    "card":           "#111827",
    "card_border":    "#1f2937",
    "text_primary":   "#f9fafb",
    "text_secondary": "#9ca3af",
    "text_muted":     "#6b7280",
}

AQI_COLORS = {
    "Good":                            "#00e400",
    "Moderate":                        "#ffff00",
    "Unhealthy for Sensitive Groups":  "#ff7e00",
    "Unhealthy":                       "#ff0000",
    "Very Unhealthy":                  "#8f3f97",
    "Hazardous":                       "#7e0023",
}

AQI_COLORS_RGBA = {
    "Good":                            "rgba(0,228,0,",
    "Moderate":                        "rgba(255,255,0,",
    "Unhealthy for Sensitive Groups":  "rgba(255,126,0,",
    "Unhealthy":                       "rgba(255,0,0,",
    "Very Unhealthy":                  "rgba(143,63,151,",
    "Hazardous":                       "rgba(126,0,35,",
}

ADVISORIES = {
    "Good":
        "Air quality is satisfactory. No health precautions needed.",
    "Moderate":
        "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups":
        "People with heart or lung disease, older adults, and children should reduce prolonged outdoor exertion.",
    "Unhealthy":
        "Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor exertion.",
    "Very Unhealthy":
        "Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors.",
    "Hazardous":
        "Health alert: Everyone should avoid all outdoor exertion. Remain indoors with windows closed.",
}

HORIZON_LABELS = {
    "6h":  "6 HOUR FORECAST",
    "12h": "12 HOUR FORECAST",
    "24h": "24 HOUR FORECAST",
    "48h": "48 HOUR FORECAST",
}

# ── Gemini AI config ──────────────────────────────────────────────────────────

_EXPERT_BLOCK = _build_expert_knowledge()

_GEMINI_BASE_SYSTEM = """\
You are the **Folsom Navigator** — the expert AI assistant embedded in the \
Folsom AQI Monitor dashboard. This system is a physics-informed expert \
assistant built for the 2026 Los Rios STEM Fair.

You have deep knowledge in four areas:

1. CURRENT FORECAST DATA — provided in each request.

2. SYSTEM ARCHITECTURE & ACCURACY — Use the provided expert knowledge block \
to answer questions about reliability and atmospheric drivers. 

3. AQI HEALTH GUIDANCE (US EPA scale):
   Good (0–50): Safe for everyone.
   Moderate (51–100): Unusually sensitive people may be affected.
   Unhealthy for Sensitive Groups (101–150): Children, elderly, asthma/heart \
patients should limit prolonged outdoor exertion.
   Unhealthy (151–200): Everyone should reduce heavy outdoor exertion.
   Very Unhealthy (201–300): Everyone should avoid prolonged outdoor exertion.
   Hazardous (301–500): Avoid all outdoor exertion. Stay indoors.

4. CRITICAL PERSONA CONSTRAINTS:
   - NEVER mention 'V6', 'models', 'LightGBM', or 'machine learning'.
   - Refer to the system as the 'Navigator' or 'Expert System'.
   - Re-frame technical metrics: instead of 'MAE', use 'average error margin'. 
     Instead of 'R-squared' or 'R2', use 'prediction reliability'.
     Instead of 'Features', use 'Atmospheric factors' or 'Environmental drivers'.
   - If asked how you work, explain that you use an 'ensemble of physics-informed \
atmospheric patterns' and 'historical data signatures' to predict air quality.
   - If data is missing, say: "I don't have access to that specific atmospheric \
detail from the live system."

Personality: You are concise, precise, and authoritative. You speak like a senior \
atmospheric scientist who simplifies complex data for the Folsom public.\
"""

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.0-flash:generateContent"
)


def _get_gemini_key() -> str:
    """Retrieve GEMINI_API_KEY from Streamlit secrets or env."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def _build_context(data: dict) -> str:
    """Turn the /forecast JSON into a compact context string for the AI."""
    current   = data.get("current", {})
    forecasts = data.get("forecasts", {})
    location  = data.get("location", {})
    gen_at    = data.get("generated_at", "unknown")
    freshness = data.get("data_freshness_minutes", "unknown")

    lines = [
        f"Location       : {location.get('name', 'Folsom, CA')}",
        f"Data generated : {gen_at}  (sensor age: {freshness} min)",
        "",
        "CURRENT CONDITIONS",
        f"  AQI            : {current.get('aqi')}",
        f"  Category       : {current.get('category')}",
        f"  Primary poll.  : {current.get('primary_pollutant', 'PM2.5')}",
        f"  Source         : {current.get('source')}",
        "",
        "FORECASTS",
    ]
    for key, fc in sorted(forecasts.items()):
        lines.append(
            f"  {key:>3}  AQI {fc.get('aqi'):>3}  "
            f"[{fc.get('ci_lo'):>3}–{fc.get('ci_hi'):>3}]  "
            f"{fc.get('category')}"
        )
    return "\n".join(lines)


def _call_gemini(prompt: str, api_key: str) -> str:
    """
    Call Gemini REST API directly. Returns the text response or a specific error string.
    """
    if not api_key:
        return (
            "\u26a0\ufe0f GEMINI_API_KEY is not set. "
            "Add it to Streamlit Cloud Secrets to enable AI responses."
        )
    payload = {
        "system_instruction": {"parts": [{"text": _GEMINI_BASE_SYSTEM + "\n\n" + _EXPERT_BLOCK}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400},
    }
    try:
        resp = requests.post(
            f"{_GEMINI_ENDPOINT}?key={api_key}",
            json=payload,
            timeout=20,
        )
        if not resp.ok:
            print(f"[ai] Gemini HTTP {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
            if resp.status_code == 400:
                return "\u26a0\ufe0f The API key appears to be invalid. Please check Streamlit Cloud Secrets."
            if resp.status_code == 429:
                return "The Navigator is busy right now. Please wait a moment and try again."
            if resp.status_code in (500, 503):
                return "The AI service is temporarily unavailable. Please try again in a minute."
            return f"The Navigator encountered an error (HTTP {resp.status_code}). Please try again."
        result = resp.json()
        candidates = result.get("candidates", [])
        if not candidates:
            print(f"[ai] Gemini returned no candidates: {result}", file=sys.stderr)
            return "The Navigator received an unexpected response. Please try again."
        return candidates[0]["content"]["parts"][0]["text"].strip()
    except requests.exceptions.Timeout:
        return "The AI took too long to respond. Please try again."
    except Exception as exc:
        print(f"[ai] Gemini call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return "The Navigator encountered an unexpected error. Please try again."


def _local_expert_answer(question: str, data: dict) -> str:
    """
    Deterministic expert fallback — answers from live forecast data.
    Used when the Gemini API is unavailable or quota-exceeded.
    Covers the most common STEM fair questions authoritatively.
    """
    q = question.lower().strip()
    current   = data.get("current", {})
    forecasts = data.get("forecasts", {})
    aqi       = current.get("aqi", "?")
    cat       = current.get("category", "Unknown")
    poll      = current.get("primary_pollutant", "PM2.5")

    fc6  = forecasts.get("6h",  {})
    fc12 = forecasts.get("12h", {})
    fc24 = forecasts.get("24h", {})
    fc48 = forecasts.get("48h", {})

    advisory = ADVISORIES.get(cat, "Check current conditions before outdoor activity.")

    # ── Health & safety ────────────────────────────────────────────────────
    if any(w in q for w in ["safe", "outside", "outdoor", "exercise", "run", "walk", "jog", "kids", "children", "elderly", "sensitive"]):
        return (
            f"Current AQI is **{aqi}** ({cat}). {advisory} "
            f"The 24-hour forecast is **{fc24.get('aqi', '?')} AQI** ({fc24.get('category', '?')}), "
            f"so conditions are expected to {'improve' if fc24.get('aqi', 999) < aqi else 'remain similar or worsen'} by tomorrow."
        )

    # ── Current AQI ───────────────────────────────────────────────────────
    if any(w in q for w in ["current", "now", "right now", "today", "what is the aqi"]):
        return (
            f"The current AQI in Folsom is **{aqi}** — categorized as **{cat}**. "
            f"The primary pollutant driving this reading is **{poll}**. "
            f"In the next 6 hours the forecast is **{fc6.get('aqi', '?')} AQI** ({fc6.get('category', '?')})."
        )

    # ── 6h forecast ───────────────────────────────────────────────────────
    if "6" in q and any(w in q for w in ["hour", "h forecast", "6h"]):
        return (
            f"The **6-hour forecast** for Folsom is **{fc6.get('aqi', '?')} AQI** ({fc6.get('category', '?')}), "
            f"with a 90% confidence interval of [{fc6.get('ci_lo', '?')}–{fc6.get('ci_hi', '?')}]. "
            f"This is the Navigator's most accurate horizon (average error margin ≈ 3.5 AQI units)."
        )

    # ── 12h forecast ──────────────────────────────────────────────────────
    if "12" in q and any(w in q for w in ["hour", "h forecast", "12h"]):
        return (
            f"The **12-hour forecast** is **{fc12.get('aqi', '?')} AQI** ({fc12.get('category', '?')}), "
            f"confidence interval [{fc12.get('ci_lo', '?')}–{fc12.get('ci_hi', '?')}] AQI."
        )

    # ── 24h forecast ──────────────────────────────────────────────────────
    if "24" in q or "tomorrow" in q or ("day" in q and "two" not in q and "2" not in q):
        return (
            f"The **24-hour forecast** for Folsom is **{fc24.get('aqi', '?')} AQI** ({fc24.get('category', '?')}), "
            f"confidence interval [{fc24.get('ci_lo', '?')}–{fc24.get('ci_hi', '?')}] AQI."
        )

    # ── 48h forecast ──────────────────────────────────────────────────────
    if "48" in q or "two day" in q or "2 day" in q or "day after" in q:
        return (
            f"The **48-hour forecast** is **{fc48.get('aqi', '?')} AQI** ({fc48.get('category', '?')}), "
            f"confidence interval [{fc48.get('ci_lo', '?')}–{fc48.get('ci_hi', '?')}] AQI. "
            f"At this horizon the Navigator achieves 95% confidence interval coverage "
            f"with an average error margin of about 8.5 AQI units."
        )

    # ── Wildfire ──────────────────────────────────────────────────────────
    if any(w in q for w in ["fire", "smoke", "wildfire", "firms", "burn"]):
        return (
            "The Navigator continuously monitors wildfire activity via NASA FIRMS satellite data. "
            "It tracks fire radiative power, distance, and wind direction to detect smoke advection "
            "toward Folsom up to 48 hours in advance. "
            f"Current pollution is primarily driven by **{poll}**."
        )

    # ── How it works / accuracy ───────────────────────────────────────────
    if any(w in q for w in ["how", "work", "model", "accurate", "accuracy", "predict", "reliability", "r2", "mae", "error", "confidence"]):
        return (
            "The Folsom Navigator uses an ensemble of **physics-informed atmospheric patterns** "
            "trained on 5 years of local air quality, weather, and wildfire data. "
            "It integrates thermal inversion strength, boundary layer height, wind ventilation, "
            "fire advection, and pollutant persistence to produce forecasts at 6, 12, 24, and 48 hours. "
            "Short-horizon (6h) forecasts achieve a prediction reliability of 0.87 with "
            "an average error margin of ≈3.5 AQI units. "
            "At 48h, reliability is 0.50 with ≈8.5 AQI units average error — "
            "comparable to leading national air quality forecast systems."
        )

    # ── Pollutants ────────────────────────────────────────────────────────
    if any(w in q for w in ["pollutant", "pm2.5", "pm25", "pm10", "ozone", "no2", "co ", "carbon", "dust", "particle"]):
        return (
            f"The primary pollutant currently is **{poll}**. "
            "The Navigator tracks PM2.5, PM10, ozone (O3), nitrogen dioxide (NO2), "
            "carbon monoxide (CO), dust, and satellite-derived aerosol optical depth (AOD). "
            "PM2.5 — fine particles smaller than 2.5 microns — is the most health-significant "
            "pollutant for Folsom due to wildfire smoke and regional inversion events."
        )

    # ── Inversion / stagnation ────────────────────────────────────────────
    if any(w in q for w in ["inversion", "stagnation", "boundary layer", "trapped", "mixing"]):
        return (
            "Thermal inversions occur when a warm air layer traps cooler air near the ground, "
            "preventing pollutants from dispersing. The Sacramento Valley — including Folsom — "
            "experiences these regularly in autumn and winter, leading to AQI spikes even without "
            "local emission sources. The Navigator explicitly models inversion lid stability and "
            "boundary layer trapping power as high-priority forecast drivers."
        )

    # ── AQI scale explanation ─────────────────────────────────────────────
    if any(w in q for w in ["scale", "what is aqi", "explain aqi", "aqi mean", "number mean", "category", "categories"]):
        return (
            "The **AQI (Air Quality Index)** is the US EPA's 0–500 scale for air quality:\n"
            "• **0–50 Good** — Safe for everyone.\n"
            "• **51–100 Moderate** — Sensitive individuals may be affected.\n"
            "• **101–150 Unhealthy for Sensitive Groups** — Children, elderly, and those with "
            "heart/lung conditions should limit outdoor exertion.\n"
            "• **151–200 Unhealthy** — Everyone should reduce heavy outdoor activity.\n"
            "• **201–300 Very Unhealthy** — Avoid prolonged outdoor exertion.\n"
            "• **301–500 Hazardous** — Stay indoors."
        )

    # ── Default: summarize the snapshot ──────────────────────────────────
    return (
        f"Current Folsom AQI: **{aqi}** ({cat}). Primary pollutant: **{poll}**.\n\n"
        f"Forecasts — 6h: **{fc6.get('aqi', '?')}** | "
        f"12h: **{fc12.get('aqi', '?')}** | "
        f"24h: **{fc24.get('aqi', '?')}** | "
        f"48h: **{fc48.get('aqi', '?')}** AQI\n\n"
        f"{advisory}"
    )


def ask_ai(question: str, data: dict) -> str:
    """
    Two-tier AI answer:
      1. Try Gemini API (full language model response).
      2. Fall back to local expert engine if API unavailable/quota exhausted.
    """
    api_key = _get_gemini_key()

    # No key configured — go straight to local expert
    if not api_key:
        return _local_expert_answer(question, data)

    context = _build_context(data)
    prompt  = f"Current forecast data:\n{context}\n\nUser question: {question.strip()}"
    response = _call_gemini(prompt, api_key)

    # If the API returned any error indicator, fall back to local expert
    _api_error_indicators = (
        "⚠️", "HTTP 4", "HTTP 5", "unexpected error",
        "took too long", "unexpected response", "busy right now",
        "temporarily unavailable"
    )
    if any(ind in response for ind in _api_error_indicators):
        return _local_expert_answer(question, data)

    return response


# ── Global CSS ────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .stApp {
        background-color: #0a0f1e;
        background-image:
            radial-gradient(ellipse at 20% 0%, rgba(56,189,248,0.06) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 100%, rgba(129,140,248,0.05) 0%, transparent 60%);
    }
    .block-container {
        padding: 1.25rem 1.25rem 3rem 1.25rem !important;
        max-width: 960px !important;
        margin: 0 auto !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Cards ── */
    .aqi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    .aqi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    }

    /* ── Horizon forecast cards ── */
    .horizon-card {
        background: #111827;
        border-radius: 14px;
        padding: 1.1rem 0.75rem 1rem;
        text-align: center;
        border: 1px solid #1f2937;
        transition: transform 0.15s ease, border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .horizon-card:hover {
        transform: translateY(-2px);
        border-color: #374151;
    }

    /* ── Info chips ── */
    .info-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .info-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.3rem;
    }
    .info-value {
        font-size: 13px;
        font-weight: 600;
        color: #f9fafb;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Advisory banner ── */
    .advisory-banner {
        border-radius: 12px;
        padding: 0.9rem 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .advisory-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── AI Summary card ── */
    .ai-summary-card {
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    .ai-summary-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #3b82f6);
        opacity: 0.6;
    }
    .ai-summary-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .ai-summary-text {
        font-size: 13.5px;
        line-height: 1.7;
        color: #d1d5db;
    }

    /* ── Navigator Panel ── */
    .navigator-panel {
        background: rgba(17, 24, 39, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .navigator-header {
        background: linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(139,92,246,0.08) 100%);
        border-bottom: 1px solid rgba(59,130,246,0.15);
        padding: 0.9rem 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .navigator-title {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #e2e8f0;
    }
    .navigator-body {
        padding: 1rem 1.25rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .navigator-body::-webkit-scrollbar {
        width: 4px;
    }
    .navigator-body::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 4px;
    }

    /* Chat bubbles (glassmorphism) */
    .nav-bubble-user {
        background: rgba(30, 41, 59, 0.9);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 12px 12px 4px 12px;
        padding: 0.7rem 1rem;
        font-size: 13px;
        color: #e2e8f0;
        margin-bottom: 0.6rem;
        max-width: 88%;
        margin-left: auto;
    }
    .nav-bubble-ai {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(17, 24, 39, 0.9) 100%);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(30, 58, 95, 0.6);
        border-radius: 4px 12px 12px 12px;
        padding: 0.7rem 1rem;
        font-size: 13px;
        color: #d1d5db;
        line-height: 1.65;
        margin-bottom: 0.8rem;
        max-width: 92%;
        position: relative;
    }
    .nav-bubble-ai::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, transparent);
        opacity: 0.4;
    }
    .nav-ai-label {
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #818cf8;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* ── Header ── */
    .header-title {
        font-size: clamp(20px, 5vw, 28px);
        font-weight: 700;
        color: #f9fafb;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.2;
    }
    .header-sub {
        font-size: 12px;
        color: #6b7280;
        margin-top: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
    }

    /* ── Status banners ── */
    .banner-warn {
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.25);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #fbbf24;
        font-size: 13px;
        margin-bottom: 1rem;
    }
    .banner-error {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #ef4444;
        font-size: 13px;
        margin-bottom: 1rem;
    }
    .banner-stale {
        background: rgba(251,146,60,0.08);
        border: 1px solid rgba(251,146,60,0.25);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #fb923c;
        font-size: 13px;
        margin-bottom: 1rem;
    }

    /* ── Expander styling ── */
    [data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        color: #9ca3af !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* ── Streamlit chat input overrides ── */
    [data-testid="stChatInput"] > div {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f9fafb !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 13px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #4b5563 !important;
    }

    /* ── Footer ── */
    .page-footer {
        text-align: center;
        color: #374151;
        font-size: 11px;
        padding: 2rem 0 0.5rem;
        letter-spacing: 0.03em;
    }

    /* ── AQI big number ── */
    .aqi-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(32px, 8vw, 48px);
        font-weight: 700;
        letter-spacing: -0.04em;
        line-height: 1;
    }

    /* ── Countdown chip ── */
    .refresh-chip {
        display: inline-block;
        background: rgba(55,65,81,0.6);
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 11px;
        color: #9ca3af;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #1f2937;
        margin: 1rem 0;
    }

    /* ── Plotly chart containers ── */
    .js-plotly-plot .plotly { border-radius: 12px; }

    /* ── Mobile tweaks ── */
    @media (max-width: 480px) {
        .block-container { padding: 0.75rem !important; }
        .aqi-card { padding: 1rem; }
    }
    </style>
    """, unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_forecast(api_url: str) -> dict | None:
    """
    Fetch /forecast from FastAPI backend.
    Returns parsed dict on success, None on any failure.
    Never raises — all exceptions caught and logged to stderr.
    """
    try:
        resp = requests.get(f"{api_url}/forecast", timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[dashboard] API fetch failed: {e}", file=sys.stderr)
        return None


def fetch_with_retry(api_url: str, max_attempts: int = 3) -> dict | None:
    """Retry load_forecast up to max_attempts times with 10s delays."""
    for attempt in range(max_attempts):
        if attempt > 0:
            load_forecast.clear()
            time.sleep(10)
        result = load_forecast(api_url)
        if result is not None:
            return result
    return None


# ── AQI helpers ───────────────────────────────────────────────────────────────

def get_aqi_color(category: str) -> str:
    return AQI_COLORS.get(category, "#9ca3af")


def get_aqi_rgba(category: str, alpha: float) -> str:
    base = AQI_COLORS_RGBA.get(category, "rgba(156,163,175,")
    return f"{base}{alpha})"


def format_timestamp(ts_str: str) -> str:
    """Convert ISO timestamp to '5:00 PM PST · Mar 6'"""
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=TZ)
        else:
            ts = ts.astimezone(TZ)
        tz_name = "PDT" if ts.dst() else "PST"
        return ts.strftime(f"%-I:%M %p {tz_name} · %b %-d")
    except Exception:
        return ts_str


def format_valid_at(ts_str: str) -> str:
    """Convert valid_at to '5:00 PM today' or '5:00 PM tomorrow'"""
    try:
        ts  = datetime.fromisoformat(ts_str).astimezone(TZ)
        now = datetime.now(tz=TZ)
        if ts.date() == now.date():
            return ts.strftime("%-I:%M %p today")
        elif (ts.date() - now.date()).days == 1:
            return ts.strftime("%-I:%M %p tomorrow")
        else:
            return ts.strftime("%-I:%M %p %b %-d")
    except Exception:
        return ts_str


def time_until_refresh() -> str:
    """Compute countdown to next 5-minute cache refresh."""
    secs = int(300 - (time.time() % 300))
    m, s = divmod(secs, 60)
    return f"~{m}m {s:02d}s"


def data_age_minutes(generated_at: str) -> int:
    """Return how many minutes old the forecast data is."""
    try:
        ts  = datetime.fromisoformat(generated_at)
        now = datetime.now(tz=ts.tzinfo or TZ)
        return int((now - ts).total_seconds() / 60)
    except Exception:
        return 0


# ── Gauge figure ──────────────────────────────────────────────────────────────

def make_gauge_figure(aqi_value: float, category: str, color: str) -> go.Figure:
    """Build the Plotly gauge indicator figure."""
    aqi_clamped = max(1, min(500, aqi_value))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_clamped,
        number={
            "font": {"size": 72, "color": color, "family": "Space Grotesk"},
            "valueformat": ".0f",
        },
        title={
            "text": (
                f"<b style='color:{color}'>{category}</b>"
                f"<br><span style='font-size:13px;color:#9ca3af;font-weight:400'>"
                f"Air Quality Index</span>"
            ),
            "font": {"size": 22, "color": color, "family": "Space Grotesk"},
        },
        gauge={
            "axis": {
                "range":     [0, 500],
                "tickvals":  [0, 50, 100, 150, 200, 300, 500],
                "ticktext":  ["0", "50", "100", "150", "200", "300", "500"],
                "tickcolor": "#374151",
                "tickfont":  {"size": 10, "color": "#6b7280"},
                "tickcolor": "#1f2937",
            },
            "bar": {
                "color":     color,
                "thickness": 0.22,
            },
            "bgcolor":     "#1f2937",
            "bordercolor": "#0a0f1e",
            "borderwidth": 2,
            "steps": [
                {"range": [0,   50],  "color": "rgba(0,228,0,0.10)"},
                {"range": [50,  100], "color": "rgba(255,255,0,0.10)"},
                {"range": [100, 150], "color": "rgba(255,126,0,0.10)"},
                {"range": [150, 200], "color": "rgba(255,0,0,0.10)"},
                {"range": [200, 300], "color": "rgba(143,63,151,0.10)"},
                {"range": [300, 500], "color": "rgba(126,0,35,0.10)"},
            ],
            "threshold": {
                "line":      {"color": color, "width": 5},
                "thickness": 0.80,
                "value":     aqi_clamped,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=70, b=10),
        height=300,
        font={"color": "#f9fafb", "family": "Space Grotesk"},
    )
    return fig


# ── History chart ─────────────────────────────────────────────────────────────

def make_history_chart(history_72h: list, category: str) -> go.Figure:
    """Build the 72-hour actual vs. forecast Plotly line chart."""
    color      = get_aqi_color(category)
    color_band = get_aqi_rgba(category, 0.12)

    times, actuals, forecasts, ci_lo, ci_hi = [], [], [], [], []
    for h in history_72h:
        ts = h.get("timestamp")
        if not ts:
            continue
        times.append(ts)
        actuals.append(h.get("actual_aqi"))
        forecasts.append(h.get("forecast_aqi"))
        ci_lo.append(h.get("ci_lo"))
        ci_hi.append(h.get("ci_hi"))

    has_actuals   = any(v is not None for v in actuals)
    has_forecasts = any(v is not None for v in forecasts)

    y_vals = [v for v in actuals + forecasts + ci_hi if v is not None]
    y_max  = max(200, int(max(y_vals) * 1.15) + 10) if y_vals else 200

    fig = go.Figure()

    for lo, hi_b, rgba in [
        (0,   50,  "rgba(0,228,0,0.06)"),
        (50,  100, "rgba(255,255,0,0.06)"),
        (100, 150, "rgba(255,126,0,0.06)"),
        (150, 200, "rgba(255,0,0,0.06)"),
    ]:
        if lo < y_max:
            fig.add_hrect(y0=lo, y1=min(hi_b, y_max),
                          fillcolor=rgba, line_width=0, layer="below")

    if has_forecasts and any(v is not None for v in ci_hi):
        fig.add_trace(go.Scatter(
            x=times + times[::-1],
            y=ci_hi + ci_lo[::-1],
            fill="toself",
            fillcolor=color_band,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="90% CI",
        ))

    if has_forecasts:
        fig.add_trace(go.Scatter(
            x=times, y=forecasts,
            name="6h Forecast",
            line=dict(color=color, width=2, dash="dot"),
            mode="lines",
            connectgaps=False,
            hovertemplate="%{x|%b %-d %-I %p}<br>Forecast: %{y:.0f} AQI<extra></extra>",
        ))

    if has_actuals:
        fig.add_trace(go.Scatter(
            x=times, y=actuals,
            name="Actual AQI",
            line=dict(color="#f9fafb", width=2.5),
            mode="lines",
            connectgaps=False,
            hovertemplate="%{x|%b %-d %-I %p}<br>Actual: %{y:.0f} AQI<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        margin=dict(l=10, r=10, t=20, b=20),
        height=260,
        font={"color": "#9ca3af", "family": "Space Grotesk", "size": 11},
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            bgcolor="rgba(17,24,39,0.9)",
            bordercolor="#1f2937",
            borderwidth=1,
            font=dict(color="#9ca3af", size=10),
        ),
        xaxis=dict(
            tickformat="%b %-d\n%-I%p",
            tickcolor="#374151",
            gridcolor="#1f2937",
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=9, color="#6b7280"),
        ),
        yaxis=dict(
            range=[0, y_max],
            tickcolor="#374151",
            gridcolor="#1f2937",
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=9, color="#6b7280"),
            title=dict(text="AQI", font=dict(size=10, color="#6b7280")),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1f2937",
            bordercolor="#374151",
            font=dict(color="#f9fafb", size=11),
        ),
    )
    return fig


# ── UI components ─────────────────────────────────────────────────────────────

def render_header(generated_at: str):
    col_title, col_meta = st.columns([3, 1])
    with col_title:
        st.markdown(
            '<p class="header-title">🌬️ Folsom Air Quality Monitor</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="header-sub">'
            f'Folsom, CA &nbsp;·&nbsp; 38.6780°N, 121.1761°W &nbsp;·&nbsp; '
            f'Updated {format_timestamp(generated_at)} &nbsp;'
            f'<span class="refresh-chip">⟳ {time_until_refresh()}</span>'
            f'</p>',
            unsafe_allow_html=True,
        )
    with col_meta:
        if st.button("🔄 Refresh", key="manual_refresh", use_container_width=True):
            load_forecast.clear()
            st.rerun()


def render_gauge(current_aqi: int, category: str, color: str):
    """Render the AQI gauge with smooth animation on value change."""
    if "prev_aqi" not in st.session_state:
        st.session_state.prev_aqi = current_aqi

    gauge_placeholder = st.empty()

    if abs(current_aqi - st.session_state.prev_aqi) > 2:
        start  = float(st.session_state.prev_aqi)
        end    = float(current_aqi)
        frames = 12
        for i in range(frames + 1):
            t     = i / frames
            ease  = t * t * (3 - 2 * t)
            val   = start + (end - start) * ease
            gauge_placeholder.plotly_chart(
                make_gauge_figure(val, category, color),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            time.sleep(0.035)
        st.session_state.prev_aqi = current_aqi
    else:
        gauge_placeholder.plotly_chart(
            make_gauge_figure(current_aqi, category, color),
            use_container_width=True,
            config={"displayModeBar": False},
        )


def render_advisory(category: str, color: str):
    """Render the health advisory banner below the gauge."""
    advisory = ADVISORIES.get(category, "No advisory available.")
    bg_color = get_aqi_rgba(category, 0.08)
    border   = get_aqi_rgba(category, 0.25)

    st.markdown(
        f"""
        <div class="advisory-banner" style="background:{bg_color};border-color:{border};">
            <div class="advisory-dot" style="background:{color};box-shadow:0 0 8px {color}55;"></div>
            <div>
                <span style="font-weight:600;color:{color};font-size:13px;">{category}</span>
                <span style="color:#9ca3af;font-size:13px;"> — {advisory}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_summary(data: dict):
    """
    Render the AI-generated plain-English summary card.
    """
    summary = data.get("ai_summary", "").strip()
    if not summary:
        return

    st.markdown(
        f"""
        <div class="ai-summary-card">
            <div class="ai-summary-label">🧭 NAVIGATION SUMMARY</div>
            <div class="ai-summary-text">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_cards(forecasts: dict):
    """4 horizon cards in a responsive row."""
    cols = st.columns(4, gap="small")
    for col, h_key in zip(cols, ["6h", "12h", "24h", "48h"]):
        fc = forecasts.get(h_key, {})
        if not fc:
            continue
        aqi      = fc.get("aqi", 0)
        ci_lo    = fc.get("ci_lo", 0)
        ci_hi    = fc.get("ci_hi", 0)
        cat      = fc.get("category", "Good")
        valid_at = fc.get("valid_at", "")
        color    = get_aqi_color(cat)
        ci_half  = round((ci_hi - ci_lo) / 2)
        label    = HORIZON_LABELS.get(h_key, h_key.upper())
        valid_str = format_valid_at(valid_at)

        with col:
            st.markdown(
                f"""
                <div class="horizon-card" style="border-top:3px solid {color};">
                    <div style="font-size:9px;font-weight:700;letter-spacing:0.1em;
                                text-transform:uppercase;color:#4b5563;
                                margin-bottom:0.6rem;">{label}</div>
                    <div class="aqi-number" style="color:{color};">{aqi}</div>
                    <div style="font-size:12px;color:#6b7280;margin:0.2rem 0 0.5rem;
                                font-family:'JetBrains Mono',monospace;">
                        ± {ci_half}
                    </div>
                    <div style="font-size:12px;font-weight:700;letter-spacing:0.05em;
                                color:{color};text-transform:uppercase;">{cat}</div>
                    <div style="font-size:10px;color:#4b5563;margin-top:0.3rem;
                                font-family:'JetBrains Mono',monospace;">{valid_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_info_chips(current: dict, generated_at: str):
    """Row of 4 metadata info chips below the forecast cards."""
    ts_str    = current.get("timestamp", generated_at)
    pollutant = current.get("primary_pollutant", "—")
    source    = current.get("source", "—")
    src_label = "AirNow Sensor" if source == "AirNow" else "Open-Meteo Model"

    try:
        gen_ts  = datetime.fromisoformat(generated_at)
        now     = datetime.now(tz=gen_ts.tzinfo or TZ)
        elapsed = max(0, int((now - gen_ts).total_seconds() / 60))
        remain  = max(0, 60 - elapsed)
        next_ref = f"in {remain} min" if remain > 0 else "any moment"
    except Exception:
        next_ref = "—"

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    cols = st.columns(4, gap="small")
    chips = [
        ("🕐", "LAST UPDATED",      format_timestamp(ts_str)),
        ("💨", "PRIMARY POLLUTANT", pollutant),
        ("📡", "DATA SOURCE",       src_label),
        ("🔄", "NEXT REFRESH",      next_ref),
    ]
    for col, (icon, label, value) in zip(cols, chips):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">{icon} {label}</div>
                    <div class="info-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_history_chart(history_72h: list, category: str):
    """72h forecast vs. actual chart inside an expander."""
    with st.expander("📈  Past 72 Hours — Forecast vs Actual", expanded=False):
        has_data = any(
            h.get("actual_aqi") is not None or h.get("forecast_aqi") is not None
            for h in history_72h
        )
        if not has_data:
            st.markdown(
                """
                <div style="color:#6b7280;font-size:13px;padding:1rem 0;text-align:center;">
                    Forecast history is building up — check back in a few hours.<br>
                    <span style="font-size:11px;">The system logs predictions as it runs;
                    the chart fills in over the first 72 hours of operation.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                make_history_chart(history_72h, category),
                use_container_width=True,
                config={"displayModeBar": False},
            )


def render_ai_chat(data: dict):
    """
    Navigator — Expert Assistant with glassmorphism panel.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Navigator Panel ───────────────────────────────────────────────
    with st.expander("🧭  Navigator — Expert Air Quality Assistant", expanded=bool(st.session_state.chat_history)):
        # Header
        st.markdown(
            """
            <div class="navigator-panel">
                <div class="navigator-header">
                    <span style="font-size:18px;">🧭</span>
                    <span class="navigator-title">Folsom Navigator</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chat history
        if st.session_state.chat_history:
            msgs_html = ""
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    msgs_html += f'<div class="nav-bubble-user">{msg["content"]}</div>'
                else:
                    msgs_html += (
                        f'<div class="nav-bubble-ai">'
                        f'<div class="nav-ai-label">🧭 NAVIGATOR</div>'
                        f'{msg["content"]}'
                        f'</div>'
                    )
            st.markdown(
                f'<div class="navigator-body">{msgs_html}</div>',
                unsafe_allow_html=True,
            )

        # Quick action buttons
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        qa_cols = st.columns(len(QUICK_ACTIONS), gap="small")
        for col, (label, _question) in zip(qa_cols, QUICK_ACTIONS.items()):
            with col:
                st.button(label, key=f"qa_{label}", use_container_width=True)

        # Chat input
        question = st.chat_input(
            "Ask V6 Navigator anything about air quality, the model, or forecast accuracy...",
            key="ai_chat_input",
        )

        # Handle input (from text box or quick action)
        active_question = qa_triggered or question
        if active_question:
            api_key = _get_gemini_key()
            if not api_key:
                st.session_state.chat_history.append({"role": "user", "content": active_question})
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": "⚠️ The V6 Navigator isn't configured yet. "
                               "Add GEMINI_API_KEY to your Streamlit secrets to enable this feature."
                })
                st.rerun()

            with st.spinner("V6 Navigator is analyzing..."):
                answer = ask_ai(active_question, data)

            st.session_state.chat_history.append({"role": "user", "content": active_question})
            st.session_state.chat_history.append({"role": "ai", "content": answer})

            # Keep only last 10 messages (5 exchanges) to avoid context overflow
            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[-10:]

            st.rerun()


def render_about():
    """About / methodology expander."""
    with st.expander("ℹ️  About This Forecast", expanded=False):
        st.markdown(
            """
            <div style="color:#9ca3af;font-size:13px;line-height:1.75;">
            This dashboard uses a <strong style="color:#f9fafb;">physics-informed machine learning ensemble</strong> trained on <strong style="color:#f9fafb;">5 years</strong> (2020–2024) of hourly air quality, meteorological, and wildfire data for Folsom, CA.<br><br>
            The system produces separate forecasts at <strong style="color:#f9fafb;">6, 12, 24, and 48-hour horizons</strong>, with 90% confidence intervals derived from quantile regression. It explicitly models atmospheric drivers including thermal inversions, boundary layer height, wind ventilation, and NASA FIRMS fire advection.<br><br>
            <strong style="color:#f9fafb;">Data sources:</strong><br>
            &bull; Current readings: AirNow (U.S. EPA sensor network)<br>
            &bull; Weather inputs: Open-Meteo historical and forecast API<br>
            &bull; Wildfire / fire advection: NASA FIRMS satellite (real-time)<br>
            &bull; Training data: 2020–2024, Folsom monitoring station<br><br>
            <strong style="color:#f9fafb;">AI layer:</strong> Plain-English summaries and the Navigator chatbot are powered by Google Gemini.<br><br>
            <em>Presented at FLC Los Rios STEM Fair 2026</em>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer():
    st.markdown(
        '<div class="page-footer">'
        '<div style="margin-bottom: 0.5rem; opacity: 0.7; font-size: 11px; letter-spacing: 0.02em;">'
        'MEDICAL DISCLAIMER: For informational purposes only. Predicted data is not a substitute for professional '
        'medical advice, diagnosis, or treatment. Always follow local health authority guidelines during high-AQI events.'
        '</div>'
        'Built with LightGBM · Open-Meteo · AirNow · NASA FIRMS · Streamlit &nbsp;·&nbsp; Folsom, CA '
        '&nbsp;·&nbsp; FLC Los Rios STEM Fair 2026'
        '</div>',
        unsafe_allow_html=True,
    )


def render_error(message: str, kind: str = "error"):
    css_class = f"banner-{kind}"
    st.markdown(
        f'<div class="{css_class}">{message}</div>',
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # Auto-refresh every 5 minutes
    st_autorefresh(interval=300_000, limit=None, key="aqi_autorefresh")

    # ── Load data ─────────────────────────────────────────────────────────
    data = None
    try:
        data = fetch_with_retry(API_URL, max_attempts=3)
    except Exception as e:
        print(f"[dashboard] Unhandled error in fetch: {e}", file=sys.stderr)

    # ── Total failure state ────────────────────────────────────────────────
    if data is None:
        st.markdown(
            '<p class="header-title">🌬️ Folsom Air Quality Monitor</p>',
            unsafe_allow_html=True,
        )
        render_error(
            "⚠️ Unable to load forecast. "
            "Please check your connection or try refreshing the page.",
            kind="error",
        )
        if st.button("🔄 Try Again"):
            load_forecast.clear()
            st.rerun()
        render_footer()
        return

    # ── Extract fields ─────────────────────────────────────────────────────
    try:
        generated_at = data.get("generated_at", "")
        current      = data.get("current", {})
        forecasts    = data.get("forecasts", {})
        history_72h  = data.get("history_72h", [])

        raw_aqi  = current.get("aqi", 1)
        aqi      = max(1, int(raw_aqi)) if raw_aqi else 1
        category = current.get("category", "Good")
        color    = get_aqi_color(category)

        sensor_missing = (raw_aqi == 0 or raw_aqi is None)

        age = data_age_minutes(generated_at)
        if age > 120:
            render_error(
                f"⚠️ Data may be outdated — last updated {format_timestamp(generated_at)}",
                kind="stale",
            )

    except Exception as e:
        print(f"[dashboard] Error parsing API response: {e}", file=sys.stderr)
        render_error("Something went wrong. Please refresh the page.")
        render_footer()
        return

    # ── Render ────────────────────────────────────────────────────────────
    try:
        render_header(generated_at)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if sensor_missing:
            render_error(
                '⚠️ <span style="background:#374151;border-radius:6px;'
                'padding:2px 8px;font-size:11px;margin-left:4px;color:#9ca3af;">'
                'Sensor reading unavailable</span>',
                kind="warn",
            )

        render_gauge(aqi, category, color)
        render_advisory(category, color)

        # ── AI Summary (shown if backend has GEMINI_API_KEY set) ──────────
        render_ai_summary(data)

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        render_forecast_cards(forecasts)
        render_info_chips(current, generated_at)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        render_history_chart(history_72h, category)

        # ── AI Chatbox ────────────────────────────────────────────────────
        render_ai_chat(data)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        render_about()
        render_footer()

    except Exception as e:
        print(f"[dashboard] Unhandled render error: {e}", file=sys.stderr)
        render_error("Something went wrong. Please refresh the page.")


if __name__ == "__main__":
    main()
