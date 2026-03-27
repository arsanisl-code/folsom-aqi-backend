"""
ai_layer.py — Gemini 2.0 Flash integration for Folsom AQI Forecast.

Two public functions:
    generate_summary(forecast_data)  →  one-paragraph plain-English summary
    answer_question(question, forecast_data)  →  single-turn chat answer

Both functions fail gracefully: they return "" / an error string on any
API failure so the rest of the app stays up.

Requires env var: GEMINI_API_KEY
Install:          pip install google-generativeai
"""

import os
import sys

import google.generativeai as genai

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"

# How the AI presents itself and what it knows
_SYSTEM_PROMPT = """\
You are an AI assistant embedded in the Folsom AQI Forecast dashboard — a \
machine learning project built by a freshman computer engineering student at \
Folsom Lake College (MESA Program Scholar, Phi Theta Kappa) for the 2026 \
Los Rios STEM Fair.

You have knowledge in three areas:

1. CURRENT FORECAST DATA — provided to you in each request.

2. MODEL ARCHITECTURE — The system uses LightGBM ensemble models with four \
forecast horizons (6h, 12h, 24h, 48h). Features include AQI lags, PM2.5 \
lags, boundary layer height, wind speed, aerosol optical depth (satellite \
smoke detection), wildfire proxy features (Hot-Dry-Windy Index, Vapor \
Pressure Deficit, antecedent precipitation deficit), pressure front \
differencing, and cyclical time encodings. Models use Huber loss for \
robustness to wildfire smoke spikes. Quantile models (1st and 99th \
percentile) provide confidence intervals. Training data spans 2022–present \
from Open-Meteo and AirNow sensor networks. Walk-forward validation over the \
last 30 days is used to generate honest accuracy estimates.

3. AQI HEALTH GUIDANCE (US EPA scale):
   • Good (0–50): Air quality is satisfactory. Safe for everyone.
   • Moderate (51–100): Acceptable. Unusually sensitive people should \
consider reducing prolonged outdoor exertion.
   • Unhealthy for Sensitive Groups (101–150): Children, elderly, and people \
with asthma or heart/lung disease should limit prolonged outdoor exertion. \
Others are fine.
   • Unhealthy (151–200): Everyone should reduce prolonged or heavy outdoor \
exertion. Sensitive groups should avoid it.
   • Very Unhealthy (201–300): Everyone should avoid prolonged outdoor \
exertion. Stay indoors where possible.
   • Hazardous (301–500): Emergency conditions. Everyone should avoid all \
outdoor exertion and remain indoors with windows closed.

Keep answers concise, accurate, and friendly. \
If a question is completely unrelated to air quality, environmental science, \
or this project, politely acknowledge it and redirect the user back to those \
topics in one sentence.\
"""


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_model() -> genai.GenerativeModel:
    """Configure Gemini and return a model instance. Raises if key is missing."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_SYSTEM_PROMPT,
    )


def _build_context_block(forecast_data: dict) -> str:
    """
    Convert the /forecast JSON into a compact, readable context string
    that the AI can reason over.
    """
    current   = forecast_data.get("current", {})
    forecasts = forecast_data.get("forecasts", {})
    location  = forecast_data.get("location", {})
    gen_at    = forecast_data.get("generated_at", "unknown")
    freshness = forecast_data.get("data_freshness_minutes", "unknown")

    lines = [
        f"Location       : {location.get('name', 'Folsom, CA')}",
        f"Data generated : {gen_at}  (sensor age: {freshness} min)",
        "",
        "──── CURRENT CONDITIONS ────",
        f"AQI            : {current.get('aqi')}",
        f"Category       : {current.get('category')}",
        f"Primary poll.  : {current.get('primary_pollutant', 'PM2.5')}",
        f"Source         : {current.get('source')}",
        "",
        "──── FORECASTS ────",
    ]

    for key, fc in sorted(forecasts.items()):
        lines.append(
            f"  {key:>3}  AQI {fc.get('aqi'):>3}  "
            f"[{fc.get('ci_lo'):>3} – {fc.get('ci_hi'):>3}]  "
            f"{fc.get('category')}  "
            f"(valid at {fc.get('valid_at', '')})"
        )

    return "\n".join(lines)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_summary(forecast_data: dict) -> str:
    """
    Generate a plain-English one-paragraph summary of the current AQI forecast.
    Called once per hourly refresh cycle in inference.py and cached in latest.json.

    Returns "" on any failure — the frontend should hide the summary section
    when this field is empty rather than showing an error.
    """
    try:
        model   = _get_model()
        context = _build_context_block(forecast_data)

        prompt = (
            "Using the forecast data below, write a single plain-English paragraph "
            "(3–4 sentences) summarizing current air quality in Folsom for local residents. "
            "Include: what the current AQI is and what it means in everyday terms, "
            "whether conditions are expected to improve or worsen over the next 48 hours, "
            "and one practical recommendation (e.g. whether outdoor activity is advisable). "
            "Write for the general public — no jargon, no bullet points, no markdown.\n\n"
            f"{context}"
        )

        response = model.generate_content(prompt)
        summary  = response.text.strip()
        print(f"[ai_layer] Summary generated ({len(summary)} chars)", file=sys.stderr)
        return summary

    except Exception as exc:
        print(f"[ai_layer] generate_summary failed: {exc}", file=sys.stderr)
        return ""


def answer_question(question: str, forecast_data: dict) -> str:
    """
    Answer a single user question using current forecast data + model knowledge.
    Single-turn — no conversation history is kept between calls.

    Returns a user-facing error string on failure (never crashes).
    """
    if not question or not question.strip():
        return "Please type a question and I'll do my best to answer it."

    try:
        model   = _get_model()
        context = _build_context_block(forecast_data)

        prompt = (
            f"Current forecast data for Folsom, CA:\n{context}\n\n"
            f"User question: {question.strip()}"
        )

        response = model.generate_content(prompt)
        answer   = response.text.strip()
        print(f"[ai_layer] Question answered ({len(answer)} chars)", file=sys.stderr)
        return answer

    except Exception as exc:
        print(f"[ai_layer] answer_question failed: {exc}", file=sys.stderr)
        return (
            "Sorry, I couldn't process that question right now. "
            "Please check that the GEMINI_API_KEY is set and try again."
        )
