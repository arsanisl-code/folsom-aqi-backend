"""
ai_layer.py — Gemini integration for the Folsom AQI Forecast system.

Public API:
    generate_summary(forecast_data)
        → one-paragraph plain-English summary, cached in latest.json.
    answer_question(question, forecast_data)
        → single-turn Q&A using the google-generativeai SDK + GEMINI_API_KEY env var.
    answer_question_with_key(question, forecast_data, api_key)
        → same Q&A via direct REST call, accepting the key as a parameter.
          Used by the Streamlit frontend so it can pass a key sourced from
          st.secrets without this module needing to know about Streamlit.

All functions fail gracefully: they return "" or a user-facing error string
on any API failure so the rest of the system stays up.
"""

import os

import requests
import google.generativeai as genai

from logger import get_logger

log = get_logger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_MODEL    = "gemini-2.5-flash-lite"
_GEMINI_REST_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.0-flash:generateContent"
)

# How the AI presents itself and what it knows
_SYSTEM_PROMPT = """\
You are the **Folsom Navigator** — the expert AI assistant embedded in the \
Folsom AQI Monitor dashboard. This system is a physics-informed expert \
assistant built for environmental monitoring.

You have deep knowledge in three areas:

1. CURRENT FORECAST DATA — provided to you in each request.

2. SYSTEM ARCHITECTURE & ACCURACY — The system uses an ensemble of \
atmospheric patterns and historical physics signatures (PM2.5 logs, \
boundary layer depth, wind dilution, and wildfire advection proxies). \
Confidence intervals are derived from atmospheric uncertainty. Accuracy is \
highest in the 6h–12h windows and degrades as the forecast horizon extends \
to 48h due to NWP precision loss.

3. AQI HEALTH GUIDANCE (US EPA scale):
   • Good (0–50): Air quality is satisfactory. Safe for everyone.
   • Moderate (51–100): Acceptable. Unusually sensitive people should \
consider reducing prolonged outdoor exertion.
   • Unhealthy for Sensitive Groups (101–150): Children, elderly, and people \
with asthma or heart/lung disease should limit prolonged outdoor exertion.
   • Unhealthy (151–200): Everyone should reduce prolonged or heavy outdoor \
exertion. Sensitive groups should avoid it.
   • Very Unhealthy (201–300): Everyone should avoid prolonged outdoor \
exertion. Stay indoors where possible.
   • Hazardous (301–500): Emergency conditions. Everyone should avoid all \
outdoor exertion and remain indoors.

CRITICAL PERSONA CONSTRAINTS:
- NEVER mention 'V6', 'models'.
- NEVER mention any developer names, college affiliations, or STEM fairs.
- Refer to the system as the 'Navigator' or 'Expert System'.
- If asked how you work, explain that you use an 'ensemble of physics-informed \
atmospheric patterns' to predict air quality.
- Keep answers concise, authoritative, and friendly. Speak like a senior \
atmospheric scientist who simplifies complex data for the Folsom public.
"""


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_model() -> genai.GenerativeModel:
    """Configure Gemini SDK and return a model instance. Raises if key is missing."""
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


def _format_forecast_as_ai_context(forecast_data: dict) -> str:
    """
    Format the /forecast JSON payload into a compact, readable context string
    that the AI can reason over. Renamed from _build_context_block to express
    that this function formats data for AI prompt injection, not general display.
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

    Returns "" on any failure — the frontend hides the summary section when
    this field is empty rather than showing an error.
    """
    try:
        model   = _get_model()
        context = _format_forecast_as_ai_context(forecast_data)
        prompt  = (
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
        log.info("Summary generated (%s chars)", len(summary))
        return summary

    except Exception as exc:
        log.error("generate_summary failed: %s", exc, exc_info=True)
        return ""


def answer_question(question: str, forecast_data: dict) -> str:
    """
    Answer a single user question using current forecast data + model knowledge.
    Uses the google-generativeai SDK with GEMINI_API_KEY from the environment.
    Single-turn — no conversation history is kept between calls.

    Returns a user-facing error string on failure (never raises).
    """
    if not question or not question.strip():
        return "Please type a question and I'll do my best to answer it."

    try:
        model   = _get_model()
        context = _format_forecast_as_ai_context(forecast_data)
        prompt  = (
            f"Current forecast data for Folsom, CA:\n{context}\n\n"
            f"User question: {question.strip()}"
        )
        response = model.generate_content(prompt)
        answer   = response.text.strip()
        log.info("Question answered (%s chars)", len(answer))
        return answer

    except Exception as exc:
        log.error("answer_question failed: %s", exc, exc_info=True)
        return (
            "Sorry, I couldn't process that question right now. "
            "Please check that the GEMINI_API_KEY is set and try again."
        )


def answer_question_with_key(
    question: str,
    forecast_data: dict,
    api_key: str,
) -> str:
    """
    Answer a user question via the Gemini REST API, accepting the key as a parameter.

    Why this variant exists: the Streamlit frontend sources its API key from
    st.secrets, not from os.environ. Accepting the key as a parameter lets the
    Dashboard pass it directly without this module needing to know about
    Streamlit's secrets API.

    Uses the same REST endpoint and system prompt as the frontend's former
    inline _call_gemini() function, now consolidated here as the single
    canonical implementation.

    Returns a user-facing error string on failure (never raises).
    """
    if not question or not question.strip():
        return "Please type a question and I'll do my best to answer it."

    if not api_key:
        return (
            "⚠️ GEMINI_API_KEY is not set. "
            "Add it to Streamlit Cloud Secrets to enable AI responses."
        )

    context = _format_forecast_as_ai_context(forecast_data)
    prompt  = f"Current forecast data:\n{context}\n\nUser question: {question.strip()}"

    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents":           [{"parts": [{"text": prompt}]}],
        "generationConfig":   {"temperature": 0.4, "maxOutputTokens": 400},
    }
    try:
        resp = requests.post(
            f"{_GEMINI_REST_ENDPOINT}?key={api_key}",
            json=payload,
            timeout=20,
        )
        if not resp.ok:
            log.warning("Gemini REST HTTP %s: %s", resp.status_code, resp.text[:200])
            if resp.status_code == 400:
                return "⚠️ The API key appears to be invalid. Please check Streamlit Cloud Secrets."
            if resp.status_code == 429:
                return "The Navigator is busy right now. Please wait a moment and try again."
            if resp.status_code in (500, 503):
                return "The AI service is temporarily unavailable. Please try again in a minute."
            return f"The Navigator encountered an error (HTTP {resp.status_code}). Please try again."

        result     = resp.json()
        candidates = result.get("candidates", [])
        if not candidates:
            log.warning("Gemini REST returned no candidates: %s", result)
            return "The Navigator received an unexpected response. Please try again."

        answer = candidates[0]["content"]["parts"][0]["text"].strip()
        log.info("Question answered via REST (%s chars)", len(answer))
        return answer

    except requests.exceptions.Timeout:
        return "The AI took too long to respond. Please try again."
    except Exception as exc:
        log.error("answer_question_with_key failed: %s", exc, exc_info=True)
        return "The Navigator encountered an unexpected error. Please try again."
