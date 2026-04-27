"""
api.py — FastAPI backend for Folsom AQI Forecast.
Serves cached forecasts from the GitHub CDN — never runs ML models inline.

Start with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()  # Load .env file (AIRNOW_API_KEY etc.)

from inference import cache_age_minutes, load_cached_forecast
from logger import get_logger

log = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Maximum acceptable age of the cached forecast before signaling staleness.
# GitHub Actions runs every hour; 90 minutes allows one missed run before alerting.
STALE_FORECAST_THRESHOLD_MINUTES: int = 90

# ─── Module-level state ───────────────────────────────────────────────────────

# Track startup time for health endpoint
_startup_time: str = ""
_models_loaded: bool = False


# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server startup/shutdown lifecycle hook.

    CDN-proxy mode: ML inference is handled by the GitHub Actions worker.
    Models are NOT loaded into Render's process memory because the 512MB
    memory limit (Render free tier) causes R14 OOM restarts when all 12
    LightGBM models (~20,000 trees each) are held in RAM simultaneously.
    The /forecast endpoint fetches pre-computed JSON from the GitHub CDN instead.
    """
    global _models_loaded
    # CDN-proxy mode: intentionally False — see docstring above.
    _models_loaded = False
    log.info("Running in CDN-proxy mode. ML inference handled by external worker.")
    yield
    # (no shutdown cleanup needed)


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Folsom AQI Forecast API",
    description="Real-time AQI forecast for Folsom, CA using LightGBM + Open-Meteo",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins so the Streamlit Cloud frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """
    Root endpoint. Returns a simple welcome message since all actual
    API logic is under specific routes (like /forecast or /health).
    """
    return {
        "message": "Folsom AQI Forecast API is running.",
        "status_endpoint": "/health",
        "doc_endpoint": "/docs",
    }


@app.get("/health")
async def health():
    """
    Health check. Returns model load status, last refresh time, and staleness signal.
    Used by deploy.sh to verify the service is up.

    Returns HTTP 200 even when data_stale=True so Render's health check does not
    restart the service due to a missed GitHub Actions refresh run.
    """
    cached = load_cached_forecast(prefer_remote=True)
    last_refresh = cached.get("generated_at", "never") if cached else "never"
    age = cache_age_minutes()

    return {
        "status": "ok",
        "models_loaded": True,  # CDN-proxy mode: models are pre-computed externally
        "startup_time": _startup_time,
        "last_refresh": last_refresh,
        "cache_age_minutes": age,
        "data_stale": age > STALE_FORECAST_THRESHOLD_MINUTES,
        "stale_threshold_minutes": STALE_FORECAST_THRESHOLD_MINUTES,
    }


@app.get("/forecast")
async def get_forecast():
    """
    Return the cached AQI forecast JSON.
    Always pulls from the GitHub CDN (data-cache branch) to bypass
    Render IP rate limits.
    """
    cached = load_cached_forecast(prefer_remote=True)
    if cached is None:
        raise HTTPException(
            status_code=503, detail="No forecast data available from CDN or local cache."
        )

    return JSONResponse(content=cached)


@app.get("/refresh")
async def trigger_refresh():
    """
    Manually trigger a forecast fetch from the CDN.
    Note: Live recalculation is disabled on Render to avoid 429s.
    Recalculation happens automatically via GitHub Actions every hour.
    """
    try:
        result = load_cached_forecast(prefer_remote=True)
        if not result:
            raise RuntimeError("Could not fetch remote cache.")
        return {
            "refreshed": True,
            "source": "GitHub CDN (data-cache)",
            "generated_at": result.get("generated_at"),
            "data_freshness": result.get("data_freshness_minutes"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Remote refresh failed: {exc}")


@app.get("/current")
async def get_current():
    """
    Return only the current AQI reading (subset of /forecast).
    Lightweight endpoint for simple widgets.
    """
    # FIXED: Force remote CDN fetch
    cached = load_cached_forecast(prefer_remote=True)
    if not cached:
        raise HTTPException(status_code=503, detail="No forecast data available yet.")
    return {
        "current": cached.get("current"),
        "generated_at": cached.get("generated_at"),
        "location": cached.get("location"),
    }


@app.get("/history")
async def get_history():
    """
    Return the 72-hour history array (actual vs forecast).
    Used by the dashboard to draw the comparison chart.
    """
    # FIXED: Force remote CDN fetch
    cached = load_cached_forecast(prefer_remote=True)
    if not cached:
        raise HTTPException(status_code=503, detail="No forecast data available yet.")
    return {
        "history_72h": cached.get("history_72h", []),
        "generated_at": cached.get("generated_at"),
    }
