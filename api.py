"""
api.py — FastAPI backend for Folsom AQI Forecast.
Loads all 12 models once at startup. Serves cached forecasts — never runs
ML models inline on a web request.

Start with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()  # Load .env file (AIRNOW_API_KEY etc.)

from inference import load_all_models, predict_now, load_cached_forecast, cache_age_minutes

# ─── CORS ─────────────────────────────────────────────────────────────────────

# Track startup time for health endpoint
_startup_time: str = ""
_models_loaded: bool = False


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all 12 models and imputers into memory at startup.
    This is the only time we touch the disk for model files.
    Keeps per-request latency < 200ms.
    """
    global _startup_time, _models_loaded
    _startup_time = datetime.now().isoformat()

    print("[api] Loading models...", file=sys.stderr)
    models_dir = Path("models")
    if not models_dir.exists():
        raise RuntimeError(
            "models/ directory not found. "
            "Run train.py locally and copy models/ to the server with deploy.sh."
        )

    try:
        load_all_models()
        _models_loaded = True
        print("[api] All 12 models loaded. Ready to serve.", file=sys.stderr)
    except RuntimeError as exc:
        print(f"[api] FATAL: {exc}", file=sys.stderr)
        raise

    yield   # app runs here
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
        "doc_endpoint": "/docs"
    }


@app.get("/health")
async def health():
    """
    Health check. Returns model load status and last refresh time.
    Used by deploy.sh to verify the service is up.
    """
    cached = load_cached_forecast()
    last_refresh = cached.get("generated_at", "never") if cached else "never"

    return {
        "status":        "ok",
        "models_loaded": _models_loaded,
        "startup_time":  _startup_time,
        "last_refresh":  last_refresh,
        "cache_age_minutes": cache_age_minutes(),
    }


@app.get("/forecast")
async def get_forecast():
    """
    Return the cached AQI forecast JSON.

    If the cache is older than 90 minutes, triggers a synchronous refresh
    before returning (this takes ~5-10 seconds but keeps data fresh even if
    the cron job fails).

    Response time target: < 200ms when cache is warm.
    """
    age = cache_age_minutes()

    # If cache is stale (or missing), refresh synchronously
    if age > 90:
        print(f"[api] Cache is {age} min old — refreshing synchronously...", file=sys.stderr)
        try:
            result = predict_now()
            return JSONResponse(content=result)
        except Exception as exc:
            # If refresh fails, try returning whatever is cached
            print(f"[api] Refresh failed: {exc}", file=sys.stderr)
            cached = load_cached_forecast()
            if cached:
                cached["_stale_warning"] = f"Cache is {age} min old; refresh failed: {exc}"
                return JSONResponse(content=cached, status_code=200)
            raise HTTPException(status_code=503, detail=f"No forecast available: {exc}")

    # Serve from cache (fast path)
    cached = load_cached_forecast()
    if cached is None:
        # No cache at all — first-run scenario
        print("[api] No cache found — generating first forecast...", file=sys.stderr)
        try:
            result = predict_now()
            return JSONResponse(content=result)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Failed to generate forecast: {exc}")

    return JSONResponse(content=cached)


@app.get("/refresh")
async def trigger_refresh():
    """
    Manually trigger a forecast refresh. Called by cron every hour.
    Also useful for testing immediately after deployment.
    Returns confirmation with the generated_at timestamp.
    """
    try:
        result = predict_now()
        return {
            "refreshed":     True,
            "generated_at":  result.get("generated_at"),
            "data_freshness_minutes": result.get("data_freshness_minutes"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {exc}")


@app.get("/current")
async def get_current():
    """
    Return only the current AQI reading (subset of /forecast).
    Lightweight endpoint for simple widgets.
    """
    cached = load_cached_forecast()
    if not cached:
        raise HTTPException(status_code=503, detail="No forecast data available yet.")
    return {
        "current":      cached.get("current"),
        "generated_at": cached.get("generated_at"),
        "location":     cached.get("location"),
    }


@app.get("/history")
async def get_history():
    """
    Return the 72-hour history array (actual vs forecast).
    Used by the dashboard to draw the comparison chart.
    """
    cached = load_cached_forecast()
    if not cached:
        raise HTTPException(status_code=503, detail="No forecast data available yet.")
    return {
        "history_72h":  cached.get("history_72h", []),
        "generated_at": cached.get("generated_at"),
    }
