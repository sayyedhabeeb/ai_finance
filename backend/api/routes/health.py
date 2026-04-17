"""
API Routes - Health & Metrics endpoints.

Lightweight liveness / readiness probes and Prometheus metrics exposition.
"""

import logging
import time
from typing import Any, Dict

import prometheus_client
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    status: str  # healthy | degraded | unhealthy
    latency_ms: float = 0.0
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall: healthy | degraded | unhealthy")
    version: str = "1.0.0"
    uptime_seconds: float
    components: Dict[str, ComponentHealth]


# ---------------------------------------------------------------------------
# Start time
# ---------------------------------------------------------------------------

_start_time = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness & readiness probe",
    description="Checks connectivity to PostgreSQL, Redis, and Weaviate. Returns 200 if healthy.",
)
async def health_check(request: Request) -> HealthResponse:
    components: Dict[str, ComponentHealth] = {}
    overall = "healthy"

    # --- PostgreSQL ---
    db = getattr(request.app.state, "db", None)
    if db is not None:
        t0 = time.perf_counter()
        try:
            await db.fetch_one("SELECT 1 as ok")
            components["postgres"] = ComponentHealth(
                status="healthy",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            components["postgres"] = ComponentHealth(status="unhealthy", error=str(exc))
            overall = "unhealthy"
    else:
        components["postgres"] = ComponentHealth(status="unhealthy", error="not initialised")
        overall = "degraded"

    # --- Redis ---
    redis = getattr(request.app.state, "redis", None)
    if redis is not None:
        t0 = time.perf_counter()
        try:
            await redis.ping()
            components["redis"] = ComponentHealth(
                status="healthy",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            components["redis"] = ComponentHealth(status="unhealthy", error=str(exc))
            if overall == "healthy":
                overall = "degraded"
    else:
        components["redis"] = ComponentHealth(status="unhealthy", error="not initialised")
        if overall == "healthy":
            overall = "degraded"

    # --- Weaviate ---
    weaviate = getattr(request.app.state, "weaviate", None)
    if weaviate is not None:
        t0 = time.perf_counter()
        try:
            is_ready = weaviate.is_ready()
            latency = round((time.perf_counter() - t0) * 1000, 2)
            if is_ready:
                components["weaviate"] = ComponentHealth(status="healthy", latency_ms=latency)
            else:
                components["weaviate"] = ComponentHealth(status="unhealthy", latency_ms=latency)
                if overall == "healthy":
                    overall = "degraded"
        except Exception as exc:
            components["weaviate"] = ComponentHealth(status="unhealthy", error=str(exc))
            if overall == "healthy":
                overall = "degraded"
    else:
        components["weaviate"] = ComponentHealth(status="unhealthy", error="not initialised")
        if overall == "healthy":
            overall = "degraded"

    return HealthResponse(
        status=overall,
        uptime_seconds=round(time.time() - _start_time, 2),
        components=components,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=PlainTextResponse,
    description="Exposes Prometheus-format metrics for scraping.",
)
async def metrics():
    """Return the Prometheus metrics in text exposition format."""
    output = prometheus_client.generate_latest()
    return PlainTextResponse(content=output.decode("utf-8"), media_type="text/plain; version=0.0.4; charset=utf-8")
