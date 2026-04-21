"""
AI Financial Brain - FastAPI Application Entry Point.

Full production FastAPI application with lifespan management, CORS,
Prometheus metrics, and all API routers.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import prometheus_client
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.routes.query import router as query_router, legacy_router
from backend.api.routes.portfolio import router as portfolio_router
from backend.api.routes.risk import router as risk_router
from backend.api.routes.memory import router as memory_router
from backend.api.routes.health import router as health_router
from backend.api.websockets.stream import router as ws_router
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = prometheus_client.Counter(
    "aifb_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = prometheus_client.Histogram(
    "aifb_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0],
)

REQUEST_IN_PROGRESS = prometheus_client.Gauge(
    "aifb_http_requests_in_progress",
    "HTTP requests currently being processed",
    ["method", "endpoint"],
)

ACTIVE_WS_CONNECTIONS = prometheus_client.Gauge(
    "aifb_websocket_connections_active",
    "Active WebSocket connections",
)

MODEL_INFERENCE_COUNT = prometheus_client.Counter(
    "aifb_model_inference_total",
    "Total model inference calls",
    ["model_name"],
)

MODEL_INFERENCE_LATENCY = prometheus_client.Histogram(
    "aifb_model_inference_duration_seconds",
    "Model inference latency in seconds",
    ["model_name"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that records Prometheus metrics for every HTTP request."""

    async def dispatch(self, request: Request, call_next):
        method = request.method
        # Strip query params for cleaner labels
        path = request.url.path

        REQUEST_IN_PROGRESS.labels(method=method, endpoint=path).inc()

        start_time = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            REQUEST_IN_PROGRESS.labels(method=method, endpoint=path).dec()
            REQUEST_COUNT.labels(method=method, endpoint=path, status_code=500).inc()
            raise exc

        duration = time.perf_counter() - start_time
        status = response.status_code

        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
        REQUEST_COUNT.labels(method=method, endpoint=path, status_code=status).inc()
        REQUEST_IN_PROGRESS.labels(method=method, endpoint=path).dec()

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach an X-Request-ID header to every response (create if absent)."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or _generate_request_id()
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def _generate_request_id() -> str:
    import uuid
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup & shutdown hooks."""
    logger.info("AI Financial Brain API starting up ...")
    settings = get_settings()

    # --- Startup ---
    from backend.database.connection import DatabaseManager  # noqa: delayed import

    db = DatabaseManager()
    await db.initialize_pool()
    app.state.db = db
    logger.info("PostgreSQL connection pool initialised")

    try:
        import redis.asyncio as aioredis
        app.state.redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        await app.state.redis.ping()
        logger.info("Redis connection established")
    except Exception:
        logger.warning("Redis not available — caching disabled")
        app.state.redis = None

    try:
        import weaviate
        app.state.weaviate = weaviate.Client(settings.weaviate_url)
        app.state.weaviate.is_ready()
        logger.info("Weaviate connection established")
    except Exception:
        logger.warning("Weaviate not available — vector search disabled")
        app.state.weaviate = None

    # Warm Prometheus metrics registry
    prometheus_client.REGISTRY._names_to_collectors  # noqa: force init

    logger.info("AI Financial Brain API ready")

    yield

    # --- Shutdown ---
    logger.info("AI Financial Brain API shutting down …")

    if hasattr(app.state, "db"):
        await app.state.db.close_pool()
        logger.info("PostgreSQL connection pool closed")

    if hasattr(app.state, "redis") and app.state.redis is not None:
        await app.state.redis.close()
        logger.info("Redis connection closed")

    # Weaviate client has no explicit close in v3

    logger.info("AI Financial Brain API stopped")


# ---------------------------------------------------------------------------
# FastAPI Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Financial Brain API",
    description=(
        "Production-grade REST + WebSocket API for the AI Financial Brain platform. "
        "Provides portfolio management, risk analytics, market intelligence, "
        "and conversational AI query endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "AI Financial Brain Team", "email": "admin@aifinbrain.io"},
    license_info={"name": "Proprietary"},
)

# ---------------------------------------------------------------------------
# Middleware Registration (order matters)
# ---------------------------------------------------------------------------

app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # tighten in production
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health_router)
app.include_router(legacy_router)
app.include_router(query_router)
app.include_router(portfolio_router)
app.include_router(risk_router)
app.include_router(memory_router)
app.include_router(ws_router)


# ---------------------------------------------------------------------------
# Global Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
    return JSONResponse(status_code=403, content={"detail": str(exc)})
