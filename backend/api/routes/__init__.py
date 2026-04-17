"""
API Routes package.
"""

from backend.api.routes.query import router as query_router
from backend.api.routes.portfolio import router as portfolio_router
from backend.api.routes.risk import router as risk_router
from backend.api.routes.memory import router as memory_router
from backend.api.routes.health import router as health_router

__all__ = [
    "query_router",
    "portfolio_router",
    "risk_router",
    "memory_router",
    "health_router",
]
