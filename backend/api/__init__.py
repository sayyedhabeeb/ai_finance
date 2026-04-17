"""
AI Financial Brain - FastAPI Application Package.

Provides the REST API layer, WebSocket streaming, authentication middleware,
and Prometheus metrics instrumentation for the AI Financial Brain platform.
"""

from backend.api.app import app

__all__ = ["app"]
