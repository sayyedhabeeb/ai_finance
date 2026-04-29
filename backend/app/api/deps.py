"""Shared FastAPI dependency helpers for backend API routes."""

from __future__ import annotations

from typing import Any

from fastapi import Header, HTTPException, Request, status

from backend.config.settings import Settings, get_settings as _get_settings


def get_settings() -> Settings:
    """Return the active application settings."""
    return _get_settings()


def get_db(request: Request) -> Any:
    """Return the PostgreSQL database manager from app state."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not initialised.",
        )
    return db


def get_redis(request: Request) -> Any:
    """Return the Redis client from app state."""
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client is not available.",
        )
    return redis_client


def get_weaviate(request: Request) -> Any:
    """Return the Weaviate client from app state."""
    weaviate_client = getattr(request.app.state, "weaviate", None)
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate client is not available.",
        )
    return weaviate_client


def get_request_id(request: Request) -> str:
    """Return the current request ID from request state."""
    return getattr(request.state, "request_id", "")


def get_current_user(authorization: str | None = Header(None)) -> dict:
    """Simple bearer-token authentication dependency."""
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid format")

    token = authorization.split(" ", 1)[1]

    if token != "demo-token":
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"user_id": "demo_user"}
