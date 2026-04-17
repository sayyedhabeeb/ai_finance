"""
API Routes - Memory & Context endpoints.

Manages user preferences, conversation context, and session history
using Redis for fast access and PostgreSQL for persistence.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["memory"])


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class UserContext(BaseModel):
    user_id: str
    preferences: Dict[str, Any]
    risk_profile: Optional[str] = None
    investment_horizon: Optional[str] = None
    portfolio_summary: Optional[Dict[str, Any]] = None
    recent_queries: List[Dict[str, str]] = Field(default_factory=list)
    financial_goals: List[Dict[str, Any]] = Field(default_factory=list)


class SetPreferencesRequest(BaseModel):
    preferences: Dict[str, Any] = Field(
        ...,
        description="Key-value preferences. Supports nested dicts.",
        example={
            "risk_tolerance": "moderate",
            "preferred_sectors": ["technology", "healthcare"],
            "communication_style": "concise",
            "currency": "USD",
            "notification_threshold_pct": 5.0,
        },
    )


class SetPreferencesResponse(BaseModel):
    user_id: str
    updated_keys: List[str]
    message: str


class SessionHistoryItem(BaseModel):
    session_id: str
    query_id: str
    query: str
    answer_summary: Optional[str] = None
    agent_type: Optional[str] = None
    created_at: str
    latency_ms: Optional[float] = None


class SessionHistoryResponse(BaseModel):
    user_id: str
    sessions: List[SessionHistoryItem]
    total_count: int
    has_more: bool


class ClearSessionResponse(BaseModel):
    user_id: str
    session_id: str
    deleted_count: int
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_redis(request: Request):
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return redis


def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/memory/{user_id}/context",
    response_model=UserContext,
    summary="Get full user context for personalisation",
)
async def get_user_context(user_id: str, request: Request) -> UserContext:
    redis = _get_redis(request)
    db = _get_db(request)

    # Try Redis cache first
    cache_key = f"user:{user_id}:context"
    cached = await redis.get(cache_key)
    if cached:
        return UserContext(**json.loads(cached))

    # Fetch from DB
    prefs_rows = await db.fetch_all(
        "SELECT preference_key, preference_value FROM user_preferences WHERE user_id = $1",
        user_id,
    )
    preferences: Dict[str, Any] = {}
    for row in prefs_rows:
        preferences[row["preference_key"]] = row["preference_value"]

    # Fetch risk profile from users table
    user_row = await db.fetch_one("SELECT risk_profile FROM users WHERE id = $1", user_id)
    risk_profile = str(user_row["risk_profile"]) if user_row and user_row["risk_profile"] else None

    # Fetch financial goals
    goals_rows = await db.fetch_all(
        "SELECT goal_type, target_amount, current_amount, target_date, priority FROM financial_goals WHERE user_id = $1",
        user_id,
    )
    financial_goals = [dict(g) for g in goals_rows]

    # Fetch recent queries
    recent_rows = await db.fetch_all(
        """
        SELECT query_id, agent_type, output, latency_ms, created_at
        FROM agent_executions
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT 10
        """,
        user_id,
    )
    recent_queries = []
    for r in recent_rows:
        output = r.get("output")
        summary = ""
        if output:
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    output = {}
            summary = str(output.get("answer", ""))[:120]
        recent_queries.append({
            "query_id": str(r["query_id"]),
            "answer_summary": summary,
            "agent_type": r.get("agent_type"),
            "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
        })

    # Fetch portfolio summary
    portfolio = await db.fetch_one(
        "SELECT name, currency, total_value FROM portfolios WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 1",
        user_id,
    )
    portfolio_summary = dict(portfolio) if portfolio else None

    context = UserContext(
        user_id=user_id,
        preferences=preferences,
        risk_profile=risk_profile,
        financial_goals=financial_goals,
        recent_queries=recent_queries,
        portfolio_summary=portfolio_summary,
    )

    # Cache for 5 minutes
    await redis.setex(cache_key, 300, json.dumps(context.model_dump()))

    return context


@router.post(
    "/memory/{user_id}/preferences",
    response_model=SetPreferencesResponse,
    summary="Set or update user preferences",
)
async def set_preferences(
    user_id: str,
    body: SetPreferencesRequest,
    request: Request,
) -> SetPreferencesResponse:
    redis = _get_redis(request)
    db = _get_db(request)

    updated_keys: List[str] = []
    for key, value in body.preferences.items():
        # Upsert into PostgreSQL
        await db.execute(
            """
            INSERT INTO user_preferences (id, user_id, preference_key, preference_value)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (user_id, preference_key) DO UPDATE
                SET preference_value = $4::jsonb
            """,
            str(uuid.uuid4()),
            user_id,
            key,
            json.dumps(value),
        )
        updated_keys.append(key)

    # Invalidate cache
    await redis.delete(f"user:{user_id}:context")

    return SetPreferencesResponse(
        user_id=user_id,
        updated_keys=updated_keys,
        message=f"Updated {len(updated_keys)} preference(s)",
    )


@router.get(
    "/memory/{user_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get user session / query history",
)
async def get_session_history(
    user_id: str,
    request: Request,
    limit: int = 20,
    offset: int = 0,
) -> SessionHistoryResponse:
    db = _get_db(request)

    rows = await db.fetch_all(
        """
        SELECT query_id, agent_type, input_data, output, latency_ms, created_at
        FROM agent_executions
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        """,
        user_id,
        limit,
        offset,
    )

    total_row = await db.fetch_one(
        "SELECT COUNT(*) as cnt FROM agent_executions WHERE user_id = $1",
        user_id,
    )
    total_count = int(total_row["cnt"]) if total_row else 0

    items: List[SessionHistoryItem] = []
    for r in rows:
        input_data = r.get("input_data")
        query_text = ""
        answer_summary = ""
        if input_data:
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except (json.JSONDecodeError, TypeError):
                    input_data = {}
            query_text = str(input_data.get("query", ""))[:500]

        output = r.get("output")
        if output:
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    output = {}
            answer_summary = str(output.get("answer", ""))[:200]

        items.append(
            SessionHistoryItem(
                session_id=str(r["query_id"]),
                query_id=str(r["query_id"]),
                query=query_text,
                answer_summary=answer_summary or None,
                agent_type=r.get("agent_type"),
                created_at=r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
                latency_ms=float(r["latency_ms"]) if r.get("latency_ms") is not None else None,
            )
        )

    return SessionHistoryResponse(
        user_id=user_id,
        sessions=items,
        total_count=total_count,
        has_more=(offset + limit) < total_count,
    )


@router.delete(
    "/memory/{user_id}/session/{session_id}",
    response_model=ClearSessionResponse,
    summary="Delete a session's query history",
    status_code=status.HTTP_200_OK,
)
async def clear_session(
    user_id: str,
    session_id: str,
    request: Request,
) -> ClearSessionResponse:
    db = _get_db(request)
    redis = _get_redis(request)

    result = await db.execute(
        "DELETE FROM agent_executions WHERE user_id = $1 AND query_id = $2",
        user_id,
        session_id,
    )

    # Invalidate user context cache
    await redis.delete(f"user:{user_id}:context")

    return ClearSessionResponse(
        user_id=user_id,
        session_id=session_id,
        deleted_count=int(result or 0),
        message="Session cleared successfully",
    )
