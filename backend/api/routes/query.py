"""
API Routes - Query endpoints.

Handles user queries to the AI Financial Brain, including standard requests,
streaming-compatible endpoints, and async query status polling.
"""

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from langchain_groq import ChatGroq
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, confloat

from backend.api.middleware.auth import get_current_user
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class UserQuery(BaseModel):
    """Inbound user query payload."""
    user_id: str = Field(..., min_length=1, max_length=128, description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Optional session id for multi-turn")
    query: str = Field(..., min_length=1, max_length=10_000, description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extra context")
    stream: bool = Field(default=False, description="Request streaming response")
    model: Optional[str] = Field("default", description="Agent / model variant to invoke")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "usr_abc123",
                "session_id": "sess_xyz789",
                "query": "What is the risk-adjusted return of my tech portfolio?",
                "stream": False,
            }
        }


class StreamChunk(BaseModel):
    """Single streaming response token."""
    query_id: str
    chunk_id: int
    content: str
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None


class SystemResponse(BaseModel):
    """Structured query response."""
    query_id: str
    status: str = Field(..., description="success | processing | failed")
    answer: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    confidence: Optional[confloat(ge=0.0, le=1.0)] = None
    sources: Optional[List[str]] = None
    agent_type: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query_id": "q_550e8400-e29b-41d4-a716-446655440000",
                "status": "success",
                "answer": "Your portfolio's Sharpe ratio is 1.42 ...",
                "data": {"sharpe_ratio": 1.42},
                "confidence": 0.91,
                "sources": ["market_data", "portfolio_db"],
                "agent_type": "portfolio_analyst",
                "latency_ms": 342.5,
            }
        }


class QueryStatus(BaseModel):
    query_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory query store (replace with Redis / DB in production)
# ---------------------------------------------------------------------------

_query_store: Dict[str, Dict[str, Any]] = {}


def _store_query(query_id: str, payload: Dict[str, Any]) -> None:
    _query_store[query_id] = {
        "status": "processing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


def _complete_query(query_id: str, result: Dict[str, Any]) -> None:
    entry = _query_store.get(query_id)
    if entry is None:
        return
    entry["status"] = result.get("status", "completed")
    entry["completed_at"] = datetime.now(timezone.utc).isoformat()
    entry.update(result)


# ---------------------------------------------------------------------------
# LLM + retrieval helpers
# ---------------------------------------------------------------------------


def _retrieve_context_chunks(request: Request, query_text: str, limit: int = 3) -> List[str]:
    """Best-effort Weaviate retrieval. Returns empty list when unavailable."""
    weaviate_client = getattr(request.app.state, "weaviate", None)
    if weaviate_client is None or not hasattr(weaviate_client, "query"):
        return []

    chunks: List[str] = []
    fields = ["title", "content", "text", "source"]

    for collection in ("MarketDocuments", "NewsDocuments"):
        try:
            response = (
                weaviate_client.query
                .get(collection, fields)
                .with_near_text({"concepts": [query_text]})
                .with_limit(limit)
                .do()
            )
            items = response.get("data", {}).get("Get", {}).get(collection, [])
            for item in items:
                text = item.get("content") or item.get("text") or item.get("title")
                if text:
                    chunks.append(str(text)[:500])
        except Exception:
            logger.debug("Weaviate retrieval skipped for collection=%s", collection, exc_info=True)

    return chunks[:limit]


async def _run_groq_completion(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    settings = get_settings()
    api_key = getattr(settings, "groq_api_key", "") or os.getenv("GROQ_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured")

    llm = ChatGroq(api_key=api_key, model=model, temperature=0.1)
    response = await llm.ainvoke(f"System:\n{system_prompt}\n\nUser:\n{user_prompt}")
    return str(response.content).strip(), model


async def _run_openai_completion(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    content = response.choices[0].message.content or ""
    return content.strip(), model


async def _dispatch_to_agent(query: UserQuery, query_id: str, redis_client, request: Request) -> Dict[str, Any]:
    """Dispatch a user query to configured LLM with optional retrieval context."""
    import json

    settings = get_settings()
    provider = str(getattr(settings, "llm_provider", os.getenv("LLM_PROVIDER", "groq"))).lower().strip()

    context_chunks = _retrieve_context_chunks(request, query.query)
    system_prompt = (
        "You are an AI financial assistant. Provide accurate, concise, practical guidance. "
        "If assumptions are made, state them briefly."
    )
    user_prompt = query.query

    if context_chunks:
        context_blob = "\n\n".join(f"- {chunk}" for chunk in context_chunks)
        user_prompt = (
            f"{query.query}\n\nRelevant retrieved context:\n{context_blob}\n\n"
            "Use the context when relevant, but do not fabricate facts."
        )

    if provider == "groq":
        answer, model_name = await _run_groq_completion(system_prompt, user_prompt)
    elif provider == "openai":
        answer, model_name = await _run_openai_completion(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER='{provider}'")

    result = {
        "status": "success",
        "answer": answer,
        "data": {
            "detected_intent": "financial_query",
            "provider": provider,
            "model": model_name,
            "context_chunks": len(context_chunks),
        },
        "confidence": 0.8,
        "sources": ["llm", *(["weaviate"] if context_chunks else [])],
        "agent_type": "llm_assistant",
    }

    if redis_client is not None:
        cache_key = f"query:{query_id}:result"
        await redis_client.setex(cache_key, 3600, json.dumps(result))

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=SystemResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a query to the AI Financial Brain",
    description="Accepts a natural-language financial query and returns a structured response.",
)
async def post_query(
    body: UserQuery,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> SystemResponse:
    query_id = f"q_{uuid.uuid4()}"
    start = time.perf_counter()
    user_id = current_user.get("user_id") or body.user_id

    redis_client = getattr(request.app.state, "redis", None)
    logger.info("Received query request query_id=%s user_id=%s", query_id, user_id)

    _store_query(query_id, {"user_id": user_id, "query": body.query})

    try:
        result = await _dispatch_to_agent(body, query_id, redis_client, request)
    except Exception as exc:
        logger.exception("Agent dispatch failed for query %s", query_id)
        _complete_query(query_id, {"status": "failed", "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent pipeline error: {exc}",
        )

    latency_ms = (time.perf_counter() - start) * 1000
    _complete_query(query_id, result)

    db = getattr(request.app.state, "db", None)
    if db is not None:
        try:
            import json

            await db.execute(
                """
                INSERT INTO agent_executions (id, user_id, query_id, agent_type, input_data, output, latency_ms)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7)
                """,
                uuid.uuid4(),
                user_id,
                query_id,
                result.get("agent_type", "unknown"),
                json.dumps({"query": body.query, "context": body.context}),
                json.dumps(result),
                latency_ms,
            )
        except Exception:
            logger.warning("Failed to log agent execution to DB", exc_info=True)

    logger.info("Completed query request query_id=%s status=%s", query_id, result["status"])

    return SystemResponse(
        query_id=query_id,
        status=result["status"],
        answer=result.get("answer"),
        data=result.get("data"),
        confidence=result.get("confidence"),
        sources=result.get("sources"),
        agent_type=result.get("agent_type"),
        latency_ms=round(latency_ms, 2),
    )


@router.post(
    "/query/stream",
    response_model=StreamChunk,
    status_code=status.HTTP_200_OK,
    summary="Submit a streaming query",
    description=(
        "Accepts a query and returns the first chunk immediately. "
        "Subsequent chunks are delivered via WebSocket on /ws/stream/{session_id}."
    ),
)
async def post_query_stream(
    body: UserQuery,
    request: Request,
) -> StreamChunk:
    query_id = f"q_{uuid.uuid4()}"
    session_id = body.session_id or f"sess_{uuid.uuid4()}"

    _store_query(query_id, {"user_id": body.user_id, "query": body.query, "session_id": session_id})

    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is not None:
        await redis_client.xadd(
            f"stream:{session_id}",
            {
                "query_id": query_id,
                "chunk_id": "0",
                "content": "",
                "done": "false",
            },
        )

    return StreamChunk(
        query_id=query_id,
        chunk_id=0,
        content="",
        done=False,
        metadata={"ws_endpoint": f"/ws/stream/{session_id}"},
    )


@router.get(
    "/query/{query_id}/status",
    response_model=QueryStatus,
    summary="Poll query status",
)
async def get_query_status(query_id: str) -> QueryStatus:
    entry = _query_store.get(query_id)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query not found")
    return QueryStatus(
        query_id=query_id,
        status=entry["status"],
        created_at=entry["created_at"],
        completed_at=entry.get("completed_at"),
        error=entry.get("error"),
    )
