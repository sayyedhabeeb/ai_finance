"""
API Routes - Query endpoints.

Handles user queries to the AI Financial Brain, including standard requests,
streaming-compatible endpoints, and async query status polling.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, confloat
from starlette.concurrency import run_in_threadpool

from backend.config.schemas import AgentTask, AgentType
from backend.services.agent_factory import AgentFactory
from backend.services.llm_service import generate_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])
_agent_factory = AgentFactory()


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


class QueryRequest(BaseModel):
    """Inbound query payload for /query endpoint."""
    query: str = Field(..., min_length=1, max_length=10_000)


class QueryResponse(BaseModel):
    """Response payload for /query endpoint."""
    query: Optional[str] = None
    response: Optional[str] = None
    status: str
    message: Optional[str] = None


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


async def _run_llm_response(user_query: str) -> tuple[str, str]:
    answer = await run_in_threadpool(generate_response, user_query)
    return answer, "llama3-8b-8192"


def _route_agent_type(user_query: str) -> AgentType | None:
    lowered = user_query.lower()
    if "portfolio" in lowered:
        return AgentType.PORTFOLIO_MANAGER
    if "risk" in lowered:
        return AgentType.RISK_ANALYST
    if "market" in lowered:
        return AgentType.MARKET_ANALYST
    return None


async def _run_agent_completion(agent_type: AgentType, user_query: str) -> tuple[str, str]:
    agent = _agent_factory.create_agent(agent_type)
    task = AgentTask(
        agent_type=agent_type,
        query=user_query,
        context={},
    )
    result = await agent.execute(task)
    output = result.output
    if isinstance(output, dict):
        answer = (
            output.get("response")
            or output.get("answer")
            or output.get("summary")
            or output.get("output")
            or str(output)
        )
    else:
        answer = str(output)
    return answer.strip(), agent_type.value


async def _dispatch_to_agent(query_text: str, query_id: str, redis_client, request: Request) -> Dict[str, Any]:
    """Dispatch a user query to keyword-routed agent, with Groq fallback."""
    import json

    user_query = query_text

    routed_agent = _route_agent_type(user_query)
    answer = ""
    model_name = ""
    path = "llm"

    if routed_agent is not None:
        try:
            answer, model_name = await _run_agent_completion(routed_agent, user_query)
            path = f"agent:{routed_agent.value}"
            logger.info("Query routed to agent path=%s query_id=%s", path, query_id)
        except Exception:
            logger.exception(
                "Agent execution failed; falling back to LLM path=agent:%s query_id=%s",
                routed_agent.value,
                query_id,
            )

    if not answer:
        context_chunks = _retrieve_context_chunks(request, user_query)
        prompt_query = user_query
        if context_chunks:
            context_blob = "\n\n".join(f"- {chunk}" for chunk in context_chunks)
            prompt_query = (
                f"{user_query}\n\nRelevant retrieved context:\n{context_blob}\n\n"
                "Use the context when relevant, but do not fabricate facts."
            )
        answer, model_name = await _run_llm_response(prompt_query)
        path = "llm:groq"
        logger.info("Query routed to LLM path=%s query_id=%s", path, query_id)

    result = {
        "status": "success",
        "answer": answer,
        "data": {
            "detected_intent": "financial_query",
            "provider": "groq",
            "model": model_name,
            "path": path,
        },
        "sources": ["agent" if path.startswith("agent:") else "llm"],
        "agent_type": routed_agent.value if routed_agent and path.startswith("agent:") else "llm_assistant",
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
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a query to the AI Financial Brain",
    description="Accepts a natural-language financial query and returns a structured response.",
)
async def post_query(
    body: QueryRequest,
    request: Request,
) -> QueryResponse:
    query_id = f"q_{uuid.uuid4()}"
    user_query = body.query

    redis_client = getattr(request.app.state, "redis", None)
    logger.info("Received query request query_id=%s query=%s", query_id, user_query[:120])
    _store_query(query_id, {"query": user_query})

    try:
        result = await _dispatch_to_agent(user_query, query_id, redis_client, request)
    except Exception as exc:
        logger.exception("Agent dispatch failed for query %s", query_id)
        _complete_query(query_id, {"status": "error", "error": str(exc)})
        return QueryResponse(
            query=user_query,
            response=None,
            status="error",
            message=str(exc),
        )

    _complete_query(query_id, result)
    logger.info(
        "Completed query request query_id=%s status=success response=%s",
        query_id,
        result.get("answer", "")[:200],
    )
    return QueryResponse(
        query=user_query,
        response=result.get("answer", ""),
        status="success",
        message=None,
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
