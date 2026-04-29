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

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, confloat
from starlette.concurrency import run_in_threadpool

from backend.app.api.deps import get_current_user
from backend.config.schemas import AgentTask, AgentType
from backend.services.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])
legacy_router = APIRouter(tags=["legacy"])
_orchestrator: Optional[Orchestrator] = None


def _get_orchestrator() -> Orchestrator:
    """Lazily initialize the orchestrator on first query request."""
    global _orchestrator  # noqa: PLW0603
    if _orchestrator is None:
        _orchestrator = Orchestrator.from_settings()
    return _orchestrator


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
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class QueryResponse(BaseModel):
    """Response payload for /query endpoint."""
    query: Optional[str] = None
    response: Optional[str] = None
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    agent_type: Optional[str] = None


class LegacyChatRequest(BaseModel):
    """Inbound payload for legacy /chat endpoints."""
    message: Optional[str] = Field(
        None,
        min_length=1,
        max_length=10_000,
        description="Preferred frontend field for user input",
    )
    query: Optional[str] = Field(
        None,
        min_length=1,
        max_length=10_000,
        description="Alias field for user input",
    )
    user_id: Optional[str] = Field(
        "anonymous",
        min_length=1,
        max_length=128,
        description="Optional user identifier",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation/session identifier",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Analyze my portfolio risk and suggest rebalancing.",
                "user_id": "user_123",
                "conversation_id": "sess_abc123",
            }
        }


class OrchestrationResponse(BaseModel):
    """Normalized response payload for query/chat endpoints."""
    answer: Optional[str] = None
    agents_used: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    model_name = get_settings().groq_model or "llama-3.1-8b-instant"
    return answer, model_name


def _route_agent_type(user_query: str) -> AgentType | None:
    lowered = user_query.lower()
    if "portfolio" in lowered:
        return AgentType.PORTFOLIO_MANAGER
    if "risk" in lowered:
        return AgentType.RISK_ANALYST
    if "market" in lowered:
        return AgentType.MARKET_ANALYST
    return None


def _is_canned_response(text: str) -> bool:
    lowered = (text or "").lower()
    canned_markers = (
        "based on my analysis of your current portfolio",
        "portfolio summary",
        "analysis based on data from portfolio agent",
        "key observations",
        "recommendations",
    )
    return any(marker in lowered for marker in canned_markers)


async def _run_agent_completion(agent_type: AgentType, user_query: str) -> tuple[str, str, float]:
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
    return answer.strip(), agent_type.value, result.confidence


async def _dispatch_to_agent(query_text: str, query_id: str, redis_client, request: Request) -> Dict[str, Any]:
    """Dispatch a user query to keyword-routed agent, with Groq fallback."""
    import json
    settings = get_settings()

    user_query = query_text

    routed_agent = _route_agent_type(user_query)
    answer = ""
    model_name = ""
    confidence = 0.0
    sources = []
    path = "llm"

    if routed_agent is not None:
        try:
            answer, model_name, confidence = await _run_agent_completion(routed_agent, user_query)
            sources = ["agent"]
            if _is_canned_response(answer):
                logger.warning(
                    "Canned agent response detected; forcing LLM fallback query_id=%s path=agent:%s",
                    query_id,
                    routed_agent.value,
                )
                answer = ""
            else:
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
        confidence = 0.85  # Default LLM confidence
        sources = ["llm"]
        if context_chunks:
            sources.append("vector_db")
        path = "llm:groq"
        logger.info("Query routed to LLM path=%s query_id=%s", path, query_id)

    result = {
        "status": "success",
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "data": {
            "detected_intent": "financial_query",
            "provider": settings.llm_provider,
            "model": model_name or settings.groq_model,
            "path": path,
        },
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
    response_model=OrchestrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a query to the AI Financial Brain",
)
async def post_query(
    body: QueryRequest,
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    body.user_id = user["user_id"]
    user_query = body.query
    logger.info("orchestration_layer.routing_query user_id=%s query=%s", body.user_id, user_query[:120])

    # Process through the full LangGraph orchestration layer
    result = await _get_orchestrator().process_query(
        user_query,
        user_id=body.user_id,
        session_id=body.session_id or "",
    )

    # Map into the requested structured output format
    return {
        "answer": result.get("response"),
        "agents_used": list(result.get("agent_results", {}).keys()),
        "confidence": result.get("confidence", 0.0),
        "sources": result.get("sources", []),
        "metadata": {
            "query_type": result.get("query_type"),
            "execution_time_sec": round(result.get("execution_time", 0.0), 3),
        },
    }
@legacy_router.post(
    "/chat",
    response_model=OrchestrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Frontend alias for submitting a query",
)
async def post_chat(
    body: LegacyChatRequest,
    request: Request,
) -> Dict[str, Any]:
    """Alias for /query to support legacy/current frontend code."""
    query_text = body.message or body.query
    if not query_text:
        raise HTTPException(status_code=400, detail="Message or query is required")
    
    # Extract query and pass to same orchestration logic
    result = await _get_orchestrator().process_query(
        query_text,
        user_id=body.user_id or "anonymous",
        session_id=body.conversation_id or "",
    )
    
    return {
        "answer": result.get("response"),
        "agents_used": list(result.get("agent_results", {}).keys()),
        "confidence": result.get("confidence", 0.0),
        "sources": result.get("sources", []),
        "metadata": {
            "query_type": result.get("query_type"),
            "execution_time_sec": round(result.get("execution_time", 0.0), 3),
        }
    }


@legacy_router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Frontend alias for streaming query",
)
async def post_chat_stream(
    body: LegacyChatRequest,
    request: Request,
) -> StreamChunk:
    """Alias for /query/stream."""
    # Convert simple chat body to UserQuery-like schema
    query_text = body.message or body.query
    if not query_text:
        raise HTTPException(status_code=400, detail="Message or query is required")
    user_id = body.user_id or "anonymous"
    
    internal_body = UserQuery(
        user_id=user_id,
        query=query_text,
        session_id=body.conversation_id,
    )
    return await post_query_stream(internal_body, request)


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
