"""
WebSocket Streaming Endpoint.

Provides real-time streaming of AI agent responses over WebSocket.
Authentication, heartbeat, and graceful error handling included.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

logger = logging.getLogger(__name__)

router = APIRouter()


class _NoopMetric:
    def inc(self) -> None:
        return

    def dec(self) -> None:
        return

    def observe(self, _value: float) -> None:
        return

    def labels(self, **_kwargs):
        return self


def _load_metrics():
    try:
        from backend.api.app import ACTIVE_WS_CONNECTIONS, MODEL_INFERENCE_COUNT, MODEL_INFERENCE_LATENCY
        return ACTIVE_WS_CONNECTIONS, MODEL_INFERENCE_COUNT, MODEL_INFERENCE_LATENCY
    except Exception:
        noop = _NoopMetric()
        return noop, noop, noop

# ---------------------------------------------------------------------------
# Authentication helper for WebSocket (no HTTP headers framework)
# ---------------------------------------------------------------------------

async def _authenticate_ws(websocket: WebSocket) -> Optional[str]:
    """
    Extract and validate auth from the WebSocket handshake headers.
    Returns user_id or None.
    """
    from backend.api.middleware.auth import decode_jwt, _API_KEYS

    # Check API key
    api_key = websocket.headers.get("x-api-key")
    if api_key and api_key in _API_KEYS:
        return _API_KEYS[api_key]["user_id"]

    # Check JWT
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
        try:
            payload = decode_jwt(token)
            return payload.get("sub")
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Message Generator (simulated agent streaming)
# ---------------------------------------------------------------------------

async def _generate_agent_stream(
    query: str,
    session_id: str,
    redis_client,
) -> None:
    """
    Yield streaming chunks as dicts.
    In production this calls the LangGraph agent pipeline.
    """
    # Simulated multi-step agent reasoning
    steps = [
        {"type": "thinking", "content": f"Analysing query: {query[:60]}…"},
        {"type": "data_fetch", "content": "Fetching portfolio data…"},
        {"type": "data_fetch", "content": "Retrieving market prices…"},
        {"type": "analysis", "content": "Running risk calculations…"},
        {"type": "analysis", "content": "Computing optimisation bounds…"},
        {"type": "answer", "content": "Based on your portfolio composition…"},
        {"type": "answer", "content": "Your current Sharpe ratio is 1.42, which is above the market average."},
        {"type": "answer", "content": "I recommend increasing allocation to defensive sectors."},
        {"type": "answer", "content": "Consider trimming tech exposure by 5-8% to reduce concentration risk."},
        {"type": "done", "content": "", "metadata": {"agent_type": "financial_analyst", "confidence": 0.89}},
    ]

    for i, step in enumerate(steps):
        await asyncio.sleep(0.15)  # simulate processing time
        chunk = {
            "chunk_id": i,
            "session_id": session_id,
            "timestamp": time.time(),
            **step,
        }

        # Push to Redis stream for other consumers
        if redis_client is not None:
            try:
                await redis_client.xadd(f"stream:{session_id}", {"data": json.dumps(chunk)})
            except Exception:
                logger.debug("Redis stream write failed", exc_info=True)

        yield chunk


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming AI agent responses.

    Protocol:
    - Client sends: {"action": "query", "query": "...", "context": {...}}
    - Server sends: {"chunk_id": int, "type": str, "content": str, "done": bool}
    - Client sends: {"action": "ping"}  →  Server sends: {"action": "pong"}
    """
    # Authenticate
    active_ws_connections, model_inference_count, model_inference_latency = _load_metrics()

    user_id = await _authenticate_ws(websocket)
    if user_id is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required")
        return

    await websocket.accept()
    active_ws_connections.inc()
    logger.info("WebSocket connected: session=%s user=%s", session_id, user_id)

    redis_client = None
    try:
        redis_client = getattr(websocket.app.state, "redis", None)
    except Exception:
        pass

    try:
        while True:
            # Receive client message with timeout for heartbeat
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"action": "keepalive"})
                continue

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            action = message.get("action", "")

            if action == "ping":
                await websocket.send_json({"action": "pong", "timestamp": time.time()})
                continue

            if action == "query":
                query_text = message.get("query", "")
                if not query_text.strip():
                    await websocket.send_json({"error": "Empty query"})
                    continue

                context = message.get("context", {})

                start = time.perf_counter()
                model_inference_count.labels(model_name="financial_analyst").inc()

                async for chunk in _generate_agent_stream(query_text, session_id, redis_client):
                    done = chunk.get("type") == "done"
                    payload = {
                        "chunk_id": chunk["chunk_id"],
                        "type": chunk["type"],
                        "content": chunk.get("content", ""),
                        "done": done,
                        "metadata": chunk.get("metadata"),
                    }
                    await websocket.send_json(payload)

                duration = time.perf_counter() - start
                model_inference_latency.labels(model_name="financial_analyst").observe(duration)

            elif action == "close":
                await websocket.send_json({"action": "closing"})
                break

            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    except Exception as exc:
        logger.exception("WebSocket error: session=%s", session_id)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
    finally:
        active_ws_connections.dec()
        logger.info("WebSocket cleaned up: session=%s user=%s", session_id, user_id)
