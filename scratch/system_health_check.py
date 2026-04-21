import os
import sys
import asyncio

sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()


async def run_checks():
    # Check 7: Full orchestrator query
    from backend.services.orchestrator import Orchestrator
    orch = Orchestrator.from_settings()
    result = await orch.process_query("What is a good ETF for long-term investment?")
    has_resp = bool(result.get("response") or result.get("answer"))
    print(f"[ORCHESTRATOR_QUERY] Got response={has_resp}")
    print(f"[ORCHESTRATOR_QUERY] Query type={result.get('query_type', 'N/A')}")
    print(f"[ORCHESTRATOR_QUERY] Agents used={list(result.get('agent_results', {}).keys())}")
    preview = str(result.get("response") or result.get("answer") or "")[:120]
    print(f"[ORCHESTRATOR_QUERY] Preview: {preview}")
    print()

    # Check 8: API endpoint connectivity
    import httpx
    try:
        r = httpx.post(
            "http://localhost:8000/api/v1/query",
            json={"query": "hi", "user_id": "test", "session_id": "test-sess"},
            timeout=20,
        )
        print(f"[API_ENDPOINT] POST /api/v1/query status={r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"[API_ENDPOINT] answer preview={str(data.get('answer', ''))[:100]}")
        else:
            print(f"[API_ENDPOINT] ERROR body={r.text[:300]}")
    except Exception as e:
        print(f"[API_ENDPOINT] Connection error (is server running?): {e}")

    # Check 9: /health endpoint
    try:
        r2 = httpx.get("http://localhost:8000/health", timeout=5)
        print(f"[HEALTH] GET /health status={r2.status_code}")
        print(f"[HEALTH] body={r2.text[:200]}")
    except Exception as e:
        print(f"[HEALTH] ERROR: {e}")


asyncio.run(run_checks())
