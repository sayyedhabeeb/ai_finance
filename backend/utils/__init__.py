"""
AI Financial Brain — Autonomous Analyst + Personal CFO
========================================================

A multi-agent, dual-RAG, ML-powered financial intelligence system.

## Quick Start

1. Clone and install dependencies:
   ```bash
   cd backend
   pip install -e ".[dev]"
   ```

2. Copy and configure environment:
   ```bash
   cp ../deployments/docker/.env.example .env
   # Edit .env with your API keys
   ```

3. Start infrastructure (Docker):
   ```bash
   cd ../deployments/docker
   docker compose up -d postgres redis weaviate
   ```

4. Initialize database:
   ```bash
   python -c "from backend.database.schemas.init_postgres import init_database; import asyncio; asyncio.run(init_database())"
   ```

5. Run the server:
   ```bash
   python main.py
   ```

## Architecture

- 6 AI Agents: Personal CFO, Market Analyst, News & Sentiment, Risk Analyst, Portfolio Manager, Critic
- LangGraph orchestration with supervisor pattern and critic feedback loop
- Dual-layer RAG: Weaviate (market) + pgvector (personal finance)
- ML Models: PatchTST, FinBERT, GARCH, Isolation Forest
- Memory: Redis (session) + pgvector (semantic) + Mem0 (long-term) + Episodic
- MLOps: MLflow + Prefect + Evidently AI
- API: FastAPI with WebSocket streaming
- Frontend: Next.js + Tailwind CSS + Recharts
"""

__version__ = "1.0.0"
