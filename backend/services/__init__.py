"""Services package — LangGraph orchestration for the AI Financial Brain."""
from __future__ import annotations

from services.agent_factory import AgentFactory
from services.graph_state import GraphState
from services.orchestrator import Orchestrator, build_orchestration_graph
from services.query_router import QueryRouter
from services.synthesizer import ResponseSynthesizer

__all__ = [
    # Graph state
    "GraphState",
    # Graph builder
    "build_orchestration_graph",
    # Main entry point
    "Orchestrator",
    # Utilities
    "QueryRouter",
    "ResponseSynthesizer",
    "AgentFactory",
]
