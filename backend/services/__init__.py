"""Services package for the AI Financial Brain."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GraphState",
    "build_orchestration_graph",
    "Orchestrator",
    "QueryRouter",
    "ResponseSynthesizer",
    "AgentFactory",
]


def __getattr__(name: str) -> Any:
    if name == "GraphState":
        return import_module("backend.services.graph_state").GraphState
    if name in {"Orchestrator", "build_orchestration_graph"}:
        module = import_module("backend.services.orchestrator")
        return getattr(module, name)
    if name == "QueryRouter":
        return import_module("backend.services.query_router").QueryRouter
    if name == "ResponseSynthesizer":
        return import_module("backend.services.synthesizer").ResponseSynthesizer
    if name == "AgentFactory":
        return import_module("backend.services.agent_factory").AgentFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
