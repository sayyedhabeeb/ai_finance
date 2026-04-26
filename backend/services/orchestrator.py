"""Orchestrator — the main LangGraph workflow that coordinates all agents.

Graph topology::

    START
      │
      ▼
    classify_query_node
      │
      ▼
    activate_agents_node
      │
      ▼
    execute_agents_node          ← runs parallel groups via asyncio.gather,
      │                             then sequential agents one-by-one
      ▼
    aggregate_results_node
      │
      ├─ critic_enabled? ─── No ──► synthesize_response_node ──► END
      │                         ▲
      │  Yes                    │
      ▼                         │
    critic_node                  │
      │                         │
      ├─ score ≥ threshold ─────┘
      │
      │  score < threshold && revisions < max
      ▼
    revision_node ──► back to activate_agents_node
"""
from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from typing import Any, AsyncIterator

import structlog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.services.agent_factory import AgentFactory
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph
else:
    # At runtime we don't need the type; just use Any for the annotation.
    CompiledGraph: type = None  # type: ignore[assignment,misc]

from backend.config.schemas import AgentResult, AgentTask, AgentType, QueryType
from backend.services.graph_state import GraphState

logger = structlog.get_logger(__name__)


def _get_agent_factory_cls():
    from backend.services.agent_factory import AgentFactory

    return AgentFactory


def _build_llm_from_settings_lazy(settings: Any) -> Any:
    from backend.services.agent_factory import _build_llm_from_settings

    return _build_llm_from_settings(settings)


def _get_query_router_cls():
    from backend.services.query_router import QueryRouter

    return QueryRouter


def _get_synthesizer_cls():
    from backend.services.synthesizer import ResponseSynthesizer

    return ResponseSynthesizer


def _is_llm_unavailable_response(text: str) -> bool:
    from backend.services.llm_service import is_llm_unavailable_response

    return is_llm_unavailable_response(text)


# ════════════════════════════════════════════════════════════════
# Shared dependency container (singleton per process)
# ════════════════════════════════════════════════════════════════


class _Dependencies:
    """Holds shared singletons injected into every node via module-level getter."""

    def __init__(
        self,
        agent_factory: "AgentFactory | None" = None,
        query_router: Any = None,
        synthesizer: Any = None,
    ) -> None:
        agent_factory_cls = _get_agent_factory_cls()
        query_router_cls = _get_query_router_cls()
        synthesizer_cls = _get_synthesizer_cls()

        self.agent_factory = agent_factory or agent_factory_cls.from_settings()
        self.query_router = query_router or query_router_cls()
        self.synthesizer = synthesizer or synthesizer_cls()


_deps: _Dependencies | None = None


def _get_deps() -> _Dependencies:
    """Return the global dependency container (lazy-initialised)."""
    global _deps  # noqa: PLW0603
    if _deps is None:
        _deps = _Dependencies()
    return _deps


def _reset_deps(deps: _Dependencies | None = None) -> None:
    """Reset the global dependency container.  Used primarily in tests."""
    global _deps  # noqa: PLW0603
    _deps = deps


# ════════════════════════════════════════════════════════════════
# Agent-name → AgentType mapping
# ════════════════════════════════════════════════════════════════

_AGENT_NAME_TO_TYPE: dict[str, AgentType] = {
    "personal_cfo": AgentType.PERSONAL_CFO,
    "market_analyst": AgentType.MARKET_ANALYST,
    "news_sentiment": AgentType.NEWS_SENTIMENT,
    "risk_analyst": AgentType.RISK_ANALYST,
    "portfolio_manager": AgentType.PORTFOLIO_MANAGER,
    "critic": AgentType.CRITIC,
}

_ALL_AGENT_NAMES: list[str] = [
    "personal_cfo",
    "market_analyst",
    "news_sentiment",
    "risk_analyst",
    "portfolio_manager",
]


# ════════════════════════════════════════════════════════════════
# Agent execution helpers
# ════════════════════════════════════════════════════════════════


async def _execute_single_agent(
    agent_name: str,
    state: GraphState,
    merged_results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Execute a single agent and return ``{"agent_results": {name: result}}``.

    The *merged_results* dict is passed as context so that sequential agents
    can see results from earlier (parallel) agents.
    """
    deps = _get_deps()
    agent_type = _AGENT_NAME_TO_TYPE[agent_name]
    agent = deps.agent_factory.create_agent(agent_type)

    query = state["user_query"]

    # Collect critic feedback for revision passes
    critic_feedback = ""
    if state.get("critic_result"):
        critic_result = state["critic_result"]
        fb_list = critic_result.get("recommendations", [])
        if not fb_list:
            fb_list = critic_result.get("data", {}).get("feedback", [])
        critic_feedback = "; ".join(fb_list)

    task = AgentTask(
        agent_type=agent_type,
        query=query,
        context={
            "user_id": state.get("user_id", ""),
            "session_id": state.get("session_id", ""),
            "query_type": state.get("query_type", ""),
            "entities": state.get("metadata", {}).get("entities", {}),
            "agent_results": merged_results,
            "revision_count": state.get("revision_count", 0),
            "critic_feedback": critic_feedback,
        },
    )

    t0 = time.perf_counter()
    try:
        result: AgentResult = await agent.execute(task)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(
            "agent_execution.failed",
            agent=agent_name,
            error=str(exc),
            elapsed=elapsed,
            traceback=traceback.format_exc(),
        )
        return {
            agent_name: {
                "task_id": str(task.task_id),
                "agent_type": agent_type.value,
                "output": f"Agent {agent_name} failed: {exc}",
                "confidence": 0.0,
                "processing_time": elapsed,
                "success": False,
                "summary": f"Agent {agent_name} encountered an error: {exc}",
                "sources": [],
                "recommendations": [],
                "data": {"error": str(exc)},
            }
        }

    elapsed = time.perf_counter() - t0
    logger.info(
        "agent_execution.completed",
        agent=agent_name,
        confidence=result.confidence,
        elapsed=elapsed,
    )

    result_dict: dict[str, Any] = {
        "task_id": str(result.task_id),
        "agent_type": result.agent_type.value,
        "output": result.output,
        "confidence": result.confidence,
        "processing_time": result.processing_time + elapsed,
        "success": True,
        "summary": result.output,
        "sources": result.metadata.get("sources", []),
        "recommendations": result.metadata.get("recommendations", []),
        "data": {
            k: v
            for k, v in result.metadata.items()
            if k not in ("sources", "recommendations")
        },
    }

    return {agent_name: result_dict}


# ════════════════════════════════════════════════════════════════
# Node functions
# ════════════════════════════════════════════════════════════════
# Each node receives the full GraphState and returns a partial dict
# that LangGraph merges into the running state.
# ════════════════════════════════════════════════════════════════


# ── 1. Classify ────────────────────────────────────────────────


def classify_query_node(state: GraphState) -> dict[str, Any]:
    """Node 1 — classify the user query and extract entities."""
    deps = _get_deps()
    query = state["user_query"]
    t0 = time.perf_counter()

    logger.info("node.classify_query.start", query=query[:120])

    try:
        query_type = deps.query_router.classify_query(query)
        entities = deps.query_router.extract_entities(query)
        elapsed = time.perf_counter() - t0

        logger.info(
            "node.classify_query.done",
            query_type=query_type.value,
            entities=entities,
            elapsed=elapsed,
        )

        meta = dict(state.get("metadata", {}))
        meta["classification_time"] = elapsed
        meta["entities"] = entities

        return {
            "query_type": query_type.value,
            "metadata": meta,
            "error": None,
        }
    except Exception as exc:
        logger.error("node.classify_query.error", error=str(exc))
        return {
            "query_type": "general",
            "error": f"classify_query_node: {exc}",
        }


# ── 2. Activate agents ────────────────────────────────────────


def activate_agents_node(state: GraphState) -> dict[str, Any]:
    """Node 2 — determine which agents to run and in what order."""
    deps = _get_deps()
    query_type_str = state.get("query_type", "general")
    query = state["user_query"]
    t0 = time.perf_counter()

    logger.info("node.activate_agents.start", query_type=query_type_str)

    try:
        query_type = QueryType(query_type_str)
    except ValueError:
        query_type = QueryType.GENERAL

    sequential, parallel_groups = deps.query_router.determine_active_agents(
        query_type, query
    )
    elapsed = time.perf_counter() - t0

    # Build flat active_agents list (parallel first, then sequential)
    active: list[str] = []
    for group in parallel_groups:
        active.extend(group)
    active.extend(sequential)

    logger.info(
        "node.activate_agents.done",
        active_agents=active,
        parallel_groups=parallel_groups,
        sequential=sequential,
        elapsed=elapsed,
    )

    meta = dict(state.get("metadata", {}))
    meta["activation_time"] = elapsed

    return {
        "active_agents": active,
        "sequential_agents": sequential,
        "parallel_groups": parallel_groups,
        "metadata": meta,
    }


# ── 3. Execute agents (parallel + sequential) ──────────────────


async def execute_agents_node(state: GraphState) -> dict[str, Any]:
    """Node 3 — execute all active agents with proper parallel/sequential ordering.

    *Phase 1 (parallel):* Agents within each ``parallel_group`` run
    concurrently via ``asyncio.gather``.  Multiple groups run one after
    another (each subsequent group can see results from earlier groups).

    *Phase 2 (sequential):* Sequential agents (e.g. portfolio_manager
    after market analysis) run one-by-one.  Each receives the merged
    results of all parallel agents as context.
    """
    parallel_groups: list[list[str]] = state.get("parallel_groups", [])
    sequential: list[str] = state.get("sequential_agents", [])
    existing_results = dict(state.get("agent_results", {}))

    t0 = time.perf_counter()
    logger.info(
        "node.execute_agents.start",
        parallel_groups=parallel_groups,
        sequential=sequential,
    )

    # ── Phase 1: Parallel groups ───────────────────────────────
    for group_idx, group in enumerate(parallel_groups):
        coros = [
            _execute_single_agent(name, state, existing_results)
            for name in group
        ]
        group_results = await asyncio.gather(*coros, return_exceptions=True)

        for name, result in zip(group, group_results):
            if isinstance(result, BaseException):
                logger.error(
                    "node.execute_agents.parallel_failure",
                    agent=name,
                    error=str(result),
                    group=group_idx,
                )
                existing_results[name] = {
                    "task_id": "",
                    "agent_type": name,
                    "output": f"Agent {name} failed: {result}",
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "success": False,
                    "summary": f"Agent {name} encountered an error.",
                    "sources": [],
                    "recommendations": [],
                    "data": {"error": str(result)},
                }
            else:
                existing_results.update(result)

    # ── Phase 2: Sequential agents ─────────────────────────────
    for name in sequential:
        try:
            result = await _execute_single_agent(name, state, existing_results)
            existing_results.update(result)
        except Exception as exc:
            logger.error(
                "node.execute_agents.sequential_failure",
                agent=name,
                error=str(exc),
            )
            existing_results[name] = {
                "task_id": "",
                "agent_type": name,
                "output": f"Agent {name} failed: {exc}",
                "confidence": 0.0,
                "processing_time": 0.0,
                "success": False,
                "summary": f"Agent {name} encountered an error.",
                "sources": [],
                "recommendations": [],
                "data": {"error": str(exc)},
            }

    elapsed = time.perf_counter() - t0

    meta = dict(state.get("metadata", {}))
    meta["agent_execution_time"] = elapsed

    logger.info(
        "node.execute_agents.done",
        agents_completed=len(existing_results),
        elapsed=elapsed,
    )

    return {
        "agent_results": existing_results,
        "metadata": meta,
    }


# ── 4. Aggregate results ──────────────────────────────────────


def aggregate_results_node(state: GraphState) -> dict[str, Any]:
    """Node 4 — merge all agent results into a unified structure.

    Computes aggregate confidence, deduplicates sources and
    recommendations across all agent outputs.
    """
    t0 = time.perf_counter()
    agent_results: dict[str, Any] = state.get("agent_results", {})

    logger.info(
        "node.aggregate_results.start", agent_count=len(agent_results)
    )

    total_conf: float = 0.0
    conf_count = 0
    all_sources: list[str] = []
    all_recommendations: list[str] = []
    source_seen: set[str] = set()
    rec_seen: set[str] = set()
    llm_unavailable = False

    for agent_name, result in agent_results.items():
        if not isinstance(result, dict):
            continue
        if result.get("success"):
            total_conf += result.get("confidence", 0.0)
            conf_count += 1

        summary_text = str(result.get("summary", ""))
        output_text = str(result.get("output", ""))
        if _is_llm_unavailable_response(summary_text) or _is_llm_unavailable_response(output_text):
            llm_unavailable = True

        for src in result.get("sources", []):
            if src not in source_seen:
                source_seen.add(src)
                all_sources.append(src)
        for rec in result.get("recommendations", []):
            if rec not in rec_seen:
                rec_seen.add(rec)
                all_recommendations.append(rec)

    confidence = total_conf / conf_count if conf_count else 0.0
    if llm_unavailable:
        confidence = min(confidence, 0.2)
    elapsed = time.perf_counter() - t0

    meta = dict(state.get("metadata", {}))
    meta["aggregation_time"] = elapsed
    meta["llm_unavailable"] = llm_unavailable

    logger.info(
        "node.aggregate_results.done",
        confidence=confidence,
        source_count=len(all_sources),
        llm_unavailable=llm_unavailable,
        elapsed=elapsed,
    )

    return {
        "confidence": confidence,
        "sources": all_sources,
        "recommendations": all_recommendations,
        "metadata": meta,
    }


# ── 5. Critic ─────────────────────────────────────────────────


async def critic_node(state: GraphState) -> dict[str, Any]:
    """Node 5 — run the critic agent to evaluate aggregated results."""
    t0 = time.perf_counter()
    logger.info("node.critic.start")

    result_map = await _execute_single_agent("critic", state, state.get("agent_results", {}))
    critic_data = result_map.get("critic", {})
    elapsed = time.perf_counter() - t0

    meta = dict(state.get("metadata", {}))
    meta["critic_time"] = elapsed

    logger.info(
        "node.critic.done",
        score=critic_data.get("data", {}).get("overall_score", 0),
        elapsed=elapsed,
    )

    return {
        "critic_result": critic_data,
        "metadata": meta,
    }


# ── 6. Revision ───────────────────────────────────────────────


def revision_node(state: GraphState) -> dict[str, Any]:
    """Node 6 — apply critic feedback and increment revision counter."""
    revision_count = state.get("revision_count", 0) + 1
    max_revisions = state.get("max_revisions", 3)

    logger.info(
        "node.revision",
        revision_count=revision_count,
        max_revisions=max_revisions,
    )

    feedback = ""
    if state.get("critic_result"):
        fb_list = state["critic_result"].get("recommendations", [])
        if not fb_list:
            fb_list = state["critic_result"].get("data", {}).get("feedback", [])
        feedback = "; ".join(fb_list)

    meta = dict(state.get("metadata", {}))
    meta["last_revision_feedback"] = feedback

    return {
        "revision_count": revision_count,
        "metadata": meta,
    }


# ── 7. Synthesize final response ──────────────────────────────


def synthesize_response_node(state: GraphState) -> dict[str, Any]:
    """Node 7 — produce the final natural-language response."""
    deps = _get_deps()
    query = state["user_query"]
    agent_results = state.get("agent_results", {})
    t0 = time.perf_counter()

    logger.info("node.synthesize_response.start")

    try:
        context: dict[str, Any] = {
            "query_type": state.get("query_type", "general"),
            "user_id": state.get("user_id", ""),
            "revision_count": state.get("revision_count", 0),
            "confidence": state.get("confidence", 0.0),
        }
        if state.get("critic_result"):
            critic_result = state["critic_result"]
            feedback = critic_result.get("recommendations", [])
            if not feedback:
                feedback = critic_result.get("data", {}).get("feedback", [])
            context["critic_feedback"] = "; ".join(feedback)

        final_response = deps.synthesizer.synthesize(query, agent_results, context)
        elapsed = time.perf_counter() - t0

        # Generate follow-up suggestions
        follow_ups = deps.synthesizer.generate_follow_up_suggestions(
            query, final_response, context.get("query_type", "general")
        )

        meta = dict(state.get("metadata", {}))
        meta["synthesis_time"] = elapsed
        total = (
            meta.get("classification_time", 0)
            + meta.get("activation_time", 0)
            + meta.get("agent_execution_time", 0)
            + meta.get("aggregation_time", 0)
            + meta.get("critic_time", 0)
            + elapsed
        )
        meta["total_time"] = total

        logger.info("node.synthesize_response.done", elapsed=elapsed)

        return {
            "final_response": final_response,
            "metadata": meta,
            "error": None,
        }
    except Exception as exc:
        logger.error("node.synthesize_response.error", error=str(exc))
        return {
            "final_response": (
                f"An error occurred while generating the response: {exc}"
            ),
            "error": f"synthesize_response_node: {exc}",
        }


# ════════════════════════════════════════════════════════════════
# Conditional routing helpers
# ════════════════════════════════════════════════════════════════


def _should_run_critic(state: GraphState) -> str:
    """Conditional edge from ``aggregate_results_node``.

    Routes to critic when critic is enabled, otherwise skips to synthesis.
    """
    # Disabled: Groq free-tier 429 fix (re-enable after upgrade)
    return "synthesize_response"

def _route_by_critique(state: GraphState) -> str:
    """Conditional edge from ``critic_node``.

    Returns:
    * ``"revision"`` — if quality score is below threshold and revisions remain.
    * ``"synthesize_response"`` — otherwise (response is good enough or max
      revisions reached).
    """
    # Disabled: Groq free-tier 429 fix (re-enable after upgrade)
    return "synthesize_response"

# ════════════════════════════════════════════════════════════════
# Graph builder
# ════════════════════════════════════════════════════════════════


def build_orchestration_graph() -> CompiledGraph:
    """Build and compile the full agent orchestration graph.

    Returns a :class:`CompiledGraph` ready for ``.invoke()`` / ``.ainvoke()``
    / ``.astream()`` calls.

    Graph structure::

        START
          │
          ▼
        classify_query_node ──► activate_agents_node ──► execute_agents_node
          │                                                    │
          │                                                    ▼
          │                                            aggregate_results_node
          │                                              │              │
          │                                     critic_enabled?   critic_enabled?
          │                                         │ No              │ Yes
          │                                         ▼                 ▼
          │                              synthesize_response    critic_node
          │                                       ▲            │         │
          │                                       │     score<threshold  score≥threshold
          │                                       │             │             │
          │                                       │         revision_node      │
          │                                       │             │              │
          │                                       └─────────────┘              │
          │                                       (back to activate_agents)    │
          │                                                                    │
          └──── END ◄────────────────── synthesize_response_node ◄─────────────┘
    """
    from langgraph.graph import END, StateGraph

    graph = StateGraph(GraphState)

    # ── Add all nodes ───────────────────────────────────────────
    graph.add_node("classify_query", classify_query_node)
    graph.add_node("activate_agents", activate_agents_node)
    graph.add_node("execute_agents", execute_agents_node)
    graph.add_node("aggregate_results", aggregate_results_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revision", revision_node)
    graph.add_node("synthesize_response", synthesize_response_node)

    # ── Set entry point ─────────────────────────────────────────
    graph.set_entry_point("classify_query")

    # ── Fixed edges ─────────────────────────────────────────────
    graph.add_edge("classify_query", "activate_agents")
    graph.add_edge("activate_agents", "execute_agents")
    graph.add_edge("execute_agents", "aggregate_results")

    # aggregate_results → critic or synthesize (conditional)
    graph.add_conditional_edges(
        "aggregate_results",
        _should_run_critic,
        {
            "critic": "critic",
            "synthesize_response": "synthesize_response",
        },
    )

    # critic → revision or synthesize (conditional)
    graph.add_conditional_edges(
        "critic",
        _route_by_critique,
        {
            "revision": "revision",
            "synthesize_response": "synthesize_response",
        },
    )

    # revision → back to activate_agents for another pass
    graph.add_edge("revision", "activate_agents")

    # synthesize → END
    graph.add_edge("synthesize_response", END)

    # ── Compile ─────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info("orchestration_graph.compiled")
    return compiled


# ════════════════════════════════════════════════════════════════
# Orchestrator class
# ════════════════════════════════════════════════════════════════


class Orchestrator:
    """High-level wrapper around the compiled LangGraph workflow.

    Provides:
    * ``process_query`` — async execution returning a result dict.
    * ``process_query_stream`` — async generator yielding streaming events.
    * Configurable critic loop, timeout, and error recovery.
    """

    def __init__(
        self,
        *,
        # Disabled: Groq free-tier 429 fix (re-enable after upgrade to Dev/Pro tier)
        # critic_enabled: bool = True,
        critic_enabled: bool = False,
        # max_revisions: int = 3,
        max_revisions: int = 0,
        min_quality_score: float = 0.75,
        timeout_seconds: float = 120.0,
        agent_factory: "AgentFactory | None" = None,
    ) -> None:
        print("DEBUG: Orchestrator initializing...")
        # Initialise global deps
        from backend.config.settings import get_settings
        settings = get_settings()
        
        # Build LLM from settings to pass to synthesizer and factory
        _llm = _build_llm_from_settings_lazy(settings)
        print("DEBUG: Orchestrator dependencies loaded successfully")
        
        # Task 5: Add a startup log to confirm LLM loaded
        import logging
        logging.getLogger(__name__).info(f"Orchestrator LLM loaded: {type(_llm).__name__ if _llm else 'NONE — CHECK GROQ_API_KEY'}")

        agent_factory_cls = _get_agent_factory_cls()
        query_router_cls = _get_query_router_cls()
        synthesizer_cls = _get_synthesizer_cls()

        factory = agent_factory or agent_factory_cls(llm=_llm)
        router = query_router_cls(llm=_llm)
        synthesizer = synthesizer_cls(llm=_llm)
        _reset_deps(
            _Dependencies(
                agent_factory=factory,
                query_router=router,
                synthesizer=synthesizer,
            )
        )

        self._graph: CompiledGraph = build_orchestration_graph()
        self._critic_enabled = critic_enabled
        self._max_revisions = max_revisions
        self._min_quality_score = min_quality_score
        self._timeout_seconds = timeout_seconds

    # ── Async API ───────────────────────────────────────────────

    async def process_query(
        self,
        user_query: str,
        user_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """Execute the full orchestration pipeline and return a result dict.

        Parameters
        ----------
        user_query:
            The user's natural-language question.
        user_id:
            Optional user identifier for personalisation.
        session_id:
            Optional session identifier for conversation continuity.

        Returns
        -------
        dict
            Structured result with keys: ``response``, ``confidence``,
            ``sources``, ``recommendations``, ``agent_results``,
            ``query_type``, ``execution_time``, ``revision_count``,
            ``error``, ``metadata``.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        t0 = time.perf_counter()

        initial_state: GraphState = {
            "user_query": user_query,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [],
            "agent_results": {},
            "active_agents": [],
            "sequential_agents": [],
            "parallel_groups": [],
            "critic_result": None,
            "revision_count": 0,
            "max_revisions": self._max_revisions,
            "final_response": "",
            "confidence": 0.0,
            "sources": [],
            "recommendations": [],
            "error": None,
            "metadata": {
                "critic_enabled": self._critic_enabled,
                "min_quality_score": self._min_quality_score,
            },
        }

        try:
            final_state = await asyncio.wait_for(
                self._graph.ainvoke(initial_state),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error("orchestrator.timeout", timeout=self._timeout_seconds)
            return {
                "response": "The request timed out. Please try a more specific query.",
                "confidence": 0.0,
                "sources": [],
                "recommendations": [],
                "agent_results": {},
                "query_type": "",
                "execution_time": time.perf_counter() - t0,
                "revision_count": 0,
                "error": "TIMEOUT",
                "metadata": {"timeout_seconds": self._timeout_seconds},
            }
        except Exception as exc:
            logger.error("orchestrator.unhandled_error", error=str(exc))
            return {
                "response": f"An unexpected error occurred: {exc}",
                "confidence": 0.0,
                "sources": [],
                "recommendations": [],
                "agent_results": {},
                "query_type": "",
                "execution_time": time.perf_counter() - t0,
                "revision_count": 0,
                "error": str(exc),
                "metadata": {},
            }

        execution_time = time.perf_counter() - t0

        return {
            "response": final_state.get("final_response", ""),
            "confidence": final_state.get("confidence", 0.0),
            "sources": final_state.get("sources", []),
            "recommendations": final_state.get("recommendations", []),
            "agent_results": final_state.get("agent_results", {}),
            "query_type": final_state.get("query_type", ""),
            "execution_time": execution_time,
            "revision_count": final_state.get("revision_count", 0),
            "error": final_state.get("error"),
            "metadata": final_state.get("metadata", {}),
        }

    # ── Streaming API ───────────────────────────────────────────

    async def process_query_stream(
        self,
        user_query: str,
        user_id: str = "",
        session_id: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator that yields intermediate results as the graph executes.

        Each yielded dict has a ``type`` key (one of ``"classification"``,
        ``"agent_start"``, ``"agent_result"``, ``"aggregate"``, ``"critic"``,
        ``"revision"``, ``"final"``, ``"error"``) and corresponding data.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        initial_state: GraphState = {
            "user_query": user_query,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [],
            "agent_results": {},
            "active_agents": [],
            "sequential_agents": [],
            "parallel_groups": [],
            "critic_result": None,
            "revision_count": 0,
            "max_revisions": self._max_revisions,
            "final_response": "",
            "confidence": 0.0,
            "sources": [],
            "recommendations": [],
            "error": None,
            "metadata": {
                "critic_enabled": self._critic_enabled,
                "min_quality_score": self._min_quality_score,
            },
        }

        try:
            async for event in self._graph.astream(
                initial_state, stream_mode="updates"
            ):
                if not isinstance(event, dict):
                    continue

                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        continue

                    if node_name == "classify_query":
                        yield {
                            "type": "classification",
                            "query_type": node_output.get("query_type", "general"),
                            "entities": node_output.get("metadata", {}).get("entities", {}),
                        }

                    elif node_name == "activate_agents":
                        yield {
                            "type": "activation",
                            "active_agents": node_output.get("active_agents", []),
                            "parallel_groups": node_output.get("parallel_groups", []),
                            "sequential_agents": node_output.get("sequential_agents", []),
                        }

                    elif node_name == "execute_agents":
                        agent_results = node_output.get("agent_results", {})
                        for agent_name, result in agent_results.items():
                            if isinstance(result, dict):
                                yield {
                                    "type": "agent_result",
                                    "agent": agent_name,
                                    "data": {
                                        "summary": result.get("summary", ""),
                                        "confidence": result.get("confidence", 0.0),
                                        "success": result.get("success", True),
                                        "sources": result.get("sources", []),
                                    },
                                }

                    elif node_name == "aggregate_results":
                        yield {
                            "type": "aggregate",
                            "confidence": node_output.get("confidence", 0.0),
                            "source_count": len(node_output.get("sources", [])),
                            "recommendation_count": len(node_output.get("recommendations", [])),
                        }

                    elif node_name == "critic":
                        critic_result = node_output.get("critic_result", {})
                        yield {
                            "type": "critic",
                            "result": critic_result,
                        }

                    elif node_name == "revision":
                        yield {
                            "type": "revision",
                            "revision_count": node_output.get("revision_count", 0),
                        }

                    elif node_name == "synthesize_response":
                        yield {
                            "type": "final",
                            "response": node_output.get("final_response", ""),
                            "metadata": node_output.get("metadata", {}),
                        }

        except Exception as exc:
            logger.error("orchestrator.stream_error", error=str(exc))
            yield {
                "type": "error",
                "error": str(exc),
            }

    # ── Convenience factory ─────────────────────────────────────

    @classmethod
    def from_settings(cls) -> Orchestrator:
        """Build an :class:`Orchestrator` using application settings."""
        from backend.config.settings import get_settings

        settings = get_settings()

        return cls(
            critic_enabled=getattr(settings, "critic_enabled", True),
            max_revisions=getattr(settings, "critic_max_revisions", 3),
            min_quality_score=getattr(settings, "min_quality_score", 0.75),
        )

