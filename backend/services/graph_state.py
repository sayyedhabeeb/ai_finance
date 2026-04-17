"""Graph State — the single TypedDict that flows through every node.

Uses ``Annotated`` with custom reducers so that when multiple agent nodes
write to the same keys concurrently (via ``Send`` or ``asyncio.gather``),
their outputs are merged correctly instead of last-write-wins.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


# ────────────────────────────────────────────────────────────────
# Custom reducers
# ────────────────────────────────────────────────────────────────


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge two dicts — *right* wins on key collisions."""
    merged = dict(left)
    merged.update(right)
    return merged


def _extend_unique(left: list[Any], right: list[Any]) -> list[Any]:
    """Append items from *right* that are not already in *left*."""
    seen: set[Any] = set(left) if left else set()
    out = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _merge_metadata(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Merge metadata dicts, preserving numeric accumulators."""
    merged = dict(left)
    for k, v in right.items():
        if k in merged and isinstance(merged[k], (int, float)) and isinstance(v, (int, float)):
            merged[k] = merged[k] + v  # type: ignore[assignment]
        else:
            merged[k] = v
    return merged


# ────────────────────────────────────────────────────────────────
# GraphState
# ────────────────────────────────────────────────────────────────


class GraphState(TypedDict, total=False):
    """Central state passed between all nodes in the orchestration graph.

    Every node **reads** from and **writes** to this dictionary.  Because
    LangGraph performs a shallow merge after each node returns, nodes only
    need to include the keys they wish to update in their return value.

    Keys annotated with custom reducers ensure that parallel branches merge
    their results correctly (dict merge, list dedup) instead of overwriting.
    """

    # ── Input ─────────────────────────────────────────────────
    user_query: str
    user_id: str
    session_id: str

    # ── Conversation ──────────────────────────────────────────
    messages: Annotated[list[dict[str, Any]], operator.add]

    # ── Classification & Routing ─────────────────────────────
    query_type: str
    active_agents: list[str]
    sequential_agents: list[str]
    parallel_groups: list[list[str]]

    # ── Agent Outputs ─────────────────────────────────────────
    agent_results: Annotated[dict[str, Any], _merge_dicts]

    # ── Critic Loop ───────────────────────────────────────────
    critic_result: Optional[dict[str, Any]]
    revision_count: int
    max_revisions: int

    # ── Final Output ──────────────────────────────────────────
    final_response: str
    confidence: float
    sources: Annotated[list[str], _extend_unique]
    recommendations: Annotated[list[str], _extend_unique]

    # ── Error Handling ────────────────────────────────────────
    error: Optional[str]

    # ── Miscellaneous ─────────────────────────────────────────
    metadata: Annotated[dict[str, Any], _merge_metadata]
