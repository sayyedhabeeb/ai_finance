"""Response Synthesizer — produces the final unified answer from agent outputs."""
from __future__ import annotations

from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = structlog.get_logger(__name__)

# ────────────────────────────────────────────────────────────────
# Prompt template for final synthesis
# ────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM_PROMPT = """\
You are the **Synthesizer** for the AI Financial Brain.  Your job is to merge \
the outputs of multiple specialised financial agents into a single, coherent, \
and actionable response for the user.

## Rules
1. **Accuracy first** — never hallucinate numbers. Only cite figures the \
   agents actually produced.
2. **Structure** — use clear headings, bullet lists, and short paragraphs.
3. **Actionability** — end with a concise "Key Takeaways & Next Steps" section.
4. **Confidence** — explicitly state your overall confidence level (High / \
   Medium / Low) based on the weighted average of agent confidences.
5. **Sources** — list every source cited by the agents so the user can verify.
6. **Tone** — professional yet approachable.  Avoid jargon unless the user \
   used it first.
7. **Follow-ups** — suggest 2-3 relevant follow-up questions the user might \
   want to ask.

## Agent outputs
{agent_outputs}
"""

_SYNTHESIS_USER_PROMPT = """\
## User's original question
{query}

## Additional context
{context}

## Instructions
Synthesise the agent outputs above into a final answer that directly addresses \
the user's question.  Include:
- An executive summary (2-3 sentences)
- Detailed analysis (grouped by theme)
- Confidence level with brief justification
- Sources used
- Key takeaways & next steps
- Suggested follow-up questions
"""

_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYNTHESIS_SYSTEM_PROMPT),
        ("human", _SYNTHESIS_USER_PROMPT),
    ]
)


# ────────────────────────────────────────────────────────────────
# Follow-up suggestion prompt (LLM-powered)
# ────────────────────────────────────────────────────────────────

_FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial assistant.  Given the user's question and the "
            "system's answer, suggest exactly 3 relevant follow-up questions the "
            "user might want to ask next.  Return ONLY a JSON array of strings, "
            "nothing else.",
        ),
        ("human", "Question: {query}\n\nAnswer: {response}"),
    ]
)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def _format_agent_outputs(agent_results: dict[str, Any]) -> str:
    """Serialise ``agent_results`` into a human-readable string for the LLM."""
    parts: list[str] = []
    for agent_name, result in agent_results.items():
        if not isinstance(result, dict):
            continue
        parts.append(f"### {agent_name.replace('_', ' ').title()}")
        parts.append(f"- **Summary:** {result.get('summary', 'N/A')}")
        parts.append(f"- **Confidence:** {result.get('confidence', 0):.0%}")

        sources = result.get("sources", [])
        if sources:
            parts.append(f"- **Sources:** {', '.join(sources)}")

        recommendations = result.get("recommendations", [])
        if recommendations:
            rec_lines = "\n".join(f"  - {r}" for r in recommendations)
            parts.append(f"- **Recommendations:**\n{rec_lines}")

        # Include scalar data keys (not full payloads to stay within token limits)
        data = result.get("data", {})
        if data:
            data_summary = {
                k: v for k, v in data.items() if not isinstance(v, (dict, list))
            }
            if data_summary:
                parts.append(f"- **Key data:** {data_summary}")

        parts.append("")  # blank line between agents
    return "\n".join(parts)


def _compute_confidence(agent_results: dict[str, Any]) -> float:
    """Weighted average confidence across all successful agent results."""
    if not agent_results:
        return 0.0
    total_conf: float = 0.0
    count = 0
    for result in agent_results.values():
        if isinstance(result, dict) and result.get("success"):
            total_conf += result.get("confidence", 0.0)
            count += 1
    return total_conf / count if count else 0.0


def _collect_sources(agent_results: dict[str, Any]) -> list[str]:
    """Deduplicated, ordered source list from all agent results."""
    seen: set[str] = set()
    out: list[str] = []
    for result in agent_results.values():
        if not isinstance(result, dict):
            continue
        for src in result.get("sources", []):
            if src not in seen:
                seen.add(src)
                out.append(src)
    return out


def _collect_recommendations(agent_results: dict[str, Any]) -> list[str]:
    """Deduplicated recommendation list from all agent results."""
    seen: set[str] = set()
    out: list[str] = []
    for result in agent_results.values():
        if not isinstance(result, dict):
            continue
        for rec in result.get("recommendations", []):
            if rec not in seen:
                seen.add(rec)
                out.append(rec)
    return out


def _confidence_label(confidence: float) -> str:
    """Convert a 0-1 confidence score to a human-readable label."""
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.6:
        return "Medium"
    return "Low"


# ── Heuristic follow-up suggestions by query type ──────────────

_FOLLOW_UPS_MAP: dict[str, list[str]] = {
    "market_query": [
        "What are the key risk factors I should watch?",
        "How does this compare to the sector benchmark?",
        "Should I adjust my portfolio based on this outlook?",
    ],
    "portfolio_query": [
        "What is the expected risk (VaR) of this portfolio?",
        "How would this allocation perform in a market downturn?",
        "Can you show a comparison with a passive index fund approach?",
    ],
    "risk_query": [
        "How can I hedge the identified risks?",
        "What stress scenarios should I prepare for?",
        "How has my risk profile changed over the past quarter?",
    ],
    "personal_finance_query": [
        "How can I optimise my tax savings?",
        "Am I on track for my retirement goal?",
        "What insurance gaps should I address?",
    ],
    "general": [
        "Can you analyse this in more detail?",
        "What are the latest trends in this area?",
        "How does this affect my overall financial plan?",
    ],
}


# ────────────────────────────────────────────────────────────────
# ResponseSynthesizer
# ────────────────────────────────────────────────────────────────


class ResponseSynthesizer:
    """Synthesizes final response from multiple agent outputs.

    Two synthesis strategies are available:

    1. **LLM-based** (preferred) — sends all agent results to an LLM which
       produces a polished, structured answer.
    2. **Template-based** (fallback) — assembles a markdown report without
       calling an LLM, useful when no API key is configured or when the LLM
       call fails.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    # ── Public API ──────────────────────────────────────────────

    def synthesize(
        self,
        query: str,
        agent_results: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Produce a unified, formatted final response.

        Parameters
        ----------
        query:
            The original user query.
        agent_results:
            Mapping of agent name → result dict.
        context:
            Optional extra context (query_type, entities, revision_count, …).
        """
        ctx = context or {}
        query_type = ctx.get("query_type", "general")
        revision_count = ctx.get("revision_count", 0)
        critic_feedback = ctx.get("critic_feedback", "")

        if self._llm is not None:
            return self._synthesize_via_llm(
                query, agent_results, ctx, query_type, revision_count, critic_feedback
            )

        return self._synthesize_template(
            query, agent_results, query_type, revision_count, critic_feedback
        )

    def generate_follow_up_suggestions(
        self,
        query: str,
        response: str,
        query_type: str = "general",
    ) -> list[str]:
        """Return 2-3 suggested follow-up questions.

        Uses the LLM when available for context-aware suggestions, otherwise
        falls back to a static map keyed by query type.
        """
        if self._llm is not None:
            return self._follow_ups_via_llm(query, response)

        return list(_FOLLOW_UPS_MAP.get(query_type, _FOLLOW_UPS_MAP["general"]))

    # ── LLM-based synthesis ─────────────────────────────────────

    def _synthesize_via_llm(
        self,
        query: str,
        agent_results: dict[str, Any],
        ctx: dict[str, Any],
        query_type: str,
        revision_count: int,
        critic_feedback: str,
    ) -> str:
        agent_text = _format_agent_outputs(agent_results)
        confidence = _compute_confidence(agent_results)

        context_str = (
            f"Query type: {query_type}\n"
            f"Overall confidence: {confidence:.0%} ({_confidence_label(confidence)})\n"
        )
        if revision_count > 0:
            context_str += f"Revision round: {revision_count}\n"
        if critic_feedback:
            context_str += f"Critic feedback applied: {critic_feedback}\n"

        chain = _SYNTHESIS_PROMPT | self._llm
        try:
            response = chain.invoke({
                "query": query,
                "agent_outputs": agent_text,
                "context": context_str,
            })
            return response.content.strip()
        except Exception as exc:
            logger.error("synthesizer.llm_failed", error=str(exc))
            return self._synthesize_template(
                query, agent_results, query_type, revision_count, critic_feedback
            )

    # ── LLM-based follow-ups ────────────────────────────────────

    def _follow_ups_via_llm(self, query: str, response: str) -> list[str]:
        chain = _FOLLOW_UP_PROMPT | self._llm
        try:
            import json

            result = chain.invoke({"query": query, "response": response[:4000]})
            text = result.content.strip()
            if text.startswith("```"):
                import re
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)
            suggestions = json.loads(text)
            if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
                return suggestions[:3]
        except Exception as exc:
            logger.warning("synthesizer.follow_ups_llm_failed", error=str(exc))

        # Fallback to generic suggestions
        return list(_FOLLOW_UPS_MAP["general"])

    # ── Template-based synthesis (no LLM) ───────────────────────

    @staticmethod
    def _synthesize_template(
        query: str,
        agent_results: dict[str, Any],
        query_type: str,
        revision_count: int,
        critic_feedback: str,
    ) -> str:
        """Produce a readable response without calling an LLM."""
        confidence = _compute_confidence(agent_results)
        sources = _collect_sources(agent_results)
        recommendations = _collect_recommendations(agent_results)
        follow_ups = list(
            _FOLLOW_UPS_MAP.get(query_type, _FOLLOW_UPS_MAP["general"])
        )

        sections: list[str] = []

        # ── Header ──────────────────────────────────────────────
        sections.append("# AI Financial Brain — Analysis Report\n")
        sections.append(f"**Query:** {query}\n")
        sections.append(
            f"**Type:** {query_type.replace('_', ' ').title()}"
        )
        sections.append(
            f"**Confidence:** {_confidence_label(confidence)} ({confidence:.0%})"
        )
        if revision_count > 0:
            sections.append(f"**Revisions:** {revision_count}")
        sections.append("")

        # ── Executive summary ───────────────────────────────────
        sections.append("## Executive Summary\n")
        summary_parts: list[str] = []
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and result.get("summary"):
                summary_parts.append(result["summary"])
        sections.append(
            " ".join(summary_parts) if summary_parts
            else "Analysis complete. See detailed sections below."
        )
        sections.append("")

        # ── Per-agent details ───────────────────────────────────
        sections.append("## Detailed Analysis\n")
        for agent_name, result in agent_results.items():
            if not isinstance(result, dict):
                continue
            sections.append(f"### {agent_name.replace('_', ' ').title()}\n")
            sections.append(result.get("summary", "No summary available."))
            data = result.get("data", {})
            if data:
                for k, v in data.items():
                    if isinstance(v, (str, int, float, bool)):
                        sections.append(f"- **{k}:** {v}")
            sections.append("")

        # ── Sources ─────────────────────────────────────────────
        if sources:
            sections.append("## Sources\n")
            for src in sources:
                sections.append(f"- {src}")
            sections.append("")

        # ── Recommendations ─────────────────────────────────────
        if recommendations:
            sections.append("## Key Takeaways & Next Steps\n")
            for rec in recommendations:
                sections.append(f"- {rec}")
            sections.append("")

        # ── Follow-ups ──────────────────────────────────────────
        if follow_ups:
            sections.append("## Suggested Follow-up Questions\n")
            for fu in follow_ups:
                sections.append(f"- {fu}")
            sections.append("")

        # ── Critic feedback ─────────────────────────────────────
        if critic_feedback:
            sections.append("## Quality Assurance Notes\n")
            sections.append(
                f"The response was revised based on: {critic_feedback}"
            )
            sections.append("")

        return "\n".join(sections)
