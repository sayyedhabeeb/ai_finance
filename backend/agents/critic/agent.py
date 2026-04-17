# ============================================================
# AI Financial Brain — Critic Agent
# ============================================================
"""
Critic Agent – evaluates quality of aggregated agent outputs using
G-Eval methodology across 5 quality dimensions, with revision loop
support and MLflow experiment logging.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.agents.critic.quality_rubric import (
    DIMENSION_NAMES,
    QUALITY_RUBRICS,
    RUBRIC_BY_DIMENSION,
    build_geval_prompt,
    build_overall_evaluation_prompt,
    build_revision_prompt,
)
from backend.config.schemas import (
    AgentResult,
    AgentTask,
    AgentType,
    CritiqueResult,
)

logger = structlog.get_logger(__name__)

# Minimum quality score for auto-approval
DEFAULT_QUALITY_THRESHOLD: float = 0.75
# Maximum revision iterations before forcing acceptance
MAX_REVISIONS: int = 3


class CriticAgent(BaseAgent):
    """Critic Agent.

    Evaluates the quality of agent outputs using a G-Eval inspired
    multi-dimensional scoring framework:

    1. **Relevance** (25%) — Does the output address the query?
    2. **Accuracy** (25%) — Are claims, figures, and rules correct?
    3. **Completeness** (15%) — Are all aspects of the query covered?
    4. **Coherence** (15%) — Is the output well-structured and logical?
    5. **Actionability** (20%) — Are there clear, specific next steps?

    Supports iterative revision loops where low-scoring outputs are
    sent back to the producing agent for improvement.
    """

    agent_type: AgentType = AgentType.CRITIC
    name: str = "Critic"
    description: str = (
        "Evaluates and scores agent output quality across 5 dimensions "
        "(relevance, accuracy, completeness, coherence, actionability) "
        "using G-Eval methodology with revision loop support."
    )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT: str = """\
You are a rigorous quality evaluator for an AI financial advisory system. \
Your job is to objectively score agent outputs on multiple quality dimensions \
and provide actionable feedback for improvement.

## Your Role
You are the final quality gate between agent outputs and user-facing responses. \
You must be fair, consistent, and thorough in your evaluations.

## Evaluation Principles
1. **Domain-Aware**: Evaluate with knowledge of Indian finance, tax rules, \
and market conventions. INR amounts, NSE/BSE references, and Indian tax \
sections (80C, 80D, etc.) should be accurate.
2. **User-Centric**: Score based on whether the output would actually help \
a user make informed financial decisions.
3. **Evidence-Based**: Every score must be supported by specific examples from \
the output. Don't give high scores without justification.
4. **Constructive**: Feedback should be specific and actionable, not just critical.
5. **Consistent**: Apply the same standards across all agent types and queries.
6. **Calibrated**: A score of 0.75 represents "acceptable quality" — good enough \
to show to a user without revision. A score of 0.90+ represents "excellent quality."

## Scoring Behaviour
- Be strict on factual accuracy (wrong tax rates, incorrect formulas are serious)
- Be generous on completeness if the core question is answered well
- Be strict on actionability — vague advice is not helpful
- Apply higher standards for tax/legal content (must be precise)
- Apply lower bar for market commentary (inherently uncertain)
"""

    # ------------------------------------------------------------------
    # Tool registry (Critic has no tools — it evaluates)
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list:
        """Critic agent has no tools — it evaluates using LLM reasoning."""
        return []

    def get_tools(self) -> list:
        """Return empty list — critic has no tools."""
        return []

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input(self, task: AgentTask) -> bool:
        """Validate the task has both a query and agent results to evaluate."""
        if not task.query or not task.query.strip():
            return False
        agent_results = task.context.get("agent_results")
        agent_output = task.context.get("agent_output")
        if not agent_results and not agent_output:
            return False
        return True

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a quality evaluation task."""
        return await self._run_with_metrics(task, self._execute_inner)

    async def _execute_inner(self, task: AgentTask) -> AgentResult:
        """Inner execution — run G-Eval scoring."""
        query = task.query
        agent_output = (
            task.context.get("agent_output", "")
            or self._extract_output_from_results(task.context.get("agent_results", {}))
        )
        agent_type_str = task.context.get("producing_agent_type", "unknown")

        if not agent_output:
            return self._make_error_result(
                task,
                ValueError("No agent output to evaluate"),
            )

        # Run per-dimension G-Eval scoring
        critique = await self.evaluate_output(
            query=query,
            output=agent_output,
            agent_type=agent_type_str,
        )

        # Log to MLflow (best-effort)
        self._log_to_mlflow(task, critique, agent_type_str)

        # Format output
        output = self._format_critique_output(critique, agent_type_str)

        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            output=output,
            confidence=critique.score,
            processing_time=0.0,
            metadata={
                "scores": {
                    "overall": critique["score"],
                    "dimensions": critique.get("dimensions", {}),
                },
                "passed": not critique["revision_needed"],
                "revision_count": task.context.get("revision_count", 0),
                "suggested_improvements": critique.get("suggested_improvements", []),
            },
        )

    # ------------------------------------------------------------------
    # G-Eval Scoring
    # ------------------------------------------------------------------

    async def evaluate_output(
        self,
        query: str,
        output: str,
        agent_type: str = "unknown",
    ) -> CritiqueResult:
        """Evaluate an agent output across all 5 quality dimensions.

        Parameters
        ----------
        query:
            The original user query.
        output:
            The agent output to evaluate.
        agent_type:
            The type of agent that produced the output.

        Returns
        -------
        CritiqueResult
            Structured evaluation with per-dimension scores and feedback.
        """
        dimension_scores: dict[str, dict[str, Any]] = {}

        for dimension_name in DIMENSION_NAMES:
            rubric = RUBRIC_BY_DIMENSION.get(dimension_name)
            if not rubric:
                continue

            # Build G-Eval prompt for this dimension
            prompt = build_geval_prompt(
                dimension=dimension_name,
                rubric=rubric,
                query=query,
                output=output,
                agent_type=agent_type,
            )

            messages = [
                SystemMessage(content=self._SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            try:
                response = await self._llm_with_retry_async(messages)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Parse the JSON response
                parsed = self._extract_json_from_text(response_text)
                score = float(parsed.get("score", 0.5))
                reasoning = str(parsed.get("reasoning", ""))
            except Exception as exc:
                logger.warning(
                    "critic_dimension_eval_failed",
                    dimension=dimension_name,
                    error=str(exc),
                )
                score = 0.5
                reasoning = f"Evaluation failed: {exc}"

            # Clamp score
            score = max(0.0, min(1.0, round(score, 4)))

            dimension_scores[dimension_name] = {
                "score": score,
                "reasoning": reasoning,
                "weight": rubric.weight,
            }

        # Build dimensions dict
        dimensions: dict[str, float] = {
            "relevance": dimension_scores.get("relevance", {}).get("score", 0.0),
            "accuracy": dimension_scores.get("accuracy", {}).get("score", 0.0),
            "completeness": dimension_scores.get("completeness", {}).get("score", 0.0),
            "coherence": dimension_scores.get("coherence", {}).get("score", 0.0),
            "actionability": dimension_scores.get("actionability", {}).get("score", 0.0),
        }

        # Weighted overall score
        weights = {
            "relevance": 0.25,
            "accuracy": 0.25,
            "completeness": 0.15,
            "coherence": 0.15,
            "actionability": 0.20,
        }
        overall_score = round(
            sum(dimensions[dim] * weights[dim] for dim in dimensions), 4
        )

        # Run holistic evaluation for feedback and improvements
        feedback_result = await self._generate_holistic_feedback(
            query=query,
            output=output,
            dimension_scores=dimension_scores,
            overall_score=overall_score,
            agent_type=agent_type,
        )

        passed = overall_score >= DEFAULT_QUALITY_THRESHOLD

        return CritiqueResult(
            score=overall_score,
            dimensions=dimensions,
            feedback=feedback_result.get("feedback", ""),
            revision_needed=not passed,
            suggested_improvements=feedback_result.get("suggested_improvements", []),
        ).model_dump()

    async def generate_critique(
        self,
        query: str,
        output: str,
        agent_type: str = "unknown",
    ) -> dict[str, Any]:
        """Generate a detailed critique of the output.

        Parameters
        ----------
        query:
            Original query.
        output:
            Agent output.
        agent_type:
            Agent type.

        Returns
        -------
        dict
            Detailed critique with scores, feedback, and improvement suggestions.
        """
        critique_result = await self.evaluate_output(query, output, agent_type)
        result_dict = critique_result if isinstance(critique_result, dict) else critique_result.model_dump()

        return {
            "overall_score": result_dict["score"],
            "passed": not result_dict["revision_needed"],
            "dimensions": result_dict.get("dimensions", {}),
            "feedback": result_dict["feedback"],
            "suggested_improvements": result_dict["suggested_improvements"],
            "revision_needed": result_dict["revision_needed"],
        }

    def suggest_revisions(
        self,
        query: str,
        output: str,
        critique: dict[str, Any],
    ) -> str:
        """Build a revision prompt for the producing agent.

        Parameters
        ----------
        query:
            Original query.
        output:
            Current output.
        critique:
            Critique result.

        Returns
        -------
        str
            Revision prompt.
        """
        return build_revision_prompt(query, output, critique)

    def check_quality_threshold(
        self,
        critique_result: dict[str, Any],
        threshold: float = DEFAULT_QUALITY_THRESHOLD,
    ) -> bool:
        """Check if the critique result meets the quality threshold.

        Parameters
        ----------
        critique_result:
            The evaluation result (dict or CritiqueResult).
        threshold:
            Minimum acceptable score.

        Returns
        -------
        bool
            True if quality is acceptable.
        """
        score = critique_result["score"] if isinstance(critique_result, dict) else critique_result.score
        return score >= threshold

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    async def _generate_holistic_feedback(
        self,
        query: str,
        output: str,
        dimension_scores: dict[str, dict[str, Any]],
        overall_score: float,
        agent_type: str,
    ) -> dict[str, Any]:
        """Generate overall feedback using the LLM.

        Parameters
        ----------
        query:
            Original query.
        output:
            Agent output.
        dimension_scores:
            Per-dimension evaluation results.
        overall_score:
            Computed overall score.
        agent_type:
            Agent type.

        Returns
        -------
        dict
            Feedback and suggested improvements.
        """
        prompt = build_overall_evaluation_prompt(
            query=query,
            output=output,
            dimension_evaluations=[
                {
                    "dimension": dim,
                    "score": data["score"],
                    "reasoning": data["reasoning"],
                }
                for dim, data in dimension_scores.items()
            ],
            agent_type=agent_type,
        )

        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self._llm_with_retry_async(messages)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            parsed = self._extract_json_from_text(response_text)
            return {
                "feedback": parsed.get("feedback", "No feedback generated."),
                "suggested_improvements": parsed.get("suggested_improvements", []),
                "passed": parsed.get("passed", overall_score >= DEFAULT_QUALITY_THRESHOLD),
            }
        except Exception as exc:
            logger.warning("holistic_feedback_failed", error=str(exc))
            # Fallback: auto-generate basic feedback
            low_dims = [
                dim for dim, data in dimension_scores.items()
                if data["score"] < 0.7
            ]
            improvements = [
                f"Improve {dim.replace('_', ' ')} score"
                for dim in low_dims
            ]
            return {
                "feedback": (
                    f"Overall score: {overall_score:.2f}. "
                    f"{'Output meets quality standards.' if overall_score >= DEFAULT_QUALITY_THRESHOLD else 'Output needs revision before delivery.'} "
                    f"Weak areas: {', '.join(low_dims) if low_dims else 'none'}."
                ),
                "suggested_improvements": improvements,
                "passed": overall_score >= DEFAULT_QUALITY_THRESHOLD,
            }

    @staticmethod
    def _extract_output_from_results(agent_results: dict[str, Any]) -> str:
        """Extract the primary output from agent results dict.

        Parameters
        ----------
        agent_results:
            Dict of agent results.

        Returns
        -------
        str
            The best output text found.
        """
        if isinstance(agent_results, dict):
            # Try 'output' key directly
            output = agent_results.get("output", "")
            if output:
                return str(output)

            # Try 'message' key
            message = agent_results.get("message", "")
            if message:
                return str(message)

            # Try concatenating all result outputs
            outputs = []
            if isinstance(agent_results, dict):
                for key, value in agent_results.items():
                    if isinstance(value, dict):
                        out = value.get("output") or value.get("message") or ""
                        if out:
                            outputs.append(f"[{key}]: {out}")
                    elif isinstance(value, str):
                        outputs.append(f"[{key}]: {value}")

            if outputs:
                return "\n\n".join(outputs)

        return str(agent_results) if agent_results else ""

    @staticmethod
    def _format_critique_output(
        critique: dict[str, Any],
        agent_type: str,
    ) -> str:
        """Format the critique into a readable output string.

        Parameters
        ----------
        critique:
            Critique evaluation result dict.
        agent_type:
            Agent type that was evaluated.

        Returns
        -------
        str
            Formatted critique output.
        """
        score = critique.get("score", 0)
        revision_needed = critique.get("revision_needed", False)
        dims = critique.get("dimensions", {})
        status = "✅ PASSED" if not revision_needed else "❌ NEEDS REVISION"
        lines = [
            f"# Quality Evaluation — {agent_type}",
            f"**Overall Score**: {score:.2f}/1.0 ({status})",
            "",
            "## Dimension Scores",
            f"| Dimension | Score | Weight |",
            f"|-----------|-------|--------|",
            f"| Relevance | {dims.get('relevance', 0):.2f} | 25% |",
            f"| Accuracy | {dims.get('accuracy', 0):.2f} | 25% |",
            f"| Completeness | {dims.get('completeness', 0):.2f} | 15% |",
            f"| Coherence | {dims.get('coherence', 0):.2f} | 15% |",
            f"| Actionability | {dims.get('actionability', 0):.2f} | 20% |",
        ]

        if critique.get("feedback"):
            lines.extend([
                "",
                "## Feedback",
                critique["feedback"],
            ])

        improvements = critique.get("suggested_improvements", [])
        if improvements:
            lines.extend([
                "",
                "## Suggested Improvements",
            ])
            for i, imp in enumerate(improvements, 1):
                lines.append(f"{i}. {imp}")

        return "\n".join(lines)

    def _log_to_mlflow(
        self,
        task: AgentTask,
        critique: dict[str, Any],
        agent_type: str,
    ) -> None:
        """Log evaluation metrics to MLflow (best-effort, non-blocking).

        Parameters
        ----------
        task:
            The original task.
        critique:
            The evaluation result.
        agent_type:
            Agent type being evaluated.
        """
        try:
            import mlflow

            mlflow.log_metrics({
                "critic_overall_score": critique.get("score", 0),
                f"critic_{agent_type}_relevance": critique.get("dimensions", {}).get("relevance", 0),
                f"critic_{agent_type}_accuracy": critique.get("dimensions", {}).get("accuracy", 0),
                f"critic_{agent_type}_completeness": critique.get("dimensions", {}).get("completeness", 0),
                f"critic_{agent_type}_coherence": critique.get("dimensions", {}).get("coherence", 0),
                f"critic_{agent_type}_actionability": critique.get("dimensions", {}).get("actionability", 0),
                "critic_revision_needed": int(critique.get("revision_needed", False)),
            })
        except Exception:
            # MLflow logging is best-effort — never fail the critic for this
            pass

