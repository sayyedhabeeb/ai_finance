# ============================================================
# AI Financial Brain — Critic Agent Quality Rubric
# ============================================================
"""
Defines the multi-dimensional quality rubric used by the Critic Agent
for G-Eval scoring of all agent outputs.

Each dimension has:
- A weight (summing to 1.0)
- A detailed criteria description
- A scoring guide mapping score ranges to qualitative levels
- Chain-of-thought prompts for LLM evaluation
"""

from __future__ import annotations

from typing import Any, Optional

from config.schemas import QualityRubric


# ============================================================
# Rubric Dimensions
# ============================================================

RELEVANCE_RUBRIC = QualityRubric(
    dimension="relevance",
    weight=0.25,
    criteria=(
        "Measures how well the output addresses the user's specific query. "
        "A high-relevance output directly answers the question asked, uses "
        "the correct financial domain context, and avoids tangential information. "
        "For Indian market queries, relevance includes using INR, referencing "
        "Indian exchanges (NSE/BSE), and applying India-specific tax rules."
    ),
    scoring_guide=(
        "0.0-0.3 (Poor): Output does not address the user's query or is completely "
        "off-topic. May discuss unrelated financial concepts.\n"
        "0.3-0.5 (Below Average): Partially addresses the query but misses key "
        "aspects. Includes significant irrelevant information.\n"
        "0.5-0.7 (Adequate): Addresses the main question but lacks depth or "
        "misses important nuances. Some irrelevant content present.\n"
        "0.7-0.9 (Good): Directly and thoroughly addresses the query with "
        "minimal irrelevant content. Appropriate domain specificity.\n"
        "0.9-1.0 (Excellent): Perfectly targeted response that addresses every "
        "aspect of the query with domain-appropriate precision."
    ),
)

ACCURACY_RUBRIC = QualityRubric(
    dimension="accuracy",
    weight=0.25,
    criteria=(
        "Measures the factual correctness of claims, figures, and recommendations. "
        "A high-accuracy output uses correct financial formulas, cites accurate "
        "tax rates and regulations, provides correct calculations, and does not "
        "make unsubstantiated claims. Numbers should be internally consistent."
    ),
    scoring_guide=(
        "0.0-0.3 (Poor): Contains significant factual errors, wrong formulas, or "
        "fabricated data. Tax rates, regulations, or financial figures are incorrect.\n"
        "0.3-0.5 (Below Average): Contains notable inaccuracies. Some numbers don't "
        "add up or rules are partially misstated.\n"
        "0.5-0.7 (Adequate): Mostly accurate with minor errors. Core calculations "
        "are correct but some details may be slightly off.\n"
        "0.7-0.9 (Good): High factual accuracy. All key figures and rules are correct. "
        "Minor rounding or precision issues only.\n"
        "0.9-1.0 (Excellent): Flawless accuracy. Every number, rule, and claim is "
        "verified and internally consistent."
    ),
)

COMPLETENESS_RUBRIC = QualityRubric(
    dimension="completeness",
    weight=0.15,
    criteria=(
        "Measures whether all aspects of the user's query are covered. A complete "
        "output covers all sub-questions, provides all requested calculations, "
        "includes necessary context and caveats, and doesn't leave the user needing "
        "to ask follow-up questions for basic information."
    ),
    scoring_guide=(
        "0.0-0.3 (Poor): Covers less than half of the query. Major sections missing.\n"
        "0.3-0.5 (Below Average): Covers some aspects but misses critical components. "
        "User would need significant follow-up.\n"
        "0.5-0.7 (Adequate): Covers most main aspects but may skip secondary questions "
        "or omit important caveats.\n"
        "0.7-0.9 (Good): Comprehensive coverage of the query with minor gaps. "
        "Covers edge cases and caveats.\n"
        "0.9-1.0 (Excellent): Exhaustive coverage. Every aspect, sub-question, "
        "caveat, and relevant context is addressed."
    ),
)

COHERENCE_RUBRIC = QualityRubric(
    dimension="coherence",
    weight=0.15,
    criteria=(
        "Measures logical consistency, readability, and structure of the output. "
        "A coherent output flows logically from one section to the next, uses "
        "consistent terminology, maintains a clear narrative, and is well-structured "
        "with appropriate headers, bullet points, and formatting."
    ),
    scoring_guide=(
        "0.0-0.3 (Poor): Disorganized, contradictory, or unreadable. No clear "
        "structure. Ideas jump between topics randomly.\n"
        "0.3-0.5 (Below Average): Somewhat organized but with logical gaps. "
        "Formatting is inconsistent or hard to follow.\n"
        "0.5-0.7 (Adequate): Generally well-organized with clear sections. "
        "Some transitions may be abrupt or formatting inconsistent.\n"
        "0.7-0.9 (Good): Well-structured and logical. Good use of sections, "
        "headers, and formatting. Reads naturally.\n"
        "0.9-1.0 (Excellent): Exceptionally clear and well-organized. Perfect "
        "logical flow. Professional formatting throughout."
    ),
)

ACTIONABILITY_RUBRIC = QualityRubric(
    dimension="actionability",
    weight=0.20,
    criteria=(
        "Measures whether the output provides clear, specific, and implementable "
        "next steps. An actionable output gives the user concrete actions to take, "
        "specifies instruments (PPF, ELSS, Nifty BeES), provides amounts (₹), "
        "timeline, and risk considerations. It should be actionable without "
        "requiring additional professional consultation for basic steps."
    ),
    scoring_guide=(
        "0.0-0.3 (Poor): No actionable steps. Vague advice like 'invest wisely'. "
        "No specific instruments, amounts, or timelines.\n"
        "0.3-0.5 (Below Average): Some suggestions but too vague to act on. "
        "Missing specific amounts, instruments, or timelines.\n"
        "0.5-0.7 (Adequate): Provides actionable steps with specific instruments "
        "and approximate amounts. May lack detail on execution.\n"
        "0.7-0.9 (Good): Clear, specific, and implementable recommendations. "
        "Includes instruments, amounts, timelines, and risk notes.\n"
        "0.9-1.0 (Excellent): Extremely detailed action plan with exact instruments, "
        "amounts, step-by-step execution guide, tax implications, and risk mitigation."
    ),
)


# ============================================================
# Rubric Collection
# ============================================================

QUALITY_RUBRICS: list[QualityRubric] = [
    RELEVANCE_RUBRIC,
    ACCURACY_RUBRIC,
    COMPLETENESS_RUBRIC,
    COHERENCE_RUBRIC,
    ACTIONABILITY_RUBRIC,
]

RUBRIC_BY_DIMENSION: dict[str, QualityRubric] = {
    r.dimension: r for r in QUALITY_RUBRICS
}

# Dimension names for iteration
DIMENSION_NAMES: list[str] = [
    "relevance", "accuracy", "completeness", "coherence", "actionability"
]


# ============================================================
# G-Eval Prompt Templates
# ============================================================

def build_geval_prompt(
    dimension: str,
    rubric: QualityRubric,
    query: str,
    output: str,
    agent_type: str = "unknown",
) -> str:
    """Build a G-Eval chain-of-thought evaluation prompt.

    G-Eval is a evaluation framework that uses LLM chain-of-thought
    reasoning to generate evaluation scores that are more consistent
    and aligned with human judgments.

    Parameters
    ----------
    dimension:
        The quality dimension being evaluated.
    rubric:
        The rubric defining criteria and scoring guide.
    query:
        The original user query.
    output:
        The agent output being evaluated.
    agent_type:
        The type of agent that produced the output.

    Returns
    -------
    str
        A complete G-Eval prompt for the LLM.
    """
    prompt = f"""\
You are an expert evaluator assessing the quality of an AI financial advisor's \
response. You will evaluate the output along one specific dimension using a \
detailed rubric.

## Evaluation Task
Evaluate the following AI agent output on the "{dimension.upper()}" dimension.

## Agent Type
{agent_type}

## Quality Dimension: {dimension.upper()}
**Weight**: {rubric.weight * 100:.0f}%
**Criteria**: {rubric.criteria}

## Scoring Guide
{rubric.scoring_guide}

## Original User Query
{query}

## Agent Output to Evaluate
{output}

## Evaluation Instructions
1. First, analyse the output carefully with respect to the "{dimension}" criteria.
2. Consider the agent type ({agent_type}) and the domain (financial advice for Indian markets).
3. Assign a score between 0.0 and 1.0 based on the scoring guide above.
4. Provide a brief reasoning (2-3 sentences) explaining your score.

## Response Format
You MUST respond in the following JSON format only:
```json
{{
    "reasoning": "<your brief reasoning here>",
    "score": <float between 0.0 and 1.0>
}}
```
"""
    return prompt


def build_revision_prompt(
    query: str,
    output: str,
    critique: dict[str, Any],
) -> str:
    """Build a prompt for the original agent to revise its output.

    Parameters
    ----------
    query:
        The original user query.
    output:
        The current output that needs revision.
    critique:
        The critique result with scores and feedback.

    Returns
    -------
    str
        A revision prompt for the producing agent.
    """
    scores_text = "\n".join(
        f"- {dim}: {score:.2f}/1.0"
        for dim, score in critique.get("dimension_scores", {}).items()
    )

    improvements_text = "\n".join(
        f"- {imp}"
        for imp in critique.get("suggested_improvements", [])
    )

    prompt = f"""\
You are revising a previous response based on quality feedback. The original \
output scored below the quality threshold and needs improvement.

## Original Query
{query}

## Original Output (needs revision)
{output}

## Quality Critique
**Overall Score**: {critique.get('overall_score', 0):.2f}/1.0
**Status**: {'PASSED' if critique.get('passed', False) else 'NEEDS REVISION'}

### Dimension Scores
{scores_text}

### Feedback
{critique.get('feedback', 'No specific feedback provided.')}

### Suggested Improvements
{improvements_text if improvements_text else 'No specific improvements suggested.'}

## Revision Instructions
1. Address each dimension that scored below 0.7
2. Incorporate the suggested improvements
3. Fix any factual inaccuracies
4. Add any missing information (completeness)
5. Make the response more actionable
6. Maintain the same format and style but improve quality

## Revised Output
Provide the complete revised response:
"""
    return prompt


def build_overall_evaluation_prompt(
    query: str,
    output: str,
    dimension_evaluations: list[dict[str, Any]],
    agent_type: str = "unknown",
) -> str:
    """Build a prompt for the final holistic quality assessment.

    Parameters
    ----------
    query:
        Original user query.
    output:
        Agent output.
    dimension_evaluations:
        List of per-dimension evaluation results.
    agent_type:
        Agent type.

    Returns
    -------
    str
        Holistic evaluation prompt.
    """
    dims_text = "\n".join(
        f"- {ev['dimension']}: {ev['score']:.2f}/1.0 — {ev['reasoning']}"
        for ev in dimension_evaluations
    )

    prompt = f"""\
You are performing a final holistic quality assessment of an AI financial \
advisor's response. You have individual dimension scores. Now provide an \
overall assessment and actionable feedback.

## Agent Type
{agent_type}

## Original Query
{query}

## Agent Output
{output}

## Dimension Scores
{dims_text}

## Assessment Tasks
1. Provide an overall quality score (weighted average considering the dimension weights)
2. Write a comprehensive feedback paragraph (3-5 sentences)
3. List 2-4 specific, actionable suggestions for improvement
4. Determine if the output meets the quality threshold (overall >= 0.75)

## Response Format
```json
{{
    "overall_score": <float 0.0-1.0>,
    "feedback": "<comprehensive feedback paragraph>",
    "suggested_improvements": ["<improvement 1>", "<improvement 2>", ...],
    "passed": <true if overall_score >= 0.75, false otherwise>
}}
```
"""
    return prompt
