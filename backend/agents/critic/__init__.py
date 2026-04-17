"""Critic Agent — quality evaluation and scoring."""
from backend.agents.critic.agent import CriticAgent
from backend.agents.critic.quality_rubric import (
    ACCURACY_RUBRIC,
    ACTIONABILITY_RUBRIC,
    COHERENCE_RUBRIC,
    COMPLETENESS_RUBRIC,
    DIMENSION_NAMES,
    QUALITY_RUBRICS,
    RELEVANCE_RUBRIC,
    RUBRIC_BY_DIMENSION,
    QualityRubric,
    build_geval_prompt,
    build_overall_evaluation_prompt,
    build_revision_prompt,
)

__all__ = [
    "CriticAgent",
    "QualityRubric",
    "RELEVANCE_RUBRIC",
    "ACCURACY_RUBRIC",
    "COMPLETENESS_RUBRIC",
    "COHERENCE_RUBRIC",
    "ACTIONABILITY_RUBRIC",
    "QUALITY_RUBRICS",
    "RUBRIC_BY_DIMENSION",
    "DIMENSION_NAMES",
    "build_geval_prompt",
    "build_revision_prompt",
    "build_overall_evaluation_prompt",
]

