"""
explainability – Model interpretability and explanation.

Provides SHAP-based explainers for generating feature importance reports,
individual prediction explanations, and portfolio allocation analysis.
"""

from __future__ import annotations

from backend.ml_models.explainability.shap import ModelExplainer

__all__ = [
    "ModelExplainer",
]
