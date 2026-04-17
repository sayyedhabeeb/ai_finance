"""
Portfolio strategy constraints.

India-specific regulatory constraints, tax rules (LTCG/STCG/dividend),
SEBI concentration limits, and sector caps.
"""

from backend.portfolio.strategies.india_constraints import IndiaPortfolioConstraints

__all__ = ["IndiaPortfolioConstraints"]
