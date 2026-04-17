"""
risk – Volatility and risk modelling.

Provides GARCH-family models (GARCH, GJR-GARCH, EGARCH) for conditional
volatility estimation, VaR, CVaR, and comprehensive risk diagnostics.
"""

from __future__ import annotations

from backend.ml_models.risk.garch import GARCHRiskModeler

__all__ = [
    "GARCHRiskModeler",
]
