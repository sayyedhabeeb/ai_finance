"""
anomaly – Anomaly detection for financial data.

Provides an Isolation Forest-based detector with ensemble statistical
methods for price, volume, and correlation anomaly detection.
"""

from __future__ import annotations

from backend.ml_models.anomaly.isolation_forest import FinancialAnomalyDetector

__all__ = [
    "FinancialAnomalyDetector",
]
