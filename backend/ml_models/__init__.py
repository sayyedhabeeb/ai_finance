"""
AI Financial Brain – ML Models Package.

Exposes the complete suite of machine-learning models used across the
platform:

* **forecasting** – PatchTST-based time-series forecasting for stock prices.
* **sentiment** – FinBERT-based financial sentiment analysis.
* **risk** – GARCH-family volatility and risk modelling.
* **anomaly** – Isolation Forest anomaly detection for financial data.
* **explainability** – SHAP-based model explainability.
* **ModelRegistry** – Central model versioning and caching registry.
"""

from __future__ import annotations

from backend.ml_models.model_registry import ModelRegistry, ModelInfo

__all__ = [
    "ModelRegistry",
    "ModelInfo",
]
