"""
sentiment – Financial sentiment analysis models.

Provides the FinBERT-based sentiment analyzer for classifying financial
text into bullish / bearish / neutral categories with normalised scores.
"""

from __future__ import annotations

from backend.ml_models.sentiment.finbert import FinBERTSentimentAnalyzer

__all__ = [
    "FinBERTSentimentAnalyzer",
]
