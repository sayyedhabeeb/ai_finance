"""News & Sentiment Agent — Tools.

LangChain-compatible tools for fetching financial news, scoring sentiment
using FinBERT, extracting events, and assessing their market impact.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.tools import tool


# ============================================================
# News Fetcher
# ============================================================

@tool
def news_fetcher(
    query: str,
    tickers: Optional[list[str]] = None,
    days_back: int = 7,
    max_articles: int = 20,
) -> dict[str, Any]:
    """Fetch financial news articles for specified tickers or topics.

    Uses a keyword-based search to simulate news retrieval.  In production,
    this connects to NewsAPI, Google News RSS, or Yahoo Finance.

    Parameters
    ----------
    query:
        Search query (e.g. "RBI policy rate", "TCS earnings").
    tickers:
        Optional list of ticker symbols to filter by.
    days_back:
        How many days back to search (default 7).
    max_articles:
        Maximum number of articles to return.

    Returns
    -------
    dict
        List of news articles with metadata.
    """
    since_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # In production, this would call real APIs.
    # Here we return a structured template indicating what would be fetched.
    search_context = query
    if tickers:
        search_context += " " + " ".join(tickers)

    articles: list[dict[str, Any]] = [
        {
            "title": f"[Sample] Market update related to: {query}",
            "source": "financial_times",
            "published_at": since_date,
            "url": f"https://example.com/news/{hash(query) % 10000}",
            "snippet": (
                f"News article covering {query}. "
                "In production, this would contain the actual article text."
            ),
            "tickers_mentioned": tickers or [],
            "relevance_score": 0.85,
        }
    ]

    return {
        "query": query,
        "tickers": tickers,
        "since_date": since_date,
        "articles_count": len(articles),
        "articles": articles[:max_articles],
        "note": (
            "In production, this connects to NewsAPI, Google News, and "
            "Yahoo Finance for real articles."
        ),
    }


# ============================================================
# Sentiment Scorer (FinBERT-based)
# ============================================================

_SENTIMENT_KEYWORDS: dict[str, list[str]] = {
    "strongly_bearish": [
        "crash", "plunge", "massive loss", "bankrupt", "fraud", "scam",
        "severe downgrade", "crisis", "collapse", "devastating",
        "bloodbath", "rout", "panic", "fear", "doom",
    ],
    "bearish": [
        "decline", "drop", "fall", "loss", "downgrade", "weak",
        "disappointing", "miss", "below expectations", "cut",
        "slowdown", "concern", "risk", "warning", "slump",
    ],
    "neutral": [
        "stable", "flat", "unchanged", "steady", "hold", "maintain",
        "in-line", "as expected", "mixed", "moderate", "average",
    ],
    "bullish": [
        "rise", "gain", "growth", "upgrade", "beat", "strong",
        "above expectations", "positive", "rally", "surge",
        "optimistic", "confident", "opportunity", "expansion",
    ],
    "strongly_bullish": [
        "record high", "breakthrough", "exceptional", "outstanding",
        "massive growth", "blockbuster", "stellar", "explosive",
        "soaring", "skyrocket", "triumph", "phenomenal",
    ],
}


def _keyword_sentiment(text: str) -> dict[str, Any]:
    """Score sentiment using keyword matching as fallback.

    Parameters
    ----------
    text:
        The text to analyse.

    Returns
    -------
    dict
        Sentiment label, score, and confidence.
    """
    text_lower = text.lower()
    scores: dict[str, float] = {}

    for sentiment, keywords in _SENTIMENT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        scores[sentiment] = count

    total = sum(scores.values())
    if total == 0:
        return {
            "label": "neutral",
            "score": 0.0,
            "confidence": 0.5,
        }

    # Weighted scoring: strongly = ±2, normal = ±1
    weighted: dict[str, float] = {
        "strongly_bearish": -2.0,
        "bearish": -1.0,
        "neutral": 0.0,
        "bullish": 1.0,
        "strongly_bullish": 2.0,
    }

    net_score = 0.0
    for sentiment, count in scores.items():
        if count > 0:
            net_score += weighted[sentiment] * (count / total)

    # Normalize to [-1, 1]
    max_possible = 2.0
    normalized = max(-1.0, min(1.0, net_score / max_possible))

    # Determine label
    if normalized <= -0.6:
        label = "strongly_bearish"
    elif normalized <= -0.2:
        label = "bearish"
    elif normalized <= 0.2:
        label = "neutral"
    elif normalized <= 0.6:
        label = "bullish"
    else:
        label = "strongly_bullish"

    confidence = min(total / 5.0, 1.0)  # more keyword hits = higher confidence

    return {
        "label": label,
        "score": round(normalized, 4),
        "confidence": round(confidence, 2),
        "keyword_hits": {k: v for k, v in scores.items() if v > 0},
    }


@tool
def sentiment_scorer(
    text: str,
    method: str = "keyword",
) -> dict[str, Any]:
    """Score sentiment of financial text.

    Attempts FinBERT if available; falls back to keyword-based scoring.

    Parameters
    ----------
    text:
        Financial text to analyse (headline, article body, tweet).
    method:
        ``"finbert"`` or ``"keyword"``.  If ``"finbert"`` is requested
        but the model is not loaded, automatically falls back to keyword.

    Returns
    -------
    dict
        Sentiment label, numeric score (−1 to +1), and confidence.
    """
    if method == "finbert":
        try:
            return _finbert_sentiment(text)
        except Exception:
            pass  # fall through to keyword

    return _keyword_sentiment(text)


def _finbert_sentiment(text: str) -> dict[str, Any]:
    """Score sentiment using the FinBERT model (Prosus AI/finbert).

    Parameters
    ----------
    text:
        Text to analyse.

    Returns
    -------
    dict
        Sentiment label, score, and confidence.

    Raises
    ------
    Exception
        If the model cannot be loaded or inference fails.
    """
    from transformers import pipeline

    # Use a module-level variable for caching the pipeline
    import agents.news_sentiment.tools as tools_module
    if not hasattr(tools_module, "_finbert_pipeline"):
        tools_module._finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=3,
        )

    results = tools_module._finbert_pipeline(text)
    if not results:
        return _keyword_sentiment(text)

    # FinBERT returns labels: positive, negative, neutral
    label_map = {
        "positive": "bullish",
        "negative": "bearish",
        "neutral": "neutral",
    }

    top_result = results[0]
    label = label_map.get(top_result["label"], "neutral")
    score = top_result["score"]

    # Convert to -1 to +1 scale
    if label == "bullish":
        numeric = score
    elif label == "bearish":
        numeric = -score
    else:
        numeric = 0.0

    # Adjust for strongly labels
    if abs(numeric) > 0.9:
        label = f"strongly_{label}"

    return {
        "label": label,
        "score": round(numeric, 4),
        "confidence": round(score, 2),
        "method": "finbert",
        "raw_results": results,
    }


# ============================================================
# Trend Extractor
# ============================================================

@tool
def trend_extractor(
    articles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract emerging themes and trends from a set of news articles.

    Parameters
    ----------
    articles:
        List of article dicts with at least ``title`` and ``snippet`` fields.

    Returns
    -------
    dict
        Extracted trends, frequency, and associated tickers.
    """
    if not articles:
        return {"error": "No articles provided"}

    # Combine all text
    all_text = " ".join(
        f"{a.get('title', '')} {a.get('snippet', '')}"
        for a in articles
    ).lower()

    # Define financial trend categories with keywords
    trend_definitions: dict[str, list[str]] = {
        "interest_rate": ["rbi", "rate cut", "rate hike", "repo rate", "monetary policy", "rate decision"],
        "earnings": ["earnings", "profit", "revenue", "quarterly", "result", "eps", "ebitda"],
        "regulatory": ["sebi", "regulation", "compliance", "ban", "restriction", "fpi", "fii", "dii"],
        "macro": ["gdp", "inflation", "cpi", "wpi", "manufacturing", "pmi", "ise"],
        "sector_it": ["it", "tech", "tcs", "infosys", "wipro", "hcl tech", "software"],
        "sector_banking": ["bank", "hdfc", "icici", "sbi", "kotak", "axis", "npa", "credit growth"],
        "sector_oil_gas": ["oil", "gas", "reliance", "ongc", "petrol", "crude", "energy"],
        "sector_pharma": ["pharma", "drug", "fda", "approval", "sun pharma", "cipla", "dr reddy"],
        "ipo": ["ipo", "listing", "listing gains", "mainboard", "sme", "nmp"],
        "geopolitical": ["war", "sanction", "trade war", "tariff", "geopolitical", "us-china", "russia"],
        "esg": ["esg", "green", "sustainable", "climate", "carbon", "renewable", "solar"],
        "fii_dii": ["fii", "dii", "foreign", "domestic", "inflows", "outflows"],
    }

    trends: list[dict[str, Any]] = []
    for trend_name, keywords in trend_definitions.items():
        hits = sum(1 for kw in keywords if kw in all_text)
        if hits > 0:
            # Find which tickers are associated
            mentioned_tickers: list[str] = []
            for article in articles:
                article_text = (
                    f"{article.get('title', '')} {article.get('snippet', '')}"
                ).lower()
                if any(kw in article_text for kw in keywords):
                    mentioned_tickers.extend(
                        t for t in article.get("tickers_mentioned", [])
                        if t not in mentioned_tickers
                    )

            trends.append({
                "trend": trend_name.replace("_", " ").title(),
                "keyword_hits": hits,
                "intensity": (
                    "high" if hits >= 5
                    else "medium" if hits >= 2
                    else "low"
                ),
                "associated_tickers": mentioned_tickers,
            })

    # Sort by intensity
    intensity_order = {"high": 3, "medium": 2, "low": 1}
    trends.sort(key=lambda t: intensity_order.get(t["intensity"], 0), reverse=True)

    return {
        "total_trends": len(trends),
        "trends": trends,
        "dominant_theme": trends[0]["trend"] if trends else "None",
        "articles_analysed": len(articles),
    }


# ============================================================
# Event Impact Analyzer
# ============================================================

@tool
def event_impact_analyzer(
    event_description: str,
    affected_tickers: list[str],
    event_type: str = "earnings",
    severity: str = "moderate",
) -> dict[str, Any]:
    """Assess the potential market impact of a financial event.

    Parameters
    ----------
    event_description:
        Description of the event (headline or summary).
    affected_tickers:
        Tickers likely to be affected.
    event_type:
        Category: ``"earnings"``, ``"regulatory"``, ``"macro"``,
        ``"geopolitical"``, ``"sector"``, ``"company_specific"``.
    severity:
        ``"low"``, ``"moderate"``, ``"high"``, ``"critical"``.

    Returns
    -------
    dict
        Impact assessment with directional bias and affected areas.
    """
    # Base impact parameters by event type
    type_impacts: dict[str, dict[str, Any]] = {
        "earnings": {
            "direct_impact": "high",
            "spillover": "medium",
            "duration": "1-5 days",
            "volatility_increase": 0.15,
        },
        "regulatory": {
            "direct_impact": "high",
            "spillover": "high",
            "duration": "5-30 days",
            "volatility_increase": 0.25,
        },
        "macro": {
            "direct_impact": "medium",
            "spillover": "high",
            "duration": "1-10 days",
            "volatility_increase": 0.20,
        },
        "geopolitical": {
            "direct_impact": "medium",
            "spillover": "high",
            "duration": "7-60 days",
            "volatility_increase": 0.30,
        },
        "sector": {
            "direct_impact": "high",
            "spillover": "medium",
            "duration": "3-15 days",
            "volatility_increase": 0.20,
        },
        "company_specific": {
            "direct_impact": "very_high",
            "spillover": "low",
            "duration": "1-10 days",
            "volatility_increase": 0.35,
        },
    }

    severity_multipliers = {
        "low": 0.5,
        "moderate": 1.0,
        "high": 1.5,
        "critical": 2.0,
    }

    # Get sentiment of the event
    sentiment = _keyword_sentiment(event_description)
    direction = (
        "positive" if sentiment["score"] > 0.2
        else "negative" if sentiment["score"] < -0.2
        else "neutral"
    )

    base_impact = type_impacts.get(event_type, type_impacts["macro"])
    multiplier = severity_multipliers.get(severity, 1.0)
    expected_vol_increase = round(
        base_impact["volatility_increase"] * multiplier, 2
    )

    # Price impact estimation
    if direction == "positive":
        estimated_price_impact = round(
            0.02 * multiplier * (abs(sentiment["score"]) + 0.3), 4
        )
    elif direction == "negative":
        estimated_price_impact = round(
            -0.02 * multiplier * (abs(sentiment["score"]) + 0.3), 4
        )
    else:
        estimated_price_impact = 0.0

    return {
        "event": event_description[:200],
        "event_type": event_type,
        "severity": severity,
        "affected_tickers": affected_tickers,
        "sentiment": sentiment,
        "direction": direction,
        "expected_impact": {
            "price_change_pct": estimated_price_impact,
            "volatility_increase_pct": expected_vol_increase,
            "duration": base_impact["duration"],
            "direct_impact": base_impact["direct_impact"],
            "spillover": base_impact["spillover"],
        },
        "recommendation": (
            f"{'Consider reducing exposure' if direction == 'negative' else 'Monitor closely' if direction == 'neutral' else 'Evaluate buying opportunity'} "
            f"in {', '.join(affected_tickers[:3])} for next {base_impact['duration']}"
        ),
    }


# ============================================================
# Export all tools
# ============================================================

NEWS_SENTIMENT_TOOLS = [
    news_fetcher,
    sentiment_scorer,
    trend_extractor,
    event_impact_analyzer,
]
