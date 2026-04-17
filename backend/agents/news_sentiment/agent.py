# ============================================================
# AI Financial Brain — News & Sentiment Agent
# ============================================================
"""
News & Sentiment Agent – scrapes and analyses financial news, social
sentiment, and market events to produce actionable sentiment-driven
insights using FinBERT and keyword-based scoring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from agents.news_sentiment.tools import (
    NEWS_SENTIMENT_TOOLS,
    _keyword_sentiment,
    event_impact_analyzer,
    news_fetcher,
    sentiment_scorer,
    trend_extractor,
)
from config.schemas import (
    AgentResult,
    AgentTask,
    AgentType,
    SentimentType,
)

logger = structlog.get_logger(__name__)


class NewsSentimentAgent(BaseAgent):
    """News & Sentiment Agent.

    Provides comprehensive news and sentiment analysis:
    - News monitoring and aggregation from multiple sources
    - FinBERT-based sentiment scoring with keyword fallback
    - Trend extraction across multiple articles
    - Event impact assessment for specific tickers and sectors
    - News digest compilation with sentiment overlay
    """

    agent_type: AgentType = AgentType.NEWS_SENTIMENT
    name: str = "News & Sentiment"
    description: str = (
        "News and sentiment analyst that scrapes financial news, scores "
        "sentiment using FinBERT, extracts trends, and assesses market "
        "impact of events."
    )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT: str = """\
You are an expert financial news analyst and sentiment specialist focused \
on Indian markets. You synthesize news flows, social sentiment, and market \
events into actionable insights for investors.

## Core Capabilities
1. **News Monitoring** — Track headlines from major financial sources \
(Economic Times, Mint, Moneycontrol, LiveMint, Bloomberg Quint).
2. **Sentiment Scoring** — Use FinBERT and keyword analysis to score \
sentiment as strongly bearish / bearish / neutral / bullish / strongly bullish.
3. **Trend Extraction** — Identify emerging themes across multiple articles \
(rate decisions, sector rotation, FII/DII flows, regulatory changes).
4. **Event Impact Analysis** — Assess how events affect specific tickers, \
sectors, and the broader market with estimated price impact.
5. **Social Sentiment** — Monitor Twitter/X, Reddit, and Telegram for \
retail sentiment signals and contra-indicator analysis.

## Sentiment Methodology
- **Strongly Bullish (+0.6 to +1.0)**: Blockbuster results, strong upgrades, \
major positive catalysts, blockbuster deals.
- **Bullish (+0.2 to +0.6)**: Better-than-expected results, positive guidance, \
favourable policy changes, analyst upgrades.
- **Neutral (-0.2 to +0.2)**: In-line results, mixed signals, no major catalysts, \
routine news.
- **Bearish (-0.6 to -0.2)**: Missed estimates, downgrades, negative guidance, \
management changes.
- **Strongly Bearish (-1.0 to -0.6)**: Major negative events, fraud, crises, \
regulatory crackdowns, accounting irregularities.

## Important Guidelines
- Always cite the source of information
- Distinguish between fact and opinion/rumour
- Provide sentiment at ticker, sector, and market level
- Flag potential conflicts of interest in sources
- Note the age of information — stale news may already be priced in
- Consider contra-indicator signals (extreme bearishness as bottom signal)
- Weight recent news more heavily than older articles
- Cross-reference multiple sources before forming a conclusion
"""

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list:
        """Return the list of tools available to this agent."""
        return NEWS_SENTIMENT_TOOLS

    def get_tools(self) -> list:
        """Return the list of tools registered for this agent."""
        return NEWS_SENTIMENT_TOOLS

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input(self, task: AgentTask) -> bool:
        """Validate the task has usable content."""
        if not task.query or not task.query.strip():
            return False
        return True

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a news/sentiment analysis task."""
        return await self._run_with_metrics(task, self._execute_inner)

    async def _execute_inner(self, task: AgentTask) -> AgentResult:
        """Inner execution with tool orchestration and LLM reasoning."""
        context_str = self._format_context(task.context)
        enhanced_query = task.query
        if context_str:
            enhanced_query = f"{task.query}\n\n## Context\n{context_str}"

        # Invoke relevant tools
        tool_results = self._invoke_relevant_tools(task)

        # Build prompt with tool results
        messages = [SystemMessage(content=self._SYSTEM_PROMPT)]

        user_content = enhanced_query
        if tool_results:
            tool_summary = self._format_tool_results(tool_results)
            user_content += f"\n\n## Sentiment Analysis Data\n{tool_summary}"

        messages.append(HumanMessage(content=user_content))

        # Call LLM
        response = await self._llm_with_retry_async(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Extract structured data
        parsed_data = self._extract_json_from_text(response_text)
        if tool_results:
            parsed_data.update(tool_results)

        confidence = self._estimate_confidence(task, response_text, parsed_data)

        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            output=response_text,
            confidence=confidence,
            processing_time=0.0,
            metadata={
                "tools_used": list(tool_results.keys()) if tool_results else [],
                "query_type": "news_sentiment",
                "model": self._llm_model,
            },
        )

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _invoke_relevant_tools(self, task: AgentTask) -> dict[str, Any]:
        """Detect intent and invoke appropriate tools."""
        query_lower = task.query.lower()
        context = task.context
        results: dict[str, Any] = {}

        # --- Sentiment analysis ---
        sentiment_keywords = [
            "sentiment", "mood", "feeling", "bullish", "bearish",
            "positive", "negative", "vibe", "opinion",
        ]
        if any(kw in query_lower for kw in sentiment_keywords):
            text_to_analyze = context.get("text_to_analyze", task.query)
            results["sentiment_analysis"] = sentiment_scorer.invoke({
                "text": text_to_analyze,
                "method": context.get("sentiment_method", "keyword"),
            })

        # --- News fetching ---
        news_keywords = [
            "news", "latest", "headlines", "what happened", "recent",
            "article", "report", "update",
        ]
        if any(kw in query_lower for kw in news_keywords):
            tickers = context.get("tickers")
            if tickers or context.get("topic"):
                results["news"] = news_fetcher.invoke({
                    "query": context.get("topic", task.query),
                    "tickers": tickers,
                    "days_back": context.get("days_back", 7),
                    "max_articles": context.get("max_articles", 10),
                })

        # --- Trend extraction ---
        trend_keywords = ["trend", "emerging", "theme", "pattern", "topic", "what's trending"]
        if any(kw in query_lower for kw in trend_keywords):
            articles = context.get("articles", [])
            if articles:
                results["trends"] = trend_extractor.invoke({"articles": articles})

        # --- Event impact ---
        event_keywords = [
            "impact", "effect", "affect", "event", "what if",
            "consequence", "implication", "how will",
        ]
        if any(kw in query_lower for kw in event_keywords):
            event_desc = context.get("event_description", task.query)
            tickers = context.get("affected_tickers", [])
            if event_desc and tickers:
                results["event_impact"] = event_impact_analyzer.invoke({
                    "event_description": event_desc,
                    "affected_tickers": tickers,
                    "event_type": context.get("event_type", "earnings"),
                    "severity": context.get("severity", "moderate"),
                })

        return results

    # ------------------------------------------------------------------
    # Specialised public methods
    # ------------------------------------------------------------------

    def analyze_sentiment(
        self,
        texts: list[str],
        tickers: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Batch sentiment analysis across multiple texts.

        Parameters
        ----------
        texts:
            List of texts to analyse.
        tickers:
            Optional tickers associated with each text.

        Returns
        -------
        dict
            Aggregated sentiment results.
        """
        if not texts:
            return {"error": "No texts provided for sentiment analysis"}

        individual_results: list[dict[str, Any]] = []
        sentiment_counts: dict[str, int] = defaultdict(int)
        total_score = 0.0

        for i, text in enumerate(texts):
            result = _keyword_sentiment(text)
            result["text_index"] = i
            if tickers and i < len(tickers):
                result["ticker"] = tickers[i]
            individual_results.append(result)
            sentiment_counts[result["label"]] += 1
            total_score += result["score"]

        avg_score = total_score / len(texts) if texts else 0

        # Aggregate label
        if avg_score >= 0.6:
            aggregate = "strongly_bullish"
        elif avg_score >= 0.2:
            aggregate = "bullish"
        elif avg_score >= -0.2:
            aggregate = "neutral"
        elif avg_score >= -0.6:
            aggregate = "bearish"
        else:
            aggregate = "strongly_bearish"

        return {
            "texts_analyzed": len(texts),
            "aggregate_sentiment": aggregate,
            "average_score": round(avg_score, 4),
            "sentiment_distribution": dict(sentiment_counts),
            "individual_results": individual_results,
            "bullish_pct": round(
                (sentiment_counts.get("bullish", 0)
                 + sentiment_counts.get("strongly_bullish", 0))
                / len(texts) * 100, 1
            ),
            "bearish_pct": round(
                (sentiment_counts.get("bearish", 0)
                 + sentiment_counts.get("strongly_bearish", 0))
                / len(texts) * 100, 1
            ),
        }

    def extract_events(
        self,
        articles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract significant financial events from news articles.

        Parameters
        ----------
        articles:
            List of article dicts with title, snippet, source fields.

        Returns
        -------
        dict
            Extracted events with type, severity, and affected tickers.
        """
        if not articles:
            return {"error": "No articles to process"}

        event_patterns: dict[str, list[str]] = {
            "earnings_release": ["quarterly results", "earnings", "q1", "q2", "q3", "q4", "fy"],
            "rate_decision": ["rbi", "repo rate", "rate cut", "rate hike", "monetary policy"],
            "regulatory": ["sebi", "rbi circular", "regulation", "compliance", "penalty"],
            "ipo": ["ipo", "listing", "new issue", "fpo", "rights issue"],
            "m_a": ["acquisition", "merger", "buyout", "stake sale", "demerger"],
            "downgrade_upgrade": ["upgrade", "downgrade", "rating action", "outlook"],
            "management": ["ceo", "cfo", "resignation", "appointed", "board"],
        }

        events: list[dict[str, Any]] = []
        for article in articles:
            title = article.get("title", "").lower()
            snippet = article.get("snippet", "").lower()
            combined = f"{title} {snippet}"

            for event_type, keywords in event_patterns.items():
                if any(kw in combined for kw in keywords):
                    sentiment = _keyword_sentiment(combined)
                    severity = (
                        "high" if abs(sentiment["score"]) > 0.6
                        else "moderate" if abs(sentiment["score"]) > 0.2
                        else "low"
                    )
                    events.append({
                        "event_type": event_type,
                        "title": article.get("title", ""),
                        "source": article.get("source", ""),
                        "sentiment": sentiment["label"],
                        "severity": severity,
                        "tickers": article.get("tickers_mentioned", []),
                        "published_at": article.get("published_at", ""),
                    })
                    break

        severity_order = {"high": 3, "moderate": 2, "low": 1}
        events.sort(key=lambda e: severity_order.get(e["severity"], 0), reverse=True)

        return {
            "events_extracted": len(events),
            "events": events,
            "high_severity_count": sum(1 for e in events if e["severity"] == "high"),
        }

    def assess_impact(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assess the market impact of extracted events.

        Parameters
        ----------
        events:
            List of event dicts.

        Returns
        -------
        dict
            Impact assessment with affected tickers and net sentiment.
        """
        if not events:
            return {"error": "No events to assess"}

        ticker_impacts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        net_score = 0.0

        for event in events:
            tickers = event.get("tickers", [])
            if not tickers:
                continue

            result = event_impact_analyzer.invoke({
                "event_description": event.get("title", ""),
                "affected_tickers": tickers,
                "event_type": event.get("event_type", "earnings"),
                "severity": event.get("severity", "moderate"),
            })

            net_score += result["sentiment"]["score"]

            for ticker in tickers:
                ticker_impacts[ticker].append({
                    "event_type": event["event_type"],
                    "severity": event["severity"],
                    "direction": result["direction"],
                    "estimated_price_impact_pct": result["expected_impact"]["price_change_pct"],
                })

        ticker_summary: dict[str, dict[str, Any]] = {}
        for ticker, impacts in ticker_impacts.items():
            positive = sum(1 for imp in impacts if imp["direction"] == "positive")
            negative = sum(1 for imp in impacts if imp["direction"] == "negative")
            avg_impact = sum(
                imp["estimated_price_impact_pct"] for imp in impacts
            ) / len(impacts)

            ticker_summary[ticker] = {
                "events_count": len(impacts),
                "positive_events": positive,
                "negative_events": negative,
                "net_direction": (
                    "positive" if positive > negative
                    else "negative" if negative > positive
                    else "neutral"
                ),
                "avg_estimated_impact_pct": round(avg_impact, 4),
                "severity": (
                    "high" if any(imp["severity"] == "high" for imp in impacts)
                    else "moderate"
                ),
            }

        return {
            "total_events": len(events),
            "affected_tickers": list(ticker_summary.keys()),
            "ticker_impacts": ticker_summary,
            "net_market_sentiment": (
                "positive" if net_score > 0.3
                else "negative" if net_score < -0.3
                else "neutral"
            ),
            "net_sentiment_score": round(net_score, 4),
        }

    def compile_news_summary(
        self,
        articles: list[dict[str, Any]],
        tickers: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Compile a comprehensive news digest with sentiment overlay.

        Parameters
        ----------
        articles:
            List of news articles.
        tickers:
            Optional tickers to focus on.

        Returns
        -------
        dict
            News digest with sentiment, trends, and events.
        """
        if not articles:
            return {"error": "No articles to summarize"}

        # Score sentiment for each article
        scored_articles = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('snippet', '')}"
            sentiment = _keyword_sentiment(text)
            scored_articles.append({**article, "sentiment": sentiment})

        # Aggregate sentiment
        sentiment = self.analyze_sentiment(
            texts=[f"{a.get('title', '')} {a.get('snippet', '')}" for a in articles],
            tickers=[a.get("tickers_mentioned", ["UNKNOWN"])[0] for a in articles],
        )

        # Extract trends
        trends = trend_extractor.invoke({"articles": articles})

        # Extract events
        events = self.extract_events(articles)

        return {
            "summary_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "articles_analyzed": len(articles),
            "overall_sentiment": sentiment["aggregate_sentiment"],
            "average_sentiment_score": sentiment["average_score"],
            "sentiment_distribution": sentiment["sentiment_distribution"],
            "key_trends": trends.get("trends", [])[:5],
            "significant_events": events.get("events", [])[:5],
            "scored_articles": scored_articles[:10],
            "focus_tickers": tickers,
        }

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_confidence(
        task: AgentTask,
        response_text: str,
        data: dict[str, Any],
    ) -> float:
        """Estimate confidence for the sentiment analysis."""
        confidence = 0.60

        if data:
            confidence += 0.15

        if task.context.get("articles"):
            confidence += 0.10

        if task.context.get("text_to_analyze") or task.context.get("tickers"):
            confidence += 0.05

        if len(response_text) > 200:
            confidence += 0.05

        uncertain_markers = ["insufficient", "unclear", "conflicting signals", "limited data"]
        if any(m in response_text.lower() for m in uncertain_markers):
            confidence -= 0.10

        return max(0.0, min(confidence, 1.0))
