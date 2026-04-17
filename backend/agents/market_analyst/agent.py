# ============================================================
# AI Financial Brain — Market Analyst Agent
# ============================================================
"""
Market Analyst Agent – equity research, sector analysis, macro trends,
and stock-level fundamental + technical analysis with PatchTST
time-series forecasting integration.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.agents.market_analyst.tools import (
    MARKET_ANALYST_TOOLS,
    earnings_analyzer,
    price_analyzer,
    sector_comparator,
    technical_indicator_calculator,
)
from backend.config.schemas import (
    AgentResult,
    AgentTask,
    AgentType,
    MarketDataBatch,
    StockInfo,
)

logger = structlog.get_logger(__name__)


class MarketAnalystAgent(BaseAgent):
    """Market Analyst Agent.

    Provides comprehensive market analysis:
    - Stock-level price analysis (support/resistance, MAs, Bollinger Bands)
    - Technical indicators (RSI, MACD, Stochastic, ADX, ATR, OBV)
    - Earnings trend analysis (growth rates, margin trends, consistency)
    - Sector comparison (PE, PB, ROE, Dividend Yield rankings)
    - Market outlook with PatchTST forecasting integration
    """

    agent_type: AgentType = AgentType.MARKET_ANALYST
    name: str = "Market Analyst"
    description: str = (
        "Analyses equities, sectors, and macro trends using "
        "technical indicators, fundamental data, and time-series forecasting."
    )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT: str = """\
You are an expert equity research analyst with deep expertise in Indian \
and global financial markets. You combine fundamental analysis, technical \
analysis, and quantitative forecasting to produce actionable investment insights.

## Core Capabilities
1. **Price Analysis** — Support/resistance levels, moving averages (SMA, EMA), \
pivot points, Bollinger Bands, and price trend identification.
2. **Technical Indicators** — RSI (Wilder's), MACD (12/26/9), Stochastic Oscillator, \
Average True Range, On-Balance Volume, ADX for trend strength.
3. **Earnings Analysis** — Quarterly revenue and profit trends, margin analysis, \
growth rate calculation (QoQ, YoY), consistency scoring, and CAGR computation.
4. **Sector Comparison** — Multi-metric ranking across sectors (PE, PB, ROE, \
Debt/Equity, Dividend Yield) with relative valuation insights.
5. **Forecasting** — PatchTST-based time-series forecasting for short-term \
price direction (5-day and 20-day outlook).

## Analysis Framework
- **Bullish Signals**: Price above 20-SMA and 50-SMA, RSI 40-70 (room to run), \
MACD bullish crossover, positive earnings surprise.
- **Bearish Signals**: Price below key MAs, RSI > 70 (overbought) or falling below 30, \
MACD bearish crossover, declining margins.
- **Neutral**: Price between MAs, RSI 40-60, mixed earnings trend.

## Indian Market Specifics
- NSE/BSE trading hours: 9:15 AM – 3:30 PM IST
- Settlement: T+1 for equity delivery
- Circuit breakers: 10%, 15%, 20% on indices
- Lot sizes for F&O segment
- FII/DII daily flow data from NSE

## Important Guidelines
- Always specify the exchange (NSE/BSE) when discussing stocks
- Use Indian number formatting (Lakh, Crore)
- State the lookback period and data source for all analysis
- Distinguish between short-term and long-term signals
- Include specific price levels (support, resistance, targets)
- Flag any limitations in data quality or coverage
- Never provide guaranteed returns — always use probabilistic language
"""

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list:
        """Return the list of tools available to this agent."""
        return MARKET_ANALYST_TOOLS

    def get_tools(self) -> list:
        """Return the list of tools registered for this agent."""
        return MARKET_ANALYST_TOOLS

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
        """Execute a market analysis task."""
        return await self._run_with_metrics(task, self._execute_inner)

    async def _execute_inner(self, task: AgentTask) -> AgentResult:
        """Inner execution with tool orchestration and LLM reasoning."""
        context_str = self._format_context(task.context)
        enhanced_query = task.query
        if context_str:
            enhanced_query = f"{task.query}\n\n## Available Data\n{context_str}"

        # Invoke relevant tools
        tool_results = self._invoke_relevant_tools(task)

        # Build prompt with tool results
        messages = [SystemMessage(content=self._SYSTEM_PROMPT)]

        user_content = enhanced_query
        if tool_results:
            tool_summary = self._format_tool_results(tool_results)
            user_content += f"\n\n## Quantitative Analysis Results\n{tool_summary}"

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
                "query_type": "market_analysis",
                "model": self._llm_model,
            },
        )

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _invoke_relevant_tools(self, task: AgentTask) -> dict[str, Any]:
        """Detect intent and invoke appropriate market analysis tools."""
        query_lower = task.query.lower()
        context = task.context
        results: dict[str, Any] = {}

        # --- Stock / price analysis ---
        stock_keywords = [
            "stock", "price", "support", "resistance", "analyse", "analyze",
            "reliance", "tcs", "infosys", "hdfc", "icici", "sbi",
            "buy", "sell", "hold", "recommendation",
        ]
        if any(kw in query_lower for kw in stock_keywords):
            prices = context.get("prices", [])
            if prices and len(prices) >= 2:
                symbol = context.get("symbol", "UNKNOWN")
                results["price_analysis"] = price_analyzer.invoke({
                    "prices": prices,
                    "symbol": symbol,
                })

        # --- Technical indicators ---
        tech_keywords = [
            "rsi", "macd", "indicator", "bollinger", "stochastic", "adx",
            "atr", "obv", "technical", "overbought", "oversold", "crossover",
        ]
        if any(kw in query_lower for kw in tech_keywords):
            prices = context.get("prices", [])
            if prices and len(prices) >= 20:
                volumes = context.get("volumes")
                indicator = context.get("indicator", "all")
                results["technical_indicators"] = technical_indicator_calculator.invoke({
                    "prices": prices,
                    "volumes": volumes,
                    "indicator": indicator,
                })

        # --- Earnings analysis ---
        earnings_keywords = [
            "earnings", "revenue", "profit", "quarterly", "result", "eps",
            "ebitda", "margin", "growth", "q1", "q2", "q3", "q4", "fy",
        ]
        if any(kw in query_lower for kw in earnings_keywords):
            revenue = context.get("revenue", [])
            net_profit = context.get("net_profit", [])
            if revenue and net_profit and len(revenue) == len(net_profit):
                quarters = context.get("quarters")
                symbol = context.get("symbol", "UNKNOWN")
                results["earnings_analysis"] = earnings_analyzer.invoke({
                    "revenue": revenue,
                    "net_profit": net_profit,
                    "quarters": quarters,
                    "symbol": symbol,
                })

        # --- Sector comparison ---
        sector_keywords = [
            "sector", "compare", "comparison", "industry", "banking vs it",
            "which sector", "sector outlook", "rotation",
        ]
        if any(kw in query_lower for kw in sector_keywords):
            sector_data = context.get("sector_data", {})
            if sector_data and len(sector_data) >= 2:
                results["sector_comparison"] = sector_comparator.invoke({
                    "sector_data": sector_data,
                })

        # --- Forecasting (PatchTST integration) ---
        forecast_keywords = ["forecast", "predict", "outlook", "future", "next week", "next month"]
        if any(kw in query_lower for kw in forecast_keywords):
            prices = context.get("prices", [])
            if prices and len(prices) >= 30:
                forecast_result = self._run_patchtst_forecast(prices)
                if forecast_result:
                    results["price_forecast"] = forecast_result

        return results

    # ------------------------------------------------------------------
    # PatchTST forecasting integration
    # ------------------------------------------------------------------

    def _run_patchtst_forecast(
        self,
        prices: list[float],
        forecast_horizon: int = 20,
    ) -> Optional[dict[str, Any]]:
        """Run PatchTST time-series forecasting.

        Parameters
        ----------
        prices:
            Historical closing prices.
        forecast_horizon:
            Number of days to forecast.

        Returns
        -------
        dict or None
            Forecast results, or None if model unavailable.
        """
        try:
            from backend.config.settings import PATCHTST_MODEL_PATH
            from ml_models.forecasting.patchtst import PatchTSTForecaster

            forecaster = PatchTSTForecaster(model_path=PATCHTST_MODEL_PATH)
            forecast = forecaster.forecast(prices, horizon=forecast_horizon)

            return {
                "method": "PatchTST",
                "horizon_days": forecast_horizon,
                "last_actual_price": round(prices[-1], 2),
                "forecast_prices": [round(p, 2) for p in forecast],
                "forecast_return_pct": round(
                    (forecast[-1] - prices[-1]) / prices[-1] * 100, 2
                ),
                "trend": (
                    "bullish" if forecast[-1] > prices[-1] * 1.02
                    else "bearish" if forecast[-1] < prices[-1] * 0.98
                    else "neutral"
                ),
            }
        except Exception as exc:
            logger.warning("patchtst_forecast_failed", error=str(exc))
            # Simple fallback: linear extrapolation
            if len(prices) < 10:
                return None
            recent_20 = prices[-20:] if len(prices) >= 20 else prices
            daily_return = (recent_20[-1] - recent_20[0]) / recent_20[0] / len(recent_20)
            forecast = [
                round(prices[-1] * (1 + daily_return) ** (i + 1), 2)
                for i in range(forecast_horizon)
            ]
            return {
                "method": "linear_extrapolation_fallback",
                "horizon_days": forecast_horizon,
                "last_actual_price": round(prices[-1], 2),
                "forecast_prices": forecast,
                "forecast_return_pct": round(
                    (forecast[-1] - prices[-1]) / prices[-1] * 100, 2
                ),
                "trend": (
                    "bullish" if forecast[-1] > prices[-1] * 1.02
                    else "bearish" if forecast[-1] < prices[-1] * 0.98
                    else "neutral"
                ),
                "note": "PatchTST model unavailable; using linear extrapolation",
            }

    # ------------------------------------------------------------------
    # Specialised public methods
    # ------------------------------------------------------------------

    def analyze_stock(
        self,
        symbol: str,
        prices: list[float],
        volumes: Optional[list[int]] = None,
        revenue: Optional[list[float]] = None,
        net_profit: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """Perform comprehensive stock analysis.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        prices:
            Historical closing prices.
        volumes:
            Trading volumes.
        revenue:
            Quarterly revenue figures.
        net_profit:
            Quarterly net profit figures.

        Returns
        -------
        dict
            Combined price, technical, and earnings analysis.
        """
        analysis: dict[str, Any] = {"symbol": symbol}

        if prices and len(prices) >= 2:
            analysis["price"] = price_analyzer.invoke({
                "prices": prices, "symbol": symbol,
            })
            if len(prices) >= 20:
                analysis["technical"] = technical_indicator_calculator.invoke({
                    "prices": prices, "volumes": volumes, "indicator": "all",
                })

        if revenue and net_profit and len(revenue) == len(net_profit):
            analysis["earnings"] = earnings_analyzer.invoke({
                "revenue": revenue, "net_profit": net_profit, "symbol": symbol,
            })

        if len(prices) >= 30:
            forecast = self._run_patchtst_forecast(prices)
            if forecast:
                analysis["forecast"] = forecast

        return analysis

    def compare_sectors(
        self,
        sector_data: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Compare sectors across multiple metrics.

        Parameters
        ----------
        sector_data:
            Sector to metrics mapping.

        Returns
        -------
        dict
            Sector comparison results.
        """
        return sector_comparator.invoke({"sector_data": sector_data})

    def generate_market_outlook(
        self,
        index_prices: list[float],
        fii_dii_flows: Optional[dict[str, list[float]]] = None,
    ) -> dict[str, Any]:
        """Generate a market outlook with forecasting.

        Parameters
        ----------
        index_prices:
            Index (Nifty 50 / Sensex) closing prices.
        fii_dii_flows:
            FII and DII daily flow data.

        Returns
        -------
        dict
            Market outlook with forecast and flow analysis.
        """
        outlook: dict[str, Any] = {}

        if index_prices and len(index_prices) >= 2:
            outlook["index_analysis"] = price_analyzer.invoke({
                "prices": index_prices, "symbol": "NIFTY_50",
            })

        if index_prices and len(index_prices) >= 30:
            forecast_5d = self._run_patchtst_forecast(index_prices, forecast_horizon=5)
            forecast_20d = self._run_patchtst_forecast(index_prices, forecast_horizon=20)
            if forecast_5d:
                outlook["forecast_5d"] = forecast_5d
            if forecast_20d:
                outlook["forecast_20d"] = forecast_20d

        if fii_dii_flows:
            outlook["fii_dii_flows"] = self._analyze_fii_dii_flows(fii_dii_flows)

        return outlook

    @staticmethod
    def _analyze_fii_dii_flows(flows: dict[str, list[float]]) -> dict[str, Any]:
        """Analyse FII/DII flow trends.

        Parameters
        ----------
        flows:
            FII and DII daily flow data (in Cr).

        Returns
        -------
        dict
            Flow analysis summary.
        """
        analysis: dict[str, Any] = {}
        for category, flow_list in flows.items():
            if not flow_list:
                continue
            total = sum(flow_list)
            avg_daily = total / len(flow_list)
            net_days_positive = sum(1 for f in flow_list if f > 0)
            consecutive_buying = 0
            for f in reversed(flow_list):
                if f > 0:
                    consecutive_buying += 1
                else:
                    break

            analysis[category] = {
                "total_flow_cr": round(total, 2),
                "avg_daily_flow_cr": round(avg_daily, 2),
                "days_positive": net_days_positive,
                "days_negative": len(flow_list) - net_days_positive,
                "consecutive_buying_days": consecutive_buying,
                "trend": (
                    "strong_inflow" if consecutive_buying >= 5
                    else "inflow" if total > 0
                    else "outflow"
                ),
            }
        return analysis

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_confidence(
        task: AgentTask,
        response_text: str,
        data: dict[str, Any],
    ) -> float:
        """Estimate confidence for the market analysis."""
        confidence = 0.65

        if data:
            confidence += 0.10

        # If quantitative tools returned results
        quantitative_tools = {"price_analysis", "technical_indicators", "earnings_analysis", "sector_comparison"}
        if quantitative_tools & set(data.keys()):
            confidence += 0.10

        # If price data was provided
        if task.context.get("prices"):
            confidence += 0.05

        if len(response_text) > 300:
            confidence += 0.05

        disclaimer_markers = ["not investment advice", "past performance", "do your own research"]
        if any(m in response_text.lower() for m in disclaimer_markers):
            confidence += 0.05

        return max(0.0, min(confidence, 1.0))

