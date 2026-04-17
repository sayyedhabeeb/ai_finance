"""Market Analyst Agent — Tools.

LangChain-compatible tools for price analysis, technical indicators,
earnings analysis, and sector comparison.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from langchain_core.tools import tool


# ============================================================
# Price Analyzer
# ============================================================

@tool
def price_analyzer(
    prices: list[float],
    symbol: str = "UNKNOWN",
) -> dict[str, Any]:
    """Compute key price statistics and support/resistance levels
    from a list of historical closing prices.

    Parameters
    ----------
    prices:
        List of historical closing prices (most recent last).
    symbol:
        Ticker symbol for labelling.

    Returns
    -------
    dict
        Price statistics, support/resistance, moving averages, and trend.
    """
    if not prices or len(prices) < 2:
        return {"error": "Need at least 2 price data points"}

    n = len(prices)
    current = prices[-1]
    prev = prices[-2]
    change = current - prev
    change_pct = (change / prev) * 100 if prev != 0 else 0

    # Basic statistics
    mean_price = sum(prices) / n
    variance = sum((p - mean_price) ** 2 for p in prices) / n
    std_dev = math.sqrt(variance)
    high = max(prices)
    low = min(prices)

    # Moving averages
    sma_20 = (
        sum(prices[-20:]) / len(prices[-20:])
        if n >= 20 else mean_price
    )
    sma_50 = (
        sum(prices[-50:]) / len(prices[-50:])
        if n >= 50 else mean_price
    )

    # Exponential moving average (EMA-20)
    multiplier = 2 / (21)  # EMA period = 20
    ema_20 = prices[0]
    for p in prices[1:]:
        ema_20 = (p - ema_20) * multiplier + ema_20

    # Support & resistance (simple pivot-based)
    pivot = (high + low + current) / 3
    support_1 = 2 * pivot - high
    resistance_1 = 2 * pivot - low
    support_2 = pivot - (high - low)
    resistance_2 = pivot + (high - low)

    # Trend determination
    if current > sma_20 and current > sma_50:
        trend = "bullish"
    elif current < sma_20 and current < sma_50:
        trend = "bearish"
    else:
        trend = "sideways"

    # Bollinger Bands (20-period, 2 std)
    bb_upper = sma_20 + 2 * std_dev
    bb_lower = sma_20 - 2 * std_dev
    bb_position = (
        (current - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_upper != bb_lower else 50
    )

    # RSI (14-period simplified)
    rsi = _compute_rsi(prices, period=14)

    return {
        "symbol": symbol,
        "current_price": round(current, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "statistics": {
            "mean": round(mean_price, 2),
            "std_dev": round(std_dev, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "range": round(high - low, 2),
        },
        "moving_averages": {
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "ema_20": round(ema_20, 2),
        },
        "support_resistance": {
            "pivot": round(pivot, 2),
            "support_1": round(support_1, 2),
            "support_2": round(support_2, 2),
            "resistance_1": round(resistance_1, 2),
            "resistance_2": round(resistance_2, 2),
        },
        "indicators": {
            "trend": trend,
            "rsi_14": round(rsi, 2),
            "bollinger_upper": round(bb_upper, 2),
            "bollinger_lower": round(bb_lower, 2),
            "bollinger_position_pct": round(bb_position, 1),
        },
        "data_points": n,
    }


def _compute_rsi(prices: list[float], period: int = 14) -> float:
    """Compute the Relative Strength Index (Wilder's smoothing).

    Parameters
    ----------
    prices:
        List of closing prices.
    period:
        Look-back period for RSI.

    Returns
    -------
    float
        RSI value between 0 and 100.
    """
    if len(prices) < period + 1:
        return 50.0  # neutral when insufficient data

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ============================================================
# Technical Indicator Calculator
# ============================================================

@tool
def technical_indicator_calculator(
    prices: list[float],
    volumes: Optional[list[int]] = None,
    indicator: str = "all",
) -> dict[str, Any]:
    """Calculate a comprehensive set of technical indicators.

    Parameters
    ----------
    prices:
        List of closing prices.
    volumes:
        Optional list of trading volumes.
    indicator:
        Specific indicator to compute, or ``"all"`` for everything.
        Options: ``"rsi"``, ``"macd"``, ``"bollinger"``, ``"stochastic"``,
        ``"adx"``, ``"obv"``, ``"atr"``, ``"all"``.

    Returns
    -------
    dict
        Technical indicator values and signals.
    """
    if len(prices) < 20:
        return {"error": "Need at least 20 price data points"}

    result: dict[str, Any] = {"symbol": "UNKNOWN"}

    if indicator in ("all", "rsi"):
        result["rsi"] = {
            "rsi_14": round(_compute_rsi(prices, 14), 2),
            "rsi_7": round(_compute_rsi(prices, 7), 2),
            "signal": (
                "overbought" if _compute_rsi(prices, 14) > 70
                else "oversold" if _compute_rsi(prices, 14) < 30
                else "neutral"
            ),
        }

    if indicator in ("all", "macd"):
        ema_12 = _compute_ema(prices, 12)
        ema_26 = _compute_ema(prices, 26)
        macd_line = ema_12 - ema_26
        # Signal line (9-period EMA of MACD — approximate)
        macd_values = [
            _compute_ema(prices[max(0, i - 25):i + 1], 12)
            - _compute_ema(prices[max(0, i - 25):i + 1], 26)
            for i in range(26, len(prices))
        ]
        signal_line = (
            _compute_ema(macd_values, 9) if macd_values else macd_line
        )
        histogram = macd_line - signal_line
        result["macd"] = {
            "macd_line": round(macd_line, 4),
            "signal_line": round(signal_line, 4),
            "histogram": round(histogram, 4),
            "signal": (
                "bullish" if histogram > 0 else "bearish"
            ),
        }

    if indicator in ("all", "bollinger"):
        sma_20 = sum(prices[-20:]) / 20
        std_20 = math.sqrt(
            sum((p - sma_20) ** 2 for p in prices[-20:]) / 20
        )
        result["bollinger_bands"] = {
            "upper": round(sma_20 + 2 * std_20, 2),
            "middle": round(sma_20, 2),
            "lower": round(sma_20 - 2 * std_20, 2),
            "bandwidth": round(4 * std_20 / sma_20 * 100, 2)
            if sma_20 > 0 else 0,
            "percent_b": round(
                (prices[-1] - (sma_20 - 2 * std_20))
                / (4 * std_20) * 100, 2
            ) if std_20 > 0 else 50,
        }

    if indicator in ("all", "stochastic"):
        high_14 = max(prices[-14:])
        low_14 = min(prices[-14:])
        stoch_k = (
            (prices[-1] - low_14) / (high_14 - low_14) * 100
            if high_14 != low_14 else 50
        )
        result["stochastic"] = {
            "stoch_k": round(stoch_k, 2),
            "signal": (
                "overbought" if stoch_k > 80
                else "oversold" if stoch_k < 20
                else "neutral"
            ),
        }

    if indicator in ("all", "atr") and len(prices) > 1:
        tr_values = []
        for i in range(1, len(prices)):
            tr = abs(prices[i] - prices[i - 1])
            if volumes and i < len(volumes):
                high_low = abs(prices[i] - prices[i - 1])
                tr = max(tr, high_low)
            tr_values.append(tr)
        atr = sum(tr_values[-14:]) / min(len(tr_values), 14)
        result["atr"] = {
            "atr_14": round(atr, 2),
            "atr_pct": round(atr / prices[-1] * 100, 2)
            if prices[-1] > 0 else 0,
        }

    if indicator in ("all", "obv") and volumes and len(volumes) == len(prices):
        obv = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv += volumes[i]
            elif prices[i] < prices[i - 1]:
                obv -= volumes[i]
        result["obv"] = {"obv": obv}

    return result


def _compute_ema(prices: list[float], period: int) -> float:
    """Compute the Exponential Moving Average.

    Parameters
    ----------
    prices:
        List of closing prices.
    period:
        EMA period.

    Returns
    -------
    float
        The EMA value.
    """
    if not prices:
        return 0.0
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]:
        ema = (p - ema) * multiplier + ema
    return ema


# ============================================================
# Earnings Analyzer
# ============================================================

@tool
def earnings_analyzer(
    revenue: list[float],
    net_profit: list[float],
    quarters: Optional[list[str]] = None,
    symbol: str = "UNKNOWN",
) -> dict[str, Any]:
    """Analyse quarterly earnings trends, margins, and growth rates.

    Parameters
    ----------
    revenue:
        Quarterly revenue figures (INR Cr or as provided).
    net_profit:
        Quarterly net profit figures.
    quarters:
        Optional quarter labels, e.g. ``["Q1FY24", "Q2FY24", ...]``.
    symbol:
        Ticker symbol.

    Returns
    -------
    dict
        Earnings analysis with growth rates, margins, and trends.
    """
    if not revenue or not net_profit or len(revenue) != len(net_profit):
        return {"error": "Revenue and profit lists must be non-empty and equal length"}

    n = len(revenue)
    latest_revenue = revenue[-1]
    latest_profit = net_profit[-1]
    profit_margin = (
        (latest_profit / latest_revenue * 100) if latest_revenue > 0 else 0
    )

    # YoY / QoQ growth
    growth_qoq_rev = (
        ((revenue[-1] - revenue[-2]) / revenue[-2] * 100)
        if n >= 2 and revenue[-2] != 0 else None
    )
    growth_qoq_profit = (
        ((net_profit[-1] - net_profit[-2]) / net_profit[-2] * 100)
        if n >= 2 and net_profit[-2] != 0 else None
    )
    growth_yoy_rev = (
        ((revenue[-1] - revenue[-4]) / revenue[-4] * 100)
        if n >= 4 and revenue[-4] != 0 else None
    )
    growth_yoy_profit = (
        ((net_profit[-1] - net_profit[-4]) / net_profit[-4] * 100)
        if n >= 4 and net_profit[-4] != 0 else None
    )

    # Margin trend
    margins = [
        (net_profit[i] / revenue[i] * 100)
        for i in range(n) if revenue[i] > 0
    ]
    avg_margin = sum(margins) / len(margins) if margins else 0
    margin_trend = "improving" if len(margins) >= 2 and margins[-1] > margins[0] else "declining"

    # Consistency check
    profitable_quarters = sum(1 for p in net_profit if p > 0)
    consistency = profitable_quarters / n * 100

    # CAGR (if 4+ quarters)
    cagr = None
    if n >= 4 and revenue[0] > 0 and revenue[-1] > 0:
        years = (n - 1) / 4
        cagr = ((revenue[-1] / revenue[0]) ** (1 / years) - 1) * 100

    return {
        "symbol": symbol,
        "quarters_analysed": n,
        "latest_quarter": quarters[-1] if quarters else f"Q{n}",
        "latest": {
            "revenue": round(latest_revenue, 2),
            "net_profit": round(latest_profit, 2),
            "profit_margin_pct": round(profit_margin, 2),
        },
        "growth": {
            "qoq_revenue_pct": round(growth_qoq_rev, 2) if growth_qoq_rev is not None else None,
            "qoq_profit_pct": round(growth_qoq_profit, 2) if growth_qoq_profit is not None else None,
            "yoy_revenue_pct": round(growth_yoy_rev, 2) if growth_yoy_rev is not None else None,
            "yoy_profit_pct": round(growth_yoy_profit, 2) if growth_yoy_profit is not None else None,
        },
        "margins": {
            "average_pct": round(avg_margin, 2),
            "trend": margin_trend,
            "latest_pct": round(margins[-1], 2) if margins else 0,
        },
        "consistency": {
            "profitable_quarters": profitable_quarters,
            "consistency_pct": round(consistency, 1),
        },
        "revenue_cagr_pct": round(cagr, 2) if cagr is not None else None,
        "quarterly_data": [
            {
                "quarter": quarters[i] if quarters and i < len(quarters) else f"Q{i + 1}",
                "revenue": round(revenue[i], 2),
                "net_profit": round(net_profit[i], 2),
                "margin_pct": round(margins[i], 2) if i < len(margins) else 0,
            }
            for i in range(n)
        ],
    }


# ============================================================
# Sector Comparator
# ============================================================

@tool
def sector_comparator(
    sector_data: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compare metrics across sectors or industry groups.

    Parameters
    ----------
    sector_data:
        Mapping of sector name to its metrics, e.g.::

            {
                "IT": {"pe": 28.5, "pb": 7.2, "div_yield": 1.5, "roe": 25.0, "debt_equity": 0.1},
                "Banking": {"pe": 18.2, "pb": 2.8, "div_yield": 2.1, "roe": 14.5, "debt_equity": 0.8},
            }

    Returns
    -------
    dict
        Sector rankings and comparative analysis.
    """
    if not sector_data:
        return {"error": "No sector data provided"}

    sectors = list(sector_data.keys())
    metrics = list(sector_data[sectors[0]].keys()) if sectors else []

    # Rankings for each metric (lower is better for PE, D/E; higher is better for yield, ROE)
    lower_is_better = {"pe", "pb", "debt_equity", "p_e", "p_b"}
    rankings: dict[str, list[str]] = {}

    for metric in metrics:
        sorted_sectors = sorted(
            sectors,
            key=lambda s: sector_data[s].get(metric, 0),
            reverse=(metric not in lower_is_better),
        )
        rankings[metric] = sorted_sectors

    # Compute averages across sectors
    metric_avgs: dict[str, float] = {}
    for metric in metrics:
        values = [sector_data[s].get(metric, 0) for s in sectors]
        metric_avgs[metric] = round(sum(values) / len(values), 2) if values else 0

    # Identify best/worst sectors per metric
    best_performing: dict[str, str] = {}
    worst_performing: dict[str, str] = {}
    for metric in metrics:
        sorted_by_metric = sorted(
            sectors,
            key=lambda s: sector_data[s].get(metric, 0),
            reverse=(metric not in lower_is_better),
        )
        if sorted_by_metric:
            best_performing[metric] = sorted_by_metric[0]
            worst_performing[metric] = sorted_by_metric[-1]

    return {
        "sectors_analysed": sectors,
        "metrics_compared": metrics,
        "sector_data": sector_data,
        "market_averages": metric_avgs,
        "rankings": rankings,
        "best_performing": best_performing,
        "worst_performing": worst_performing,
        "summary": [
            f"{metric}: Best = {best_performing.get(metric, 'N/A')}, "
            f"Worst = {worst_performing.get(metric, 'N/A')}, "
            f"Avg = {metric_avgs.get(metric, 'N/A')}"
            for metric in metrics
        ],
    }


# ============================================================
# Export all tools
# ============================================================

MARKET_ANALYST_TOOLS = [
    price_analyzer,
    technical_indicator_calculator,
    earnings_analyzer,
    sector_comparator,
]
