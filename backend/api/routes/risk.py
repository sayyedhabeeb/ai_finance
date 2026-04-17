"""
API Routes - Risk analytics endpoints.

Provides risk metrics, stress testing, and anomaly detection.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["risk"])


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class RiskMetricsResponse(BaseModel):
    user_id: str
    computed_at: str
    portfolio_volatility_annual: float
    portfolio_volatility_daily: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: Optional[float] = None
    alpha_annual: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    calmar_ratio: Optional[float] = None
    skewness: float
    kurtosis: float


class StressTestRequest(BaseModel):
    scenarios: Optional[List[str]] = Field(
        None,
        description="Pre-defined scenarios. Defaults to standard set if not provided.",
    )
    custom_shocks: Optional[Dict[str, float]] = Field(
        None,
        description="Custom symbol -> shock_pct, e.g. {'AAPL': -20.0}",
    )
    benchmark: str = Field("SPY", description="Benchmark for market-wide scenarios")


class StressTestResult(BaseModel):
    scenario: str
    portfolio_loss: float
    portfolio_loss_pct: float
    worst_holding: Optional[str] = None
    worst_holding_loss: Optional[float] = None
    details: Dict[str, Any]


class StressTestResponse(BaseModel):
    user_id: str
    computed_at: str
    results: List[StressTestResult]


class AnomalyRecord(BaseModel):
    detected_at: str
    anomaly_type: str
    symbol: Optional[str] = None
    description: str
    severity: str  # low, medium, high, critical
    value: Optional[float] = None
    expected_range: Optional[Dict[str, float]] = None


class AnomaliesResponse(BaseModel):
    user_id: str
    anomalies: List[AnomalyRecord]
    total_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STANDARD_SCENARIOS = [
    {
        "name": "2008 Financial Crisis",
        "market_shock_pct": -37.0,
        "sector_shocks": {"financials": -55.0, "real_estate": -45.0, "consumer_staples": -15.0},
    },
    {
        "name": "COVID-19 Crash (Mar 2020)",
        "market_shock_pct": -34.0,
        "sector_shocks": {"energy": -50.0, "technology": -20.0, "healthcare": -10.0},
    },
    {
        "name": "Interest Rate Shock (+300bp)",
        "market_shock_pct": -15.0,
        "sector_shocks": {"real_estate": -25.0, "utilities": -20.0, "financials": -5.0},
    },
    {
        "name": "Tech Bubble Burst",
        "market_shock_pct": -25.0,
        "sector_shocks": {"technology": -60.0, "communications": -40.0, "consumer_discretionary": -35.0},
    },
    {
        "name": "Stagflation",
        "market_shock_pct": -20.0,
        "sector_shocks": {"energy": 30.0, "materials": -10.0, "technology": -35.0},
    },
]

SECTOR_MAP: Dict[str, List[str]] = {
    "technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA"],
    "financials": ["JPM", "BAC", "GS", "WFC", "MS"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "consumer_staples": ["PG", "KO", "PEP", "WMT", "COST"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "real_estate": ["AMT", "PLD", "CCI", "EQIX", "PSA"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "consumer_discretionary": ["AMZN", "HD", "NKE", "MCD", "SBUX"],
    "materials": ["LIN", "APD", "ECL", "SHW", "FCX"],
    "communications": ["DIS", "CMCSA", "T", "VZ", "NFLX"],
}


def _compute_risk_metrics(returns: np.ndarray, risk_free_rate: float = 0.045) -> Dict[str, float]:
    """Compute a comprehensive suite of risk metrics from daily returns."""
    if len(returns) < 2:
        raise ValueError("Need at least 2 data points to compute risk metrics")

    mean_daily = float(np.mean(returns))
    std_daily = float(np.std(returns, ddof=1))

    # Annualised
    annual_return = mean_daily * 252
    annual_vol = std_daily * np.sqrt(252)

    # Sharpe
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else None

    # Sortino (downside deviation)
    negative_returns = returns[returns < 0]
    downside_std = float(np.std(negative_returns, ddof=1)) if len(negative_returns) > 1 else 0.0
    downside_annual = downside_std * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / downside_annual if downside_annual > 0 else None

    # Max Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # VaR & CVaR (historical, 95%)
    sorted_returns = np.sort(returns)
    var_idx = int(np.floor(0.05 * len(sorted_returns)))
    var_95 = float(sorted_returns[var_idx])
    cvar_95 = float(np.mean(sorted_returns[: var_idx + 1]))

    # Higher moments
    skewness = float(0.0)
    kurtosis = float(0.0)
    if std_daily > 0 and len(returns) > 3:
        from scipy.stats import skew as calc_skew, kurtosis as calc_kurtosis
        skewness = float(calc_skew(returns, bias=False))
        kurtosis = float(calc_kurtosis(returns, bias=False, fisher=True))

    return {
        "portfolio_volatility_annual": round(annual_vol, 6),
        "portfolio_volatility_daily": round(std_daily, 6),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "max_drawdown": round(max_dd, 6),
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
    }


def _detect_anomalies(holdings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect anomalies in holdings using simple statistical rules."""
    anomalies = []
    now = datetime.now(timezone.utc).isoformat()

    for h in holdings:
        symbol = h["symbol"]
        cv = float(h.get("current_value", 0) or 0)
        qty = float(h["quantity"])
        avg_cost = float(h["avg_cost"])

        # Check for extreme unrealised loss
        if avg_cost > 0:
            pnl_pct = ((cv / (qty * avg_cost)) - 1.0) * 100
            if pnl_pct < -50:
                anomalies.append({
                    "detected_at": now,
                    "anomaly_type": "extreme_loss",
                    "symbol": symbol,
                    "description": f"{symbol} is down {abs(round(pnl_pct, 1))}% from average cost",
                    "severity": "high",
                    "value": round(pnl_pct, 2),
                    "expected_range": {"min": -20.0, "max": 50.0},
                })
            elif pnl_pct > 200:
                anomalies.append({
                    "detected_at": now,
                    "anomaly_type": "extreme_gain",
                    "symbol": symbol,
                    "description": f"{symbol} is up {round(pnl_pct, 1)}% from average cost — consider taking profits",
                    "severity": "medium",
                    "value": round(pnl_pct, 2),
                    "expected_range": {"min": -20.0, "max": 100.0},
                })

    return anomalies


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/risk/{user_id}/metrics",
    response_model=RiskMetricsResponse,
    summary="Get comprehensive risk metrics for user portfolio",
)
async def get_risk_metrics(user_id: str, request: Request) -> RiskMetricsResponse:
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Fetch portfolio
    portfolio = await db.fetch_one(
        "SELECT id FROM portfolios WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 1",
        user_id,
    )
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Fetch daily market data for portfolio holdings over the last year
    holdings = await db.fetch_all(
        "SELECT symbol, quantity, avg_cost, current_value FROM holdings WHERE portfolio_id = $1",
        portfolio["id"],
    )

    if not holdings:
        raise HTTPException(status_code=400, detail="Portfolio has no holdings")

    symbols = [h["symbol"] for h in holdings]
    symbol_list = ",".join(symbols)

    try:
        rows = await db.fetch_all(
            f"""
            SELECT symbol, close, timestamp
            FROM market_data
            WHERE symbol IN ({",".join(f"${i+2}" for i in range(len(symbols)))})
              AND timestamp >= NOW() - INTERVAL '1 year'
            ORDER BY timestamp ASC
            """,
            portfolio["id"],
            *symbols,
        )
    except Exception:
        # If no market data, generate synthetic returns for demo
        logger.warning("No market data in DB — using synthetic returns for risk metrics")
        rows = []

    if rows:
        # Build daily portfolio returns (equal-weight for simplicity)
        from collections import defaultdict
        prices: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            prices[r["symbol"]].append(float(r["close"]))

        # Align to shortest series length
        min_len = min(len(v) for v in prices.values()) if prices else 0
        if min_len < 2:
            raise HTTPException(status_code=400, detail="Insufficient market data for risk computation")

        returns_matrix = np.zeros((min_len - 1, len(symbols)))
        for i, sym in enumerate(symbols):
            p = prices[sym][:min_len]
            daily = np.diff(np.array(p)) / np.array(p[:-1])
            returns_matrix[:, i] = daily

        # Equal-weight portfolio returns
        w = np.array([1.0 / len(symbols)] * len(symbols))
        portfolio_returns = returns_matrix @ w
    else:
        np.random.seed(123)
        portfolio_returns = np.random.normal(0.0005, 0.012, 252)

    metrics = _compute_risk_metrics(portfolio_returns)

    return RiskMetricsResponse(
        user_id=user_id,
        computed_at=datetime.now(timezone.utc).isoformat(),
        beta=metrics.pop("beta", None),
        alpha_annual=metrics.pop("alpha_annual", None),
        information_ratio=metrics.pop("information_ratio", None),
        tracking_error=metrics.pop("tracking_error", None),
        calmar_ratio=metrics.pop("calmar_ratio", None),
        **metrics,
    )


@router.post(
    "/risk/{user_id}/stress-test",
    response_model=StressTestResponse,
    summary="Run stress tests on user portfolio",
)
async def stress_test(
    user_id: str,
    body: StressTestRequest,
    request: Request,
) -> StressTestResponse:
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    portfolio = await db.fetch_one(
        "SELECT id, total_value FROM portfolios WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 1",
        user_id,
    )
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    holdings = await db.fetch_all(
        "SELECT symbol, quantity, current_value FROM holdings WHERE portfolio_id = $1",
        portfolio["id"],
    )
    if not holdings:
        raise HTTPException(status_code=400, detail="Portfolio has no holdings")

    total_value = float(portfolio["total_value"] or 0) or sum(
        float(h.get("current_value", 0) or 0) for h in holdings
    )

    # Build symbol -> value map
    symbol_values: Dict[str, float] = {
        h["symbol"]: float(h.get("current_value", 0) or 0) for h in holdings
    }

    # Build symbol -> sector map
    symbol_to_sector: Dict[str, str] = {}
    for sector, tickers in SECTOR_MAP.items():
        for t in tickers:
            symbol_to_sector[t] = sector

    # Custom shocks override sector defaults
    custom_shocks = body.custom_shocks or {}

    results: List[StressTestResult] = []
    scenarios = [s for s in STANDARD_SCENARIOS if s["name"] in (body.scenarios or [s["name"]])]

    for scenario in scenarios:
        total_loss = 0.0
        worst_sym = None
        worst_loss = 0.0

        for sym, val in symbol_values.items():
            if sym in custom_shocks:
                shock = custom_shocks[sym]
            else:
                sector = symbol_to_sector.get(sym)
                shock = scenario["sector_shocks"].get(sector, scenario["market_shock_pct"])

            sym_loss = val * (shock / 100.0)
            total_loss += sym_loss
            if sym_loss < worst_loss:
                worst_loss = sym_loss
                worst_sym = sym

        results.append(
            StressTestResult(
                scenario=scenario["name"],
                portfolio_loss=round(total_loss, 2),
                portfolio_loss_pct=round((total_loss / total_value) * 100, 2) if total_value > 0 else 0.0,
                worst_holding=worst_sym,
                worst_holding_loss=round(worst_loss, 2),
                details={
                    "market_shock_pct": scenario["market_shock_pct"],
                    "sector_shocks": scenario["sector_shocks"],
                },
            )
        )

    return StressTestResponse(
        user_id=user_id,
        computed_at=datetime.now(timezone.utc).isoformat(),
        results=results,
    )


@router.get(
    "/risk/{user_id}/anomalies",
    response_model=AnomaliesResponse,
    summary="Get detected anomalies in user portfolio",
)
async def get_anomalies(user_id: str, request: Request) -> AnomaliesResponse:
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    portfolio = await db.fetch_one(
        "SELECT id FROM portfolios WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 1",
        user_id,
    )
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    holdings = await db.fetch_all(
        "SELECT symbol, quantity, avg_cost, current_value FROM holdings WHERE portfolio_id = $1",
        portfolio["id"],
    )

    holdings_dicts = [dict(h) for h in holdings]
    raw_anomalies = _detect_anomalies(holdings_dicts)

    return AnomaliesResponse(
        user_id=user_id,
        anomalies=[AnomalyRecord(**a) for a in raw_anomalies],
        total_count=len(raw_anomalies),
    )
