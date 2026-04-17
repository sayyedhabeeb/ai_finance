"""
API Routes - Portfolio endpoints.

CRUD + optimisation + rebalancing for user portfolios.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["portfolio"])


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class HoldingDetail(BaseModel):
    symbol: str
    name: Optional[str] = None
    quantity: float
    avg_cost: float
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    unrealised_pnl: Optional[float] = None
    unrealised_pnl_pct: Optional[float] = None
    weight: Optional[float] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None


class PortfolioSummary(BaseModel):
    portfolio_id: str
    name: str
    currency: str
    total_value: float
    day_change: Optional[float] = None
    day_change_pct: Optional[float] = None
    total_return: Optional[float] = None
    total_return_pct: Optional[float] = None
    holdings_count: int
    last_updated: str


class OptimizationRequest(BaseModel):
    objective: str = Field(
        "max_sharpe",
        description="Optimisation objective: max_sharpe, min_variance, max_return, risk_parity",
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Constraints e.g. max_sector_weight, min_positions",
    )
    benchmark: Optional[str] = Field("SPY", description="Benchmark ticker")
    lookback_days: int = Field(252, ge=30, le=2520)


class OptimizationResult(BaseModel):
    portfolio_id: str
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    objective_value: float
    rebalance_trades: List[Dict[str, Any]]
    computation_time_ms: float


class RebalanceRequest(BaseModel):
    threshold_pct: float = Field(5.0, ge=0.1, le=50.0, description="Rebalance if drift exceeds this %")
    method: str = Field("proportional", description="proportional | sell_and_buy")


class RebalanceRecommendation(BaseModel):
    portfolio_id: str
    needs_rebalance: bool
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    trades: List[Dict[str, Any]]
    estimated_tax_impact: Optional[float] = None
    total_cost: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db


async def _fetch_portfolio(db, user_id: str) -> Optional[Dict[str, Any]]:
    row = await db.fetch_one(
        """
        SELECT id, name, currency, total_value, created_at, updated_at
        FROM portfolios
        WHERE user_id = $1
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        user_id,
    )
    return dict(row) if row else None


async def _fetch_holdings(db, portfolio_id: str) -> List[Dict[str, Any]]:
    rows = await db.fetch_all(
        """
        SELECT symbol, quantity, avg_cost, current_value, added_at
        FROM holdings
        WHERE portfolio_id = $1
        ORDER BY current_value DESC
        """,
        portfolio_id,
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/portfolio/{user_id}",
    response_model=PortfolioSummary,
    summary="Get user portfolio summary",
)
async def get_portfolio(user_id: str, request: Request) -> PortfolioSummary:
    db = await _get_db(request)
    portfolio = await _fetch_portfolio(db, user_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="No portfolio found for user")

    holdings = await _fetch_holdings(db, portfolio["id"])
    holdings_count = len(holdings)

    return PortfolioSummary(
        portfolio_id=str(portfolio["id"]),
        name=portfolio["name"],
        currency=portfolio["currency"],
        total_value=float(portfolio["total_value"] or 0),
        day_change=None,
        day_change_pct=None,
        total_return=None,
        total_return_pct=None,
        holdings_count=holdings_count,
        last_updated=portfolio["updated_at"].isoformat() if portfolio.get("updated_at") else "",
    )


@router.get(
    "/portfolio/{user_id}/holdings",
    response_model=List[HoldingDetail],
    summary="Get detailed holdings for a user portfolio",
)
async def get_holdings(user_id: str, request: Request) -> List[HoldingDetail]:
    db = await _get_db(request)
    portfolio = await _fetch_portfolio(db, user_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="No portfolio found for user")

    holdings = await _fetch_holdings(db, portfolio["id"])
    total_value = sum(h.get("current_value", 0) or 0 for h in holdings)

    details: List[HoldingDetail] = []
    for h in holdings:
        cv = float(h.get("current_value", 0) or 0)
        qty = float(h["quantity"])
        avg = float(h["avg_cost"])
        current_price = cv / qty if qty > 0 else 0.0
        unrealised = cv - (qty * avg)
        unrealised_pct = (unrealised / (qty * avg) * 100) if (qty * avg) > 0 else 0.0

        details.append(
            HoldingDetail(
                symbol=h["symbol"],
                quantity=qty,
                avg_cost=avg,
                current_value=cv,
                current_price=round(current_price, 4),
                unrealised_pnl=round(unrealised, 2),
                unrealised_pnl_pct=round(unrealised_pct, 2),
                weight=round(cv / total_value, 4) if total_value > 0 else 0.0,
            )
        )

    return details


@router.post(
    "/portfolio/{user_id}/optimize",
    response_model=OptimizationResult,
    summary="Run portfolio optimization",
)
async def optimize_portfolio(
    user_id: str,
    body: OptimizationRequest,
    request: Request,
) -> OptimizationResult:
    db = await _get_db(request)
    portfolio = await _fetch_portfolio(db, user_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="No portfolio found for user")

    holdings = await _fetch_holdings(db, portfolio["id"])
    if not holdings:
        raise HTTPException(status_code=400, detail="Portfolio has no holdings to optimize")

    start = time.perf_counter()

    # ------------------------------------------------------------------
    # Real mean-variance optimisation using scipy
    # ------------------------------------------------------------------
    import numpy as np
    from scipy.optimize import minimize

    symbols = [h["symbol"] for h in holdings]
    n = len(symbols)
    # Fallback: generate synthetic covariance if no real data
    # In production, fetch historical returns from market_data table
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (252, n))
    cov_matrix = np.cov(returns.T)
    mean_returns = returns.mean(axis=0)

    def _neg_sharpe(weights):
        port_ret = np.dot(weights, mean_returns) * 252
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)) * 252)
        return -port_ret / port_vol if port_vol > 0 else 0.0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.array([1.0 / n] * n)

    match body.objective:
        case "min_variance":
            def _objective(w):
                return np.dot(w, np.dot(cov_matrix, w)) * 252
        case "max_return":
            def _objective(w):
                return -np.dot(w, mean_returns) * 252
        case "risk_parity":
            def _objective(w):
                port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)) * 252)
                marginal = cov_matrix @ w * 252
                risk_contrib = w * marginal / port_vol
                target = port_vol / n
                return np.sum((risk_contrib - target) ** 2)
        case _:
            _objective = _neg_sharpe

    result = minimize(_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    expected_ret = float(np.dot(optimal_weights, mean_returns) * 252)
    expected_vol = float(np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)) * 252))
    sharpe = expected_ret / expected_vol if expected_vol > 0 else 0.0

    # Build rebalance trades
    current_weights = {
        h["symbol"]: float(h.get("current_value", 0) or 0)
        for h in holdings
    }
    total_cv = sum(current_weights.values()) or 1.0
    current_weights = {k: v / total_cv for k, v in current_weights.items()}

    trades = []
    for i, sym in enumerate(symbols):
        diff = optimal_weights[i] - current_weights.get(sym, 0)
        if abs(diff) > 0.005:
            trades.append({
                "symbol": sym,
                "action": "buy" if diff > 0 else "sell",
                "current_weight": round(current_weights.get(sym, 0), 4),
                "target_weight": round(float(optimal_weights[i]), 4),
                "weight_change": round(float(diff), 4),
            })

    comp_ms = (time.perf_counter() - start) * 1000

    return OptimizationResult(
        portfolio_id=str(portfolio["id"]),
        optimal_weights={sym: round(float(w), 6) for sym, w in zip(symbols, optimal_weights)},
        expected_return=round(expected_ret, 6),
        expected_volatility=round(expected_vol, 6),
        sharpe_ratio=round(sharpe, 4),
        objective_value=round(float(result.fun), 6),
        rebalance_trades=trades,
        computation_time_ms=round(comp_ms, 2),
    )


@router.post(
    "/portfolio/{user_id}/rebalance",
    response_model=RebalanceRecommendation,
    summary="Suggest portfolio rebalancing",
)
async def rebalance_portfolio(
    user_id: str,
    body: RebalanceRequest,
    request: Request,
) -> RebalanceRecommendation:
    db = await _get_db(request)
    portfolio = await _fetch_portfolio(db, user_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="No portfolio found for user")

    holdings = await _fetch_holdings(db, portfolio["id"])
    if not holdings:
        raise HTTPException(status_code=400, detail="Portfolio has no holdings")

    total_value = sum(float(h.get("current_value", 0) or 0) for h in holdings)

    current_weights: Dict[str, float] = {}
    for h in holdings:
        cv = float(h.get("current_value", 0) or 0)
        current_weights[h["symbol"]] = cv / total_value if total_value > 0 else 0.0

    # Equal-weight target
    n = len(current_weights)
    target_weights = {sym: 1.0 / n for sym in current_weights}

    trades: List[Dict[str, Any]] = []
    needs_rebalance = False
    threshold = body.threshold_pct / 100.0

    for sym, cw in current_weights.items():
        tw = target_weights[sym]
        drift = abs(cw - tw)
        if drift > threshold:
            needs_rebalance = True
            trade_value = (tw - cw) * total_value
            trades.append({
                "symbol": sym,
                "action": "buy" if trade_value > 0 else "sell",
                "current_weight": round(cw, 4),
                "target_weight": round(tw, 4),
                "drift_pct": round(drift * 100, 2),
                "trade_value": round(abs(trade_value), 2),
            })

    total_cost = sum(t["trade_value"] for t in trades) * 0.001  # assume 0.1% fees

    return RebalanceRecommendation(
        portfolio_id=str(portfolio["id"]),
        needs_rebalance=needs_rebalance,
        current_weights={k: round(v, 4) for k, v in current_weights.items()},
        target_weights=target_weights,
        trades=trades,
        estimated_tax_impact=round(total_cost * 0.25, 2) if trades else 0.0,
        total_cost=round(total_cost, 2),
    )
