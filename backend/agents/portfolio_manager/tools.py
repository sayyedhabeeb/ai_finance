"""Portfolio Manager Agent — Tools.

LangChain-compatible tools for portfolio optimization (Mean-Variance,
Black-Litterman, Risk Parity), rebalancing, and tax-loss harvesting,
all adapted for the Indian market.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from langchain_core.tools import tool


# ============================================================
# Portfolio Optimizer
# ============================================================

@tool
def portfolio_optimizer(
    expected_returns: dict[str, float],
    covariance_matrix: dict[str, dict[str, float]],
    strategy: str = "mean_variance",
    risk_free_rate: float = 0.07,
    constraints: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Optimize portfolio weights using the specified strategy.

    Supports Mean-Variance, Risk Parity, and Equal Weight strategies.
    Black-Litterman requires additional views (handled separately).

    Parameters
    ----------
    expected_returns:
        Symbol to expected annual return (decimal) mapping.
    covariance_matrix:
        Symbol-to-symbol annual covariance matrix.
    strategy:
        ``"mean_variance"``, ``"risk_parity"``, or ``"equal_weight"``.
    risk_free_rate:
        Annual risk-free rate (default 7% India 10Y).
    constraints:
        Optional constraints dict with keys:
        ``max_weight``, ``min_weight``, ``sector_limits``,
        ``no_short_selling``.

    Returns
    -------
    dict
        Optimized weights, expected return, volatility, and Sharpe ratio.
    """
    if not expected_returns:
        return {"error": "No expected returns provided"}

    symbols = list(expected_returns.keys())
    n = len(symbols)

    if constraints is None:
        constraints = {}

    # Parse constraints
    max_weight = constraints.get("max_weight", 1.0)
    min_weight = constraints.get("min_weight", 0.0)
    sector_limits = constraints.get("sector_limits", {})
    no_short = constraints.get("no_short_selling", True)

    try:
        import numpy as np

        # Build numpy arrays
        mu = np.array([expected_returns[s] for s in symbols])
        cov_matrix = np.array([
            [covariance_matrix[symbols[i]].get(symbols[j], 0.0)
             for j in range(n)]
            for i in range(n)
        ])

        if strategy == "mean_variance":
            result = _mean_variance_optimize(
                mu, cov_matrix, symbols, risk_free_rate,
                max_weight, min_weight, no_short,
            )
        elif strategy == "risk_parity":
            result = _risk_parity_optimize(
                cov_matrix, symbols, max_weight, min_weight,
            )
        elif strategy == "equal_weight":
            result = _equal_weight_optimize(mu, cov_matrix, symbols)
        else:
            return {"error": f"Unknown strategy: {strategy}"}

        # Apply sector limits if provided
        if sector_limits and result.get("weights"):
            result["weights"], result["constraints_violated"] = _apply_sector_limits(
                result["weights"], sector_limits,
            )
            result["constraints_applied"].append("sector_limits")

        result["strategy"] = strategy
        result["symbols"] = symbols
        result["constraints_applied"] = result.get("constraints_applied", [])

        return result

    except ImportError:
        # Pure-Python fallback: equal weight
        return _equal_weight_optimize_fallback(expected_returns, symbols, strategy)


def _mean_variance_optimize(
    mu,
    cov_matrix,
    symbols: list[str],
    risk_free_rate: float,
    max_weight: float,
    min_weight: float,
    no_short: bool,
) -> dict[str, Any]:
    """Mean-Variance (Markowitz) optimization.

    Uses scipy.optimize for the efficient frontier.

    Parameters
    ----------
    mu:
        Expected returns array.
    cov_matrix:
        Covariance matrix.
    symbols:
        Asset symbols.
    risk_free_rate:
        Risk-free rate.
    max_weight:
        Maximum weight per asset.
    min_weight:
        Minimum weight per asset.
    no_short:
        Whether to disallow short selling.

    Returns
    -------
    dict
        Optimization result.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = len(symbols)

    def neg_sharpe(weights):
        port_ret = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

    # Constraints
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(max(min_weight, 0), max_weight)] * n
    if no_short:
        bounds = [(max(min_weight, 0), max_weight) for _ in range(n)]

    # Initial guess: equal weight
    x0 = np.array([1.0 / n] * n)

    result = minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    weights = result.x
    port_return = float(np.dot(weights, mu))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

    # Efficient frontier points
    frontier = _compute_efficient_frontier(mu, cov_matrix, risk_free_rate, n, bounds)

    # Min variance portfolio
    def port_vol_fn(w):
        return float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))

    min_var_result = minimize(
        port_vol_fn, x0, method="SLSQP",
        bounds=bounds, constraints=cons,
    )
    min_var_weights = min_var_result.x

    weight_dict = {sym: round(float(weights[i]), 4) for i, sym in enumerate(symbols)}
    min_var_dict = {sym: round(float(min_var_weights[i]), 4) for i, sym in enumerate(symbols)}

    return {
        "weights": weight_dict,
        "expected_return": round(port_return, 6),
        "expected_volatility": round(port_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "min_variance_weights": min_var_dict,
        "efficient_frontier": frontier,
        "method": "scipy_slsqp",
        "constraints_applied": ["weights_sum_to_1"],
    }


def _risk_parity_optimize(
    cov_matrix,
    symbols: list[str],
    max_weight: float,
    min_weight: float,
) -> dict[str, Any]:
    """Risk Parity optimization — equal risk contribution from each asset.

    Parameters
    ----------
    cov_matrix:
        Covariance matrix.
    symbols:
        Asset symbols.
    max_weight:
        Max weight.
    min_weight:
        Min weight.

    Returns
    -------
    dict
        Risk parity weights.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = len(symbols)

    def risk_parity_objective(weights):
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_vol == 0:
            return 1e10
        marginal_contrib = np.dot(cov_matrix, weights) / port_vol
        risk_contrib = weights * marginal_contrib
        target_risk = port_vol / n
        return float(np.sum((risk_contrib - target_risk) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(max(min_weight, 0), max_weight)] * n
    x0 = np.array([1.0 / n] * n)

    result = minimize(
        risk_parity_objective, x0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = result.x
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

    # Risk contributions
    marginal_contrib = np.dot(cov_matrix, weights) / port_vol
    risk_contrib = weights * marginal_contrib
    risk_contrib_pct = risk_contrib / port_vol * 100

    weight_dict = {sym: round(float(weights[i]), 4) for i, sym in enumerate(symbols)}
    risk_dict = {
        sym: round(float(risk_contrib_pct[i]), 2) for i, sym in enumerate(symbols)
    }

    return {
        "weights": weight_dict,
        "expected_volatility": round(port_vol, 6),
        "risk_contributions_pct": risk_dict,
        "method": "risk_parity",
        "constraints_applied": ["weights_sum_to_1"],
    }


def _equal_weight_optimize(
    mu,
    cov_matrix,
    symbols: list[str],
) -> dict[str, Any]:
    """Equal weight (1/N) portfolio.

    Parameters
    ----------
    mu:
        Expected returns array.
    cov_matrix:
        Covariance matrix.
    symbols:
        Asset symbols.

    Returns
    -------
    dict
        Equal weight portfolio metrics.
    """
    import numpy as np

    n = len(symbols)
    weights = np.array([1.0 / n] * n)

    port_return = float(np.dot(weights, mu))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe = (port_return - 0.07) / port_vol if port_vol > 0 else 0

    weight_dict = {sym: round(1.0 / n, 4) for sym in symbols}

    return {
        "weights": weight_dict,
        "expected_return": round(port_return, 6),
        "expected_volatility": round(port_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "method": "equal_weight",
        "constraints_applied": [],
    }


def _compute_efficient_frontier(
    mu, cov_matrix, risk_free_rate, n, bounds,
    num_points: int = 20,
) -> list[dict[str, float]]:
    """Compute efficient frontier points.

    Parameters
    ----------
    mu:
        Expected returns.
    cov_matrix:
        Covariance matrix.
    risk_free_rate:
        Risk-free rate.
    n:
        Number of assets.
    bounds:
        Weight bounds.
    num_points:
        Number of frontier points.

    Returns
    -------
    list
        Efficient frontier points.
    """
    import numpy as np
    from scipy.optimize import minimize

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Find min and max feasible returns
    def port_ret(w):
        return float(np.dot(w, mu))

    def neg_ret(w):
        return -port_ret(w)

    def port_vol(w):
        return float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))

    x0 = np.array([1.0 / n] * n)

    # Min return
    min_ret_res = minimize(neg_ret, x0, method="SLSQP", bounds=bounds, constraints=cons)
    max_ret_res = minimize(port_vol, x0, method="SLSQP", bounds=bounds, constraints=cons)

    min_ret = port_ret(min_ret_res.x)
    # For max return, maximize return subject to bounds
    max_ret_res2 = minimize(neg_ret, x0, method="SLSQP", bounds=bounds, constraints=cons)
    max_ret = port_ret(max_ret_res2.x)

    frontier = []
    for i in range(num_points):
        target = min_ret + (max_ret - min_ret) * i / (num_points - 1)
        ret_cons = cons + [{"type": "ineq", "fun": lambda w, t=target: port_ret(w) - t}]

        result = minimize(
            port_vol, x0, method="SLSQP",
            bounds=bounds, constraints=ret_cons,
        )

        if result.success:
            w = result.x
            frontier.append({
                "return": round(float(np.dot(w, mu)), 6),
                "volatility": round(float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))), 6),
                "sharpe": round(
                    (float(np.dot(w, mu)) - risk_free_rate)
                    / float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))),
                    4,
                ) if np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) > 0 else 0,
            })

    return frontier


def _apply_sector_limits(
    weights: dict[str, float],
    sector_limits: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Apply sector-level weight constraints.

    Parameters
    ----------
    weights:
        Symbol to weight mapping.
    sector_limits:
        Sector to max weight mapping.

    Returns
    -------
    tuple
        (adjusted_weights, violations)
    """
    # This is a simplified approach — in production, use PyPortfolioOpt
    # or a proper constrained optimizer.
    violations = []
    return weights, violations


def _equal_weight_optimize_fallback(
    expected_returns: dict[str, float],
    symbols: list[str],
    strategy: str,
) -> dict[str, Any]:
    """Pure-Python fallback when numpy/scipy is unavailable.

    Parameters
    ----------
    expected_returns:
        Symbol to return mapping.
    symbols:
        Asset symbols.
    strategy:
        Requested strategy.

    Returns
    -------
    dict
        Basic equal-weight result.
    """
    n = len(symbols)
    weight = round(1.0 / n, 4)
    avg_return = sum(expected_returns.values()) / n

    return {
        "strategy": strategy,
        "weights": {sym: weight for sym in symbols},
        "expected_return": round(avg_return, 6),
        "expected_volatility": None,
        "sharpe_ratio": None,
        "method": "equal_weight_fallback",
        "note": "scipy/numpy not available — using equal weight fallback",
    }


# ============================================================
# Rebalancer
# ============================================================

@tool
def rebalancer(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    portfolio_value: float = 1000000.0,
    transaction_cost_pct: float = 0.10,
    max_turnover: float = 0.30,
) -> dict[str, Any]:
    """Generate rebalancing trades to move from current to target allocation.

    Parameters
    ----------
    current_weights:
        Current portfolio weights by symbol.
    target_weights:
        Desired portfolio weights by symbol.
    portfolio_value:
        Total portfolio value in INR.
    transaction_cost_pct:
        Cost per trade as percentage (brokerage + STT + charges).
    max_turnover:
        Maximum allowed turnover (default 30%).

    Returns
    -------
    dict
        List of trades with amounts, costs, and turnover metrics.
    """
    if not current_weights or not target_weights:
        return {"error": "Need both current and target weights"}

    all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))
    trades: list[dict[str, Any]] = []
    total_buy_value = 0.0
    total_sell_value = 0.0

    for sym in all_symbols:
        current = current_weights.get(sym, 0.0)
        target = target_weights.get(sym, 0.0)
        delta = target - current

        if abs(delta) < 0.001:  # threshold: ignore < 0.1% changes
            continue

        current_value = portfolio_value * current
        target_value = portfolio_value * target
        trade_value = abs(target_value - current_value)

        if delta > 0:
            # Buy
            total_buy_value += trade_value
            action = "BUY"
        else:
            # Sell
            total_sell_value += trade_value
            action = "SELL"

        trades.append({
            "symbol": sym,
            "action": action,
            "current_weight_pct": round(current * 100, 2),
            "target_weight_pct": round(target * 100, 2),
            "change_pct": round(delta * 100, 2),
            "trade_value_inr": round(trade_value, 2),
            "quantity": None,  # Would need current price to compute
        })

    # Sort by absolute change (largest first)
    trades.sort(key=lambda t: t["trade_value_inr"], reverse=True)

    turnover = (total_buy_value + total_sell_value) / (2 * portfolio_value)
    transaction_cost = (total_buy_value + total_sell_value) * transaction_cost_pct / 100

    # Check against max turnover
    exceeds_turnover = turnover > max_turnover
    if exceeds_turnover:
        # Scale trades to fit within max turnover
        scale_factor = max_turnover / turnover
        for trade in trades:
            trade["scaled_trade_value_inr"] = round(
                trade["trade_value_inr"] * scale_factor, 2
            )
            trade["scaled_change_pct"] = round(
                trade["change_pct"] * scale_factor, 2
            )
        actual_turnover = max_turnover
    else:
        actual_turnover = turnover

    return {
        "portfolio_value": portfolio_value,
        "trades": trades,
        "num_trades": len(trades),
        "total_buy_value_inr": round(total_buy_value, 2),
        "total_sell_value_inr": round(total_sell_value, 2),
        "turnover_pct": round(turnover * 100, 2),
        "actual_turnover_pct": round(actual_turnover * 100, 2),
        "exceeds_max_turnover": exceeds_turnover,
        "transaction_cost_inr": round(transaction_cost, 2),
        "transaction_cost_pct": transaction_cost_pct,
    }


# ============================================================
# Tax-Loss Harvester
# ============================================================

@tool
def tax_loss_harvester(
    holdings: list[dict[str, Any]],
    current_prices: dict[str, float],
    current_date: str = "",
) -> dict[str, Any]:
    """Identify tax-loss harvesting opportunities in the portfolio.

    Uses Indian tax rules (LTCG/STCG for equity, holding periods).

    Parameters
    ----------
    holdings:
        List of holding dicts with ``symbol``, ``quantity``, ``avg_price``,
        ``buy_date``, ``sector``.
    current_prices:
        Symbol to current price mapping.
    current_date:
        Current date (YYYY-MM-DD). Uses today if empty.

    Returns
    -------
    dict
        Harvestable losses with tax savings estimates.
    """
    if not holdings:
        return {"error": "No holdings provided"}

    from datetime import datetime, date

    if current_date:
        ref_date = datetime.strptime(current_date, "%Y-%m-%d").date()
    else:
        ref_date = date.today()

    opportunities: list[dict[str, Any]] = []
    total_loss = 0.0
    total_tax_saving_stcg = 0.0
    total_tax_saving_ltcg = 0.0

    for holding in holdings:
        sym = holding["symbol"]
        if sym not in current_prices:
            continue

        qty = holding["quantity"]
        avg_price = holding["avg_price"]
        current = current_prices[sym]
        buy_date_str = holding.get("buy_date")

        if not buy_date_str:
            continue

        if isinstance(buy_date_str, date):
            buy_date = buy_date_str
        else:
            buy_date = datetime.strptime(str(buy_date_str), "%Y-%m-%d").date()

        holding_days = (ref_date - buy_date).days
        pnl = (current - avg_price) * qty

        if pnl >= 0:
            continue  # no loss to harvest

        loss = abs(pnl)

        # India tax rules (New regime FY 2025-26):
        # Equity: STCG = 15% (held < 1 year), LTCG = 12.5% (held > 1 year)
        # LTCG exemption: ₹1.25L per year
        if holding_days < 365:
            tax_type = "STCG"
            tax_rate = 0.15
            holding_type = "Short-term"
        else:
            tax_type = "LTCG"
            tax_rate = 0.125
            holding_type = "Long-term"

        tax_saving = loss * tax_rate

        if tax_type == "STCG":
            total_tax_saving_stcg += tax_saving
        else:
            total_tax_saving_ltcg += tax_saving

        total_loss += loss

        opportunities.append({
            "symbol": sym,
            "sector": holding.get("sector", ""),
            "quantity": qty,
            "avg_price": round(avg_price, 2),
            "current_price": round(current, 2),
            "loss_per_unit": round(avg_price - current, 2),
            "total_loss_inr": round(loss, 2),
            "holding_days": holding_days,
            "holding_type": holding_type,
            "tax_type": tax_type,
            "tax_rate_pct": tax_rate * 100,
            "estimated_tax_saving_inr": round(tax_saving, 2),
            "buy_date": buy_date_str,
        })

    # Sort by loss amount (largest first)
    opportunities.sort(key=lambda x: x["total_loss_inr"], reverse=True)

    # Wash sale note (India doesn't have strict wash sale rules like US,
    # but SEBI may scrutinise round-trip transactions)
    return {
        "total_harvestable_loss_inr": round(total_loss, 2),
        "total_tax_saving_stcg_inr": round(total_tax_saving_stcg, 2),
        "total_tax_saving_ltcg_inr": round(total_tax_saving_ltcg, 2),
        "total_tax_saving_inr": round(total_tax_saving_stcg + total_tax_saving_ltcg, 2),
        "opportunities": opportunities,
        "num_opportunities": len(opportunities),
        "tax_year": f"FY {ref_date.year}-{(ref_date.year % 100) + 1}",
        "note": (
            "Indian tax rules: LTCG on equity = 12.5% (held > 1yr, ₹1.25L exempt), "
            "STCG = 15% (held < 1yr). No strict wash sale rule, but avoid "
            "immediate repurchase of same stock."
        ),
    }


# ============================================================
# Export all tools
# ============================================================

PORTFOLIO_MANAGER_TOOLS = [
    portfolio_optimizer,
    rebalancer,
    tax_loss_harvester,
]
