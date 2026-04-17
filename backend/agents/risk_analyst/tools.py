"""Risk Analyst Agent — Tools.

LangChain-compatible tools for Value-at-Risk, GARCH volatility modeling,
correlation analysis, stress testing, and anomaly detection.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from langchain_core.tools import tool


# ============================================================
# Value-at-Risk Calculator
# ============================================================

@tool
def var_calculator(
    returns: list[float],
    portfolio_value: float = 1000000.0,
    confidence_levels: Optional[list[float]] = None,
    method: str = "historical",
) -> dict[str, Any]:
    """Calculate Value-at-Risk (VaR) and Conditional VaR for a portfolio.

    Supports historical simulation and parametric (Gaussian) methods.

    Parameters
    ----------
    returns:
        List of daily returns (decimal, e.g. 0.015 for 1.5%).
    portfolio_value:
        Current portfolio value in INR.
    confidence_levels:
        List of confidence levels (e.g. [0.95, 0.99]).
        Defaults to [0.95, 0.99].
    method:
        ``"historical"`` or ``"parametric"``.

    Returns
    -------
    dict
        VaR and CVaR at each confidence level, plus portfolio impact.
    """
    if not returns:
        return {"error": "No returns data provided"}
    if len(returns) < 10:
        return {"error": "Need at least 10 data points for VaR calculation"}

    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    import numpy as np

    returns_arr = np.array(returns, dtype=float)
    n = len(returns_arr)

    # Basic statistics
    mean_return = float(np.mean(returns_arr))
    std_return = float(np.std(returns_arr, ddof=1))
    annualized_vol = std_return * math.sqrt(252)

    results: dict[str, Any] = {
        "portfolio_value": portfolio_value,
        "returns_count": n,
        "mean_daily_return_pct": round(mean_return * 100, 4),
        "daily_volatility_pct": round(std_return * 100, 4),
        "annualized_volatility_pct": round(annualized_vol * 100, 2),
        "method": method,
        "var_results": [],
    }

    for cl in confidence_levels:
        if method == "historical":
            # Historical simulation
            sorted_returns = np.sort(returns_arr)
            index = int((1 - cl) * n)
            index = max(0, min(index, n - 1))
            var_return = float(-sorted_returns[index])

            # CVaR (expected shortfall) — average of losses beyond VaR
            tail_returns = sorted_returns[:index + 1]
            cvar_return = float(-np.mean(tail_returns)) if len(tail_returns) > 0 else var_return

        else:
            # Parametric (Gaussian)
            from scipy.stats import norm

            z_score = float(norm.ppf(1 - cl))
            var_return = -(mean_return + z_score * std_return)
            cvar_return = var_return + std_return * float(norm.pdf(z_score)) / (1 - cl)

        var_amount = portfolio_value * var_return
        cvar_amount = portfolio_value * cvar_return

        results["var_results"].append({
            "confidence_level": cl,
            "var_return_pct": round(var_return * 100, 4),
            "var_amount_inr": round(var_amount, 2),
            "cvar_return_pct": round(cvar_return * 100, 4),
            "cvar_amount_inr": round(cvar_amount, 2),
        })

    return results


# ============================================================
# GARCH Volatility Modeler
# ============================================================

@tool
def garch_modeler(
    returns: list[float],
    p: int = 1,
    q: int = 1,
    forecast_horizon: int = 5,
) -> dict[str, Any]:
    """Fit a GARCH(p, q) model to returns and forecast volatility.

    Uses the ``arch`` library when available; falls back to EWMA.

    Parameters
    ----------
    returns:
        List of daily returns (decimal).
    p:
        GARCH lag order (default 1).
    q:
        ARCH lag order (default 1).
    forecast_horizon:
        Number of days ahead to forecast volatility.

    Returns
    -------
    dict
        GARCH parameters, fitted volatility, and forecast.
    """
    if not returns or len(returns) < 30:
        return {"error": "Need at least 30 data points for GARCH"}

    import numpy as np

    returns_arr = np.array(returns, dtype=float)

    # Try arch library first
    try:
        from arch import arch_model

        am = arch_model(returns_arr * 100, vol="Garch", p=p, q=q, dist="normal")
        res = am.fit(disp="off", show_warning=False)

        # Extract conditional volatility (last 30 days)
        cond_vol = res.conditional_volatility / 100  # back to decimal
        recent_vol = float(cond_vol[-1])

        # Forecast
        forecast = res.forecast(horizon=forecast_horizon)
        forecast_var = forecast.variance.iloc[-1].values / 10000  # back to decimal
        forecast_vol = np.sqrt(forecast_var)

        return {
            "method": "GARCH",
            "model_specification": f"GARCH({p},{q})",
            "parameters": {
                "omega": round(float(res.params.get("omega", 0)), 8),
                "alpha": round(float(res.params.get("alpha[1]", 0)), 6),
                "beta": round(float(res.params.get("beta[1]", 0)), 6),
            },
            "fitted_volatility": {
                "latest_daily_pct": round(float(recent_vol) * 100, 4),
                "latest_annualized_pct": round(float(recent_vol) * math.sqrt(252) * 100, 2),
            },
            "volatility_forecast": [
                {
                    "day_ahead": i + 1,
                    "daily_vol_pct": round(float(v) * 100, 4),
                    "annualized_vol_pct": round(float(v) * math.sqrt(252) * 100, 2),
                }
                for i, v in enumerate(forecast_vol)
            ],
            "model_fit": {
                "aic": round(float(res.aic), 2),
                "bic": round(float(res.bic), 2),
                "log_likelihood": round(float(res.loglikelihood), 2),
            },
            "persistence": round(
                float(res.params.get("alpha[1]", 0))
                + float(res.params.get("beta[1]", 0)),
                4,
            ),
        }

    except ImportError:
        # Fallback: EWMA volatility
        return _ewma_volatility(returns, forecast_horizon)

    except Exception as exc:
        # If GARCH fitting fails, fall back to EWMA
        ewma_result = _ewma_volatility(returns, forecast_horizon)
        ewma_result["garch_error"] = str(exc)
        return ewma_result


def _ewma_volatility(
    returns: list[float],
    forecast_horizon: int = 5,
    lambda_: float = 0.94,
) -> dict[str, Any]:
    """Exponentially Weighted Moving Average volatility model.

    Parameters
    ----------
    returns:
        Daily returns (decimal).
    forecast_horizon:
        Days to forecast.
    lambda_:
        Decay factor (default RiskMetrics λ=0.94).

    Returns
    -------
    dict
        EWMA volatility estimate and forecast.
    """
    import numpy as np

    returns_arr = np.array(returns, dtype=float)
    n = len(returns_arr)

    # Compute EWMA variance
    ewma_var = np.var(returns_arr)  # initial estimate
    for i in range(n):
        ewma_var = lambda_ * ewma_var + (1 - lambda_) * returns_arr[i] ** 2

    current_vol = math.sqrt(ewma_var)

    # Mean-reverting forecast (volatility tends to revert)
    long_run_var = np.var(returns_arr)
    forecasts: list[dict[str, float]] = []
    for d in range(1, forecast_horizon + 1):
        decay = lambda_ ** d
        forecast_var = decay * ewma_var + (1 - decay) * long_run_var
        forecasts.append({
            "day_ahead": d,
            "daily_vol_pct": round(math.sqrt(forecast_var) * 100, 4),
            "annualized_vol_pct": round(
                math.sqrt(forecast_var * 252) * 100, 2
            ),
        })

    return {
        "method": "EWMA",
        "model_specification": f"EWMA(λ={lambda_})",
        "fitted_volatility": {
            "latest_daily_pct": round(current_vol * 100, 4),
            "latest_annualized_pct": round(current_vol * math.sqrt(252) * 100, 2),
        },
        "volatility_forecast": forecasts,
        "lambda": lambda_,
    }


# ============================================================
# Correlation Analyzer
# ============================================================

@tool
def correlation_analyzer(
    returns_data: dict[str, list[float]],
) -> dict[str, Any]:
    """Compute the correlation matrix for multiple assets.

    Parameters
    ----------
    returns_data:
        Mapping of asset name/symbol to its returns series.

    Returns
    -------
    dict
        Correlation matrix, eigenvalues, and insights.
    """
    if not returns_data or len(returns_data) < 2:
        return {"error": "Need at least 2 assets for correlation analysis"}

    import numpy as np

    symbols = list(returns_data.keys())
    min_len = min(len(v) for v in returns_data.values())

    if min_len < 10:
        return {"error": "Each asset needs at least 10 return observations"}

    # Build aligned return matrix
    matrix = np.array([
        returns_data[sym][:min_len]
        for sym in symbols
    ])

    # Pearson correlation
    corr_matrix = np.corrcoef(matrix)

    # Format as dict
    corr_dict: dict[str, dict[str, float]] = {}
    for i, sym_i in enumerate(symbols):
        corr_dict[sym_i] = {}
        for j, sym_j in enumerate(symbols):
            corr_dict[sym_i][sym_j] = round(float(corr_matrix[i, j]), 4)

    # Find highest/lowest correlations
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            pairs.append((symbols[i], symbols[j], corr_matrix[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

    # Eigenvalues for diversification analysis
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = sorted(eigenvalues, reverse=True)

    # Condition number
    cond_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else float("inf")

    insights: list[str] = []
    if pairs_sorted:
        highest = pairs_sorted[0]
        lowest = pairs_sorted[-1]
        insights.append(
            f"Highest correlation: {highest[0]}-{highest[1]} = {highest[2]:.4f}"
        )
        insights.append(
            f"Lowest correlation: {lowest[0]}-{lowest[1]} = {lowest[2]:.4f}"
        )

    if cond_number > 10:
        insights.append(
            f"High condition number ({cond_number:.1f}) — portfolio may be "
            "ill-diversified"
        )

    # Diversification ratio (approximate)
    avg_corr = np.mean(corr_matrix[np.triu_indices(len(symbols), k=1)])
    insights.append(
        f"Average pairwise correlation: {avg_corr:.4f}"
    )

    return {
        "symbols": symbols,
        "observations": min_len,
        "correlation_matrix": corr_dict,
        "eigenvalues": [round(float(e), 6) for e in eigenvalues],
        "condition_number": round(float(cond_number), 4),
        "average_correlation": round(float(avg_corr), 4),
        "highest_correlation": (
            {"pair": f"{pairs_sorted[0][0]}-{pairs_sorted[0][1]}",
             "value": round(pairs_sorted[0][2], 4)}
            if pairs_sorted else None
        ),
        "lowest_correlation": (
            {"pair": f"{pairs_sorted[-1][0]}-{pairs_sorted[-1][1]}",
             "value": round(pairs_sorted[-1][2], 4)}
            if pairs_sorted else None
        ),
        "insights": insights,
    }


# ============================================================
# Stress Tester
# ============================================================

@tool
def stress_tester(
    portfolio_value: float,
    holdings: dict[str, dict[str, float]],
    scenarios: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run stress tests on a portfolio under various market scenarios.

    Parameters
    ----------
    portfolio_value:
        Total portfolio value in INR.
    holdings:
        Mapping of symbol to ``{"weight": float, "beta": float}``.
    scenarios:
        Custom stress scenarios.  If *None*, uses standard India-market scenarios.

    Returns
    -------
    dict
        Scenario-by-scenario impact and worst-case assessment.
    """
    if not holdings:
        return {"error": "No holdings provided"}

    if scenarios is None:
        scenarios = [
            {
                "name": "2008 Global Financial Crisis",
                "market_drop_pct": -35.0,
                "description": "Severe global market crash",
            },
            {
                "name": "2020 COVID Crash",
                "market_drop_pct": -38.0,
                "description": "Pandemic-driven crash",
            },
            {
                "name": "RBI Rate Hike (+200bps)",
                "market_drop_pct": -10.0,
                "description": "Aggressive monetary tightening",
            },
            {
                "name": "Rupee Depreciation (-10%)",
                "market_drop_pct": -5.0,
                "description": "Significant currency depreciation",
            },
            {
                "name": "Sector-specific IT Crash (-25%)",
                "market_drop_pct": -8.0,
                "description": "IT sector downturn (US recession)",
            },
            {
                "name": "FII Panic Selling",
                "market_drop_pct": -15.0,
                "description": "Large-scale foreign outflows",
            },
            {
                "name": "Moderate Correction (-10%)",
                "market_drop_pct": -10.0,
                "description": "Normal market correction",
            },
        ]

    results: list[dict[str, Any]] = []

    for scenario in scenarios:
        market_drop = scenario["market_drop_pct"] / 100  # convert to decimal

        # Portfolio loss = weighted average of (beta * market_drop)
        weighted_loss = 0.0
        holding_impacts: list[dict[str, Any]] = []

        for symbol, data in holdings.items():
            beta = data.get("beta", 1.0)
            weight = data.get("weight", 0.0)
            holding_loss = beta * market_drop * weight
            weighted_loss += holding_loss

            holding_impacts.append({
                "symbol": symbol,
                "weight_pct": round(weight * 100, 2),
                "beta": beta,
                "estimated_loss_pct": round(beta * market_drop * 100, 2),
                "estimated_loss_inr": round(portfolio_value * holding_loss, 2),
            })

        portfolio_loss_pct = weighted_loss * 100
        remaining_value = portfolio_value * (1 + weighted_loss)

        results.append({
            "scenario": scenario["name"],
            "description": scenario.get("description", ""),
            "market_drop_pct": scenario["market_drop_pct"],
            "portfolio_loss_pct": round(portfolio_loss_pct, 2),
            "loss_amount_inr": round(abs(portfolio_value * weighted_loss), 2),
            "remaining_value_inr": round(remaining_value, 2),
            "holding_impacts": holding_impacts,
        })

    # Find worst case
    worst = max(results, key=lambda r: r["portfolio_loss_pct"])

    # Risk level classification
    max_loss = worst["portfolio_loss_pct"]
    if max_loss > 30:
        risk_level = "critical"
    elif max_loss > 20:
        risk_level = "very_high"
    elif max_loss > 10:
        risk_level = "high"
    elif max_loss > 5:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "portfolio_value": portfolio_value,
        "scenarios_tested": len(scenarios),
        "results": results,
        "worst_case": {
            "scenario": worst["scenario"],
            "loss_pct": worst["portfolio_loss_pct"],
            "loss_amount": worst["loss_amount_inr"],
        },
        "risk_level": risk_level,
        "recommendation": (
            "Portfolio is adequately diversified against stress scenarios."
            if max_loss < 15
            else "Consider adding hedges (gold, bonds) to reduce stress-test losses."
        ),
    }


# ============================================================
# Anomaly Detector (Isolation Forest)
# ============================================================

@tool
def anomaly_detector(
    returns_data: dict[str, list[float]],
    contamination: float = 0.05,
) -> dict[str, Any]:
    """Detect anomalous returns using Isolation Forest.

    Parameters
    ----------
    returns_data:
        Mapping of symbol to returns series.
    contamination:
        Expected proportion of anomalies (default 5%).

    Returns
    -------
    dict
        Anomaly flags per symbol with details.
    """
    if not returns_data:
        return {"error": "No returns data provided"}

    import numpy as np

    symbols = list(returns_data.keys())
    min_len = min(len(v) for v in returns_data.values())

    if min_len < 20:
        return {"error": "Need at least 20 observations per asset"}

    # Build feature matrix: each row is a time point, columns are asset returns
    matrix = np.array([
        returns_data[sym][-min_len:]
        for sym in symbols
    ]).T  # shape: (n_observations, n_assets)

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Scale features
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        labels = iso_forest.fit_predict(scaled)
        scores = iso_forest.decision_function(scaled)

        # Identify anomalies (label == -1)
        anomaly_indices = np.where(labels == -1)[0]

        # Analyze anomalies
        anomalies: list[dict[str, Any]] = []
        for idx in anomaly_indices:
            row_data = matrix[idx]
            # Find which assets had extreme values
            extreme_assets = []
            for i, sym in enumerate(symbols):
                z_score = abs(scaled[idx, i])
                if z_score > 2.0:
                    extreme_assets.append({
                        "symbol": sym,
                        "return_pct": round(float(row_data[i]) * 100, 4),
                        "z_score": round(float(z_score), 2),
                    })

            anomalies.append({
                "time_index": int(idx),
                "anomaly_score": round(float(scores[idx]), 4),
                "extreme_assets": extreme_assets,
                "returns": {
                    sym: round(float(row_data[i]) * 100, 4)
                    for i, sym in enumerate(symbols)
                },
            })

        # Per-symbol anomaly summary
        symbol_anomalies: dict[str, int] = {}
        for sym_idx, sym in enumerate(symbols):
            anom_count = 0
            for idx in anomaly_indices:
                if abs(scaled[idx, sym_idx]) > 2.0:
                    anom_count += 1
            symbol_anomalies[sym] = anom_count

        return {
            "method": "isolation_forest",
            "symbols_analyzed": symbols,
            "observations": min_len,
            "contamination": contamination,
            "total_anomalies": len(anomaly_indices),
            "anomaly_rate_pct": round(len(anomaly_indices) / min_len * 100, 2),
            "anomalies": anomalies,
            "anomalies_per_symbol": symbol_anomalies,
            "most_anomalous_symbols": sorted(
                symbol_anomalies.items(), key=lambda x: x[1], reverse=True
            )[:3],
        }

    except ImportError:
        # Fallback: Z-score based anomaly detection
        return _zscore_anomaly_detection(returns_data, symbols, min_len)


def _zscore_anomaly_detection(
    returns_data: dict[str, list[float]],
    symbols: list[str],
    min_len: int,
    threshold: float = 3.0,
) -> dict[str, Any]:
    """Simple Z-score based anomaly detection as fallback.

    Parameters
    ----------
    returns_data:
        Symbol to returns mapping.
    symbols:
        Symbol list.
    min_len:
        Minimum data length.
    threshold:
        Z-score threshold for anomaly.

    Returns
    -------
    dict
        Anomaly detection results.
    """
    import numpy as np

    anomalies: list[dict[str, Any]] = []
    symbol_anomalies: dict[str, int] = {sym: 0 for sym in symbols}

    for sym in symbols:
        rets = np.array(returns_data[sym][-min_len:], dtype=float)
        mean = np.mean(rets)
        std = np.std(rets)

        if std == 0:
            continue

        for i, r in enumerate(rets):
            z = abs((r - mean) / std)
            if z > threshold:
                anomalies.append({
                    "time_index": i,
                    "symbol": sym,
                    "return_pct": round(float(r) * 100, 4),
                    "z_score": round(float(z), 2),
                })
                symbol_anomalies[sym] += 1

    return {
        "method": "z_score",
        "symbols_analyzed": symbols,
        "observations": min_len,
        "threshold": threshold,
        "total_anomalies": len(anomalies),
        "anomalies": anomalies[:20],  # limit output
        "anomalies_per_symbol": symbol_anomalies,
        "note": "Using Z-score fallback (scikit-learn not available)",
    }


# ============================================================
# Export all tools
# ============================================================

RISK_ANALYST_TOOLS = [
    var_calculator,
    garch_modeler,
    correlation_analyzer,
    stress_tester,
    anomaly_detector,
]
