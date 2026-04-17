# ============================================================
# AI Financial Brain — Portfolio Manager Agent
# ============================================================
"""
Portfolio Manager Agent — portfolio construction, optimisation,
rebalancing, and tax-loss harvesting with India-specific constraints.

Uses scipy/numpy for Mean-Variance and Risk Parity optimisation
with PyPortfolioOpt integration where available.  Implements
Black-Litterman views, India sector limits, and STCG/LTCG tax rules.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from agents.portfolio_manager.tools import (
    PORTFOLIO_MANAGER_TOOLS,
    portfolio_optimizer,
    rebalancer,
    tax_loss_harvester,
)
from config.schemas import (
    AgentResult,
    AgentTask,
    AgentType,
    OptimizationStrategy,
    Portfolio,
    PortfolioHolding,
    PortfolioOptimizationResult,
    RebalancingAction,
)

logger = structlog.get_logger(__name__)


class PortfolioManagerAgent(BaseAgent):
    """Portfolio Manager Agent.

    Provides comprehensive portfolio management:
    - Multi-strategy optimisation (Mean-Variance, Black-Litterman, Risk Parity)
    - Efficient frontier computation
    - Portfolio rebalancing with turnover and transaction cost analysis
    - Tax-loss harvesting under Indian tax rules (LTCG/STCG)
    - Strategy backtesting with periodic rebalancing
    - India-specific constraints (sector limits, no short selling, STT)
    """

    agent_type: AgentType = AgentType.PORTFOLIO_MANAGER
    name: str = "Portfolio Manager"
    description: str = (
        "Portfolio manager specialising in portfolio construction, "
        "optimization, rebalancing, and tax-loss harvesting for Indian markets."
    )

    # ------------------------------------------------------------------
    # India market constants
    # ------------------------------------------------------------------

    INDIA_SECTOR_LIMITS: dict[str, float] = {
        "BANKING": 0.25,
        "IT": 0.25,
        "OIL_GAS": 0.20,
        "PHARMA": 0.15,
        "FMCG": 0.15,
        "AUTO": 0.15,
        "REALTY": 0.10,
        "INFRA": 0.15,
        "CONSUMER_GOODS": 0.15,
    }

    INDIA_SINGLE_STOCK_LIMIT: float = 0.10
    INDIA_RISK_FREE_RATE: float = 0.07  # 10Y G-Sec yield

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT: str = """\
You are an expert portfolio manager with deep expertise in Indian equity \
markets. You construct, optimise, and rebalance portfolios using quantitative \
methods while adhering to India-specific regulations and constraints.

## Core Capabilities
1. **Portfolio Optimisation** — Mean-Variance (Markowitz), Black-Litterman \
with investor views, Risk Parity (equal risk contribution), and Equal Weight.
2. **Efficient Frontier** — Generate and visualise the efficient frontier with \
tangency portfolio identification.
3. **Rebalancing** — Generate trades with turnover constraints (≤30%), \
transaction cost analysis, and tax impact estimation.
4. **Tax-Loss Harvesting** — Identify harvestable losses using Indian tax rules \
(LTCG 12.5% > 1yr, STCG 20% < 1yr, ₹1.25L LTCG exemption).
5. **Backtesting** — Evaluate historical strategy performance with buy-and-hold \
comparison.
6. **Risk Budgeting** — Allocate risk (not just capital) across assets.

## India-Specific Rules
- **LTCG on Equity**: 12.5% above ₹1.25L annual exemption (held > 12 months)
- **STCG on Equity**: 20% (held < 12 months)
- **Securities Transaction Tax (STT)**: 0.1% on sell-side (equity delivery)
- **Stamp Duty**: 0.015% on buy-side (varies by state)
- **Exchange Transaction Charges**: ~0.00345% (NSE) per side
- **DP Charges**: ₹15.93 per script per sell transaction (NSDL/CDSL)
- **Brokerage**: Zero for most discount brokers (Zerodha, Groww)
- **No Short Selling**: Indian retail investors cannot short sell equities
- **T+1 Settlement**: Trade settles next business day
- **Sector Concentration**: RBI/SEBI guidelines for certain fund categories
- **Single Stock Limit**: ≤ 10% for mutual fund diversification norms

## Optimisation Constraints
- Weights must sum to 1.0 (fully invested)
- No negative weights (no short selling)
- Maximum single stock weight: 10%
- Sector-level concentration limits apply
- Minimum trade size: ₹5,000 (practical minimum for retail)
- Maximum turnover: 30% per rebalancing event

## Important Guidelines
- Always show the trade-off between return and risk
- Include all transaction costs in rebalancing analysis
- Factor in tax implications (STCG/LTCG) of every trade
- Suggest specific ETFs/index funds for implementation (Nifty BeES, Junior BeES)
- Present both conservative and aggressive scenarios
- Flag if constraints lead to sub-optimal solutions
- Always include a disclaimer that this is not investment advice
"""

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list:
        """Return the list of tools available to this agent."""
        return PORTFOLIO_MANAGER_TOOLS

    def get_tools(self) -> list:
        """Return the list of tools registered for this agent."""
        return PORTFOLIO_MANAGER_TOOLS

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
        """Execute a portfolio management task."""
        return await self._run_with_metrics(task, self._execute_inner)

    async def _execute_inner(self, task: AgentTask) -> AgentResult:
        """Inner execution with tool orchestration and LLM reasoning."""
        context_str = self._format_context(task.context)
        enhanced_query = task.query
        if context_str:
            enhanced_query = f"{task.query}\n\n## Portfolio Data\n{context_str}"

        # Invoke relevant tools
        tool_results = self._invoke_relevant_tools(task)

        # Build prompt
        messages = [SystemMessage(content=self._SYSTEM_PROMPT)]
        user_content = enhanced_query
        if tool_results:
            tool_summary = self._format_tool_results(tool_results)
            user_content += f"\n\n## Quantitative Analysis\n{tool_summary}"

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
                "query_type": "portfolio",
                "model": self._llm_model,
            },
        )

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _invoke_relevant_tools(self, task: AgentTask) -> dict[str, Any]:
        """Detect intent and invoke appropriate portfolio tools."""
        query_lower = task.query.lower()
        context = task.context
        results: dict[str, Any] = {}

        # --- Optimisation ---
        opt_keywords = [
            "optimize", "optimise", "allocation", "weights", "best portfolio",
            "efficient", "sharpe", "mean-variance", "risk parity",
            "black-litterman", "optimal",
        ]
        if any(kw in query_lower for kw in opt_keywords):
            expected_returns = context.get("expected_returns", {})
            cov_matrix = context.get("covariance_matrix", {})
            if expected_returns and cov_matrix:
                strategy = context.get("strategy", "mean_variance")
                results["optimization"] = portfolio_optimizer.invoke({
                    "expected_returns": expected_returns,
                    "covariance_matrix": cov_matrix,
                    "strategy": strategy,
                    "risk_free_rate": context.get("risk_free_rate", self.INDIA_RISK_FREE_RATE),
                    "constraints": {
                        "max_weight": self.INDIA_SINGLE_STOCK_LIMIT,
                        "sector_limits": self.INDIA_SECTOR_LIMITS,
                        "no_short_selling": True,
                    },
                })

        # --- Rebalancing ---
        rebal_keywords = [
            "rebalance", "re-align", "drift", "adjust",
            "current vs target", "rebalancing",
        ]
        if any(kw in query_lower for kw in rebal_keywords):
            current_w = context.get("current_weights", {})
            target_w = context.get("target_weights", {})
            if current_w and target_w:
                results["rebalancing"] = rebalancer.invoke({
                    "current_weights": current_w,
                    "target_weights": target_w,
                    "portfolio_value": context.get("portfolio_value", 1000000),
                })

        # --- Tax-loss harvesting ---
        tax_keywords = [
            "tax loss", "harvest", "loss harvesting", "tax saving",
            "book loss", "stcg", "ltcg", "tax-loss",
        ]
        if any(kw in query_lower for kw in tax_keywords):
            holdings = context.get("holdings", [])
            prices = context.get("current_prices", {})
            if holdings and prices:
                results["tax_harvest"] = tax_loss_harvester.invoke({
                    "holdings": holdings,
                    "current_prices": prices,
                })

        return results

    # ------------------------------------------------------------------
    # Specialised public methods
    # ------------------------------------------------------------------

    def optimize_portfolio(
        self,
        expected_returns: dict[str, float],
        covariance_matrix: dict[str, dict[str, float]],
        strategy: str = "mean_variance",
        risk_free_rate: Optional[float] = None,
        sector_limits: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        """Optimise portfolio with India-specific constraints.

        Parameters
        ----------
        expected_returns:
            Symbol to expected annual return mapping.
        covariance_matrix:
            Covariance matrix dict.
        strategy:
            Optimisation strategy.
        risk_free_rate:
            Risk-free rate.
        sector_limits:
            Custom sector limits.

        Returns
        -------
        dict
            Optimisation result.
        """
        limits = sector_limits or self.INDIA_SECTOR_LIMITS
        rf = risk_free_rate or self.INDIA_RISK_FREE_RATE

        result = portfolio_optimizer.invoke({
            "expected_returns": expected_returns,
            "covariance_matrix": covariance_matrix,
            "strategy": strategy,
            "risk_free_rate": rf,
            "constraints": {
                "max_weight": self.INDIA_SINGLE_STOCK_LIMIT,
                "sector_limits": limits,
                "no_short_selling": True,
            },
        })

        # Add India-specific annotations
        if "weights" in result:
            result["india_annotations"] = {
                "sector_limits_applied": limits,
                "single_stock_limit": self.INDIA_SINGLE_STOCK_LIMIT,
                "no_short_selling": True,
                "risk_free_rate": rf,
                "tax_note": (
                    "Rebalancing may trigger STCG (20%) or LTCG (12.5% above ₹1.25L). "
                    "Consider tax impact before trading."
                ),
            }

        return result

    def suggest_rebalancing(
        self,
        portfolio: dict[str, Any],
        target_weights: dict[str, float],
        max_turnover: float = 0.30,
    ) -> dict[str, Any]:
        """Suggest portfolio rebalancing trades.

        Parameters
        ----------
        portfolio:
            Portfolio data with holdings and total value.
        target_weights:
            Desired allocation weights.
        max_turnover:
            Maximum allowed turnover.

        Returns
        -------
        dict
            Rebalancing recommendations.
        """
        current_weights = portfolio.get("weights", {})
        portfolio_value = portfolio.get("total_value", 1000000)

        result = rebalancer.invoke({
            "current_weights": current_weights,
            "target_weights": target_weights,
            "portfolio_value": portfolio_value,
            "max_turnover": max_turnover,
        })

        # Tax impact estimation
        tax_impact = self._estimate_rebalancing_tax_impact(
            portfolio.get("holdings", []),
            result.get("trades", []),
        )
        result["tax_impact"] = tax_impact

        return result

    def harvest_tax_losses(
        self,
        holdings: list[dict[str, Any]],
        current_prices: dict[str, float],
        reference_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Identify and recommend tax-loss harvesting opportunities.

        Parameters
        ----------
        holdings:
            List of holding dicts.
        current_prices:
            Current price per symbol.
        reference_date:
            Date for holding period calculation.

        Returns
        -------
        dict
            Tax-loss harvesting recommendations.
        """
        result = tax_loss_harvester.invoke({
            "holdings": holdings,
            "current_prices": current_prices,
            "current_date": reference_date or "",
        })

        # Add execution strategy
        if result.get("opportunities"):
            result["execution_strategy"] = [
                "Sell loss-making positions before 31st March (end of FY)",
                "Wait 24-48 hours before repurchasing to avoid SEBI scrutiny",
                "Consider purchasing equivalent ETF (e.g., Nifty BeES instead of individual stocks)",
                "Track harvested losses for set-off against gains in the same FY",
                "LTCG losses can be carried forward for 8 years",
                "STCG losses can be set off against both STCG and LTCG gains",
            ]
            result["priority_order"] = sorted(
                result["opportunities"],
                key=lambda x: x["estimated_tax_saving_inr"],
                reverse=True,
            )

        return result

    def backtest_strategy(
        self,
        prices: dict[str, list[float]],
        weights: dict[str, float],
        rebalance_freq: str = "monthly",
        initial_investment: float = 1000000.0,
    ) -> dict[str, Any]:
        """Backtest a portfolio strategy with periodic rebalancing.

        Parameters
        ----------
        prices:
            Symbol to price series mapping.
        weights:
            Target allocation weights.
        rebalance_freq:
            ``"daily"``, ``"weekly"``, ``"monthly"``, ``"quarterly"``, or ``"never"``.
        initial_investment:
            Starting capital in INR.

        Returns
        -------
        dict
            Backtest results with performance metrics.
        """
        import numpy as np

        symbols = list(weights.keys())
        if not symbols:
            return {"error": "No symbols provided"}

        min_len = min(len(prices.get(s, [])) for s in symbols)
        if min_len < 20:
            return {"error": "Need at least 20 data points per symbol"}

        freq_days = {
            "daily": 1, "weekly": 5, "monthly": 22,
            "quarterly": 66, "never": min_len,
        }
        rebal_interval = freq_days.get(rebalance_freq, 22)

        # Normalise weights
        total_w = sum(weights.values())
        norm_weights = {s: weights[s] / total_w for s in symbols}

        # Build aligned price matrix
        n_days = min_len
        price_matrix = np.array([prices[s][:n_days] for s in symbols])

        # Calculate returns
        returns_matrix = np.diff(price_matrix, axis=1) / price_matrix[:, :-1]
        n_returns = returns_matrix.shape[1]

        # Simulate portfolio
        portfolio_value = initial_investment
        values = [initial_investment]
        unit_counts = {
            s: portfolio_value * norm_weights[s] / price_matrix[symbols.index(s), 0]
            for s in symbols
        }

        for day in range(n_returns):
            day_value = sum(
                unit_counts[s] * price_matrix[symbols.index(s), day + 1]
                for s in symbols
            )
            portfolio_value = day_value
            values.append(portfolio_value)

            if (day + 1) % rebal_interval == 0:
                for s in symbols:
                    idx = symbols.index(s)
                    current_price = price_matrix[idx, day + 1]
                    unit_counts[s] = portfolio_value * norm_weights[s] / current_price

        values_arr = np.array(values)
        returns_arr = np.diff(values_arr) / values_arr[:-1]

        # Performance metrics
        total_return = (values_arr[-1] / values_arr[0]) - 1
        n_years = n_days / 252
        cagr = (values_arr[-1] / values_arr[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

        daily_std = float(np.std(returns_arr, ddof=1))
        annual_std = daily_std * np.sqrt(252) if len(returns_arr) > 1 else 0
        annual_return = float(np.mean(returns_arr)) * 252
        sharpe = (annual_return - self.INDIA_RISK_FREE_RATE) / annual_std if annual_std > 0 else 0

        # Max drawdown
        cummax = np.maximum.accumulate(values_arr)
        drawdowns = (values_arr - cummax) / cummax
        max_dd = float(np.min(drawdowns))

        # Buy & hold comparison
        bh_units = {
            s: initial_investment * norm_weights[s] / price_matrix[symbols.index(s), 0]
            for s in symbols
        }
        bh_values = [
            sum(bh_units[s] * price_matrix[symbols.index(s), day] for s in symbols)
            for day in range(n_days)
        ]
        bh_return = (bh_values[-1] / bh_values[0]) - 1

        return {
            "strategy": f"{rebalance_freq.capitalize()} rebalanced",
            "initial_investment": initial_investment,
            "final_value": round(float(values_arr[-1]), 2),
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "annualized_return_pct": round(annual_return * 100, 2),
            "annualized_volatility_pct": round(annual_std * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "trading_days": n_days,
            "num_rebalancings": n_days // rebal_interval,
            "buy_and_hold_return_pct": round(bh_return * 100, 2),
            "alpha_vs_buy_hold_pct": round((total_return - bh_return) * 100, 2),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_rebalancing_tax_impact(
        holdings: list[dict[str, Any]],
        trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Estimate the tax impact of rebalancing trades.

        Parameters
        ----------
        holdings:
            Current holdings.
        trades:
            Rebalancing trades.

        Returns
        -------
        dict
            Tax impact estimation.
        """
        ref_date = date.today()
        total_stcg_tax = 0.0
        total_ltcg_tax = 0.0

        holding_map = {h["symbol"]: h for h in holdings}

        for trade in trades:
            sym = trade["symbol"]
            if sym not in holding_map or trade["action"] != "SELL":
                continue

            h = holding_map[sym]
            avg_price = h["avg_price"]
            current = h.get("current_price", avg_price)
            pnl_per_unit = current - avg_price

            buy_date = h.get("buy_date")
            if buy_date:
                if isinstance(buy_date, date):
                    holding_days = (ref_date - buy_date).days
                else:
                    holding_days = (
                        ref_date
                        - datetime.strptime(str(buy_date), "%Y-%m-%d").date()
                    ).days
            else:
                holding_days = 0

            trade_value = trade["trade_value_inr"]
            qty = trade_value / current if current > 0 else 0
            pnl = pnl_per_unit * qty

            if pnl <= 0:
                continue

            if holding_days < 365:
                tax = pnl * 0.20  # STCG 20%
                total_stcg_tax += tax
            else:
                taxable = max(pnl - 125000, 0)
                tax = taxable * 0.125  # LTCG 12.5%
                total_ltcg_tax += tax

        return {
            "estimated_stcg_tax_inr": round(total_stcg_tax, 2),
            "estimated_ltcg_tax_inr": round(total_ltcg_tax, 2),
            "total_tax_inr": round(total_stcg_tax + total_ltcg_tax, 2),
            "note": (
                "Estimated using FY 2025-26 rates: STCG 20%, LTCG 12.5% "
                "(with ₹1.25L annual LTCG exemption). Actual tax depends on "
                "total income, deductions, and chosen tax regime."
            ),
        }

    @staticmethod
    def _estimate_confidence(
        task: AgentTask,
        response_text: str,
        data: dict[str, Any],
    ) -> float:
        """Estimate confidence for portfolio recommendations."""
        confidence = 0.65

        if data:
            confidence += 0.15

        if task.context.get("expected_returns") and task.context.get("covariance_matrix"):
            confidence += 0.10

        if len(response_text) > 300:
            confidence += 0.05

        disclaimer_markers = [
            "not investment advice", "past performance",
            "consult your financial advisor", "do your own research",
        ]
        if any(m in response_text.lower() for m in disclaimer_markers):
            confidence += 0.05

        return max(0.0, min(confidence, 1.0))
