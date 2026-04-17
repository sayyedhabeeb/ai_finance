"""
Portfolio rebalancing engine.

Computes drift from target allocation, generates trade lists, minimises
turnover, and evaluates the tax impact of rebalancing decisions.

Tax-aware rebalancing considers:
  - LTCG vs STCG classification for each position
  - The 12.5 % vs 20 % rate differential for equity
  - Remaining days to LTCG eligibility
  - Tax-loss harvesting opportunities

Typical usage::

    rebalancer = PortfolioRebalancer()
    trades = rebalancer.generate_rebalancing_trades(current, target, total_value=10_00_000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RebalanceTrade:
    """A trade recommendation from the rebalancer."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float  # Current market price (INR)
    value: float  # Trade value (INR)
    weight_change: float  # Change in portfolio weight
    priority: str  # "REQUIRED", "RECOMMENDED", "OPTIONAL"
    reason: str
    tax_impact: float = 0.0  # Estimated tax from this trade (if SELL)
    days_to_ltcg: int = -1  # Days until LTCG eligibility (if equity SELL)


@dataclass
class DriftReport:
    """Drift analysis for a single asset."""

    symbol: str
    current_weight: float
    target_weight: float
    absolute_drift: float  # target - current
    relative_drift_pct: float  # (current - target) / target * 100
    action: str  # "BUY", "SELL", "HOLD"


# ---------------------------------------------------------------------------
# Main rebalancer
# ---------------------------------------------------------------------------

class PortfolioRebalancer:
    """Rebalances portfolio based on target allocation and constraints.

    Parameters
    ----------
    drift_threshold : float
        Minimum drift (as fraction of target weight) to trigger a trade.
        Default ``0.05`` = 5 % relative drift.
    max_turnover : float
        Maximum portfolio turnover per rebalance.  Default ``0.30`` = 30 %.
    tax_aware : bool
        If ``True``, considers tax impact when prioritising trades.
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        max_turnover: float = 0.30,
        tax_aware: bool = True,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.max_turnover = max_turnover
        self.tax_aware = tax_aware

    # ------------------------------------------------------------------
    # Drift calculation
    # ------------------------------------------------------------------

    def calculate_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, DriftReport]:
        """Calculate drift between current and target allocations.

        Parameters
        ----------
        current_weights : dict[str, float]
            Symbol -> current weight (should sum to ~1.0).
        target_weights : dict[str, float]
            Symbol -> target weight.

        Returns
        -------
        dict[str, DriftReport]
            Mapping of symbol -> drift report.
        """
        # Normalise current weights
        current_total = sum(current_weights.values())
        norm_current = {
            s: w / current_total for s, w in current_weights.items() if current_total > 0
        }

        reports: dict[str, DriftReport] = {}
        all_symbols = set(list(norm_current.keys()) + list(target_weights.keys()))

        for sym in all_symbols:
            cw = norm_current.get(sym, 0.0)
            tw = target_weights.get(sym, 0.0)
            abs_drift = tw - cw

            if tw > 0:
                rel_drift = ((cw - tw) / tw) * 100
            else:
                rel_drift = cw * 100  # Completely overweight if target is 0

            if abs_drift > self.drift_threshold * tw:
                action = "BUY" if abs_drift > 0 else "SELL"
            elif abs_drift < -self.drift_threshold * max(tw, 0.01):
                action = "SELL"
            else:
                action = "HOLD"

            reports[sym] = DriftReport(
                symbol=sym,
                current_weight=round(cw, 6),
                target_weight=round(tw, 6),
                absolute_drift=round(abs_drift, 6),
                relative_drift_pct=round(rel_drift, 2),
                action=action,
            )

        return reports

    def needs_rebalancing(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> bool:
        """Check whether the portfolio needs rebalancing.

        Returns ``True`` if any asset's drift exceeds the threshold.
        """
        reports = self.calculate_drift(current_weights, target_weights)
        return any(r.action != "HOLD" for r in reports.values())

    # ------------------------------------------------------------------
    # Trade generation
    # ------------------------------------------------------------------

    def generate_rebalancing_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        total_value: float,
        prices: Optional[dict[str, float]] = None,
        holdings_days: Optional[dict[str, int]] = None,
        sector_map: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Generate a list of trades to rebalance the portfolio.

        Parameters
        ----------
        current_weights : dict[str, float]
            Current asset weights.
        target_weights : dict[str, float]
            Target asset weights.
        total_value : float
            Total portfolio value in INR.
        prices : dict[str, float] or None
            Current market prices per symbol.  Required to compute quantities.
        holdings_days : dict[str, int] or None
            Days held per symbol (for tax-aware rebalancing).
        sector_map : dict[str, str] or None
            Symbol -> sector (for constraint checking).

        Returns
        -------
        dict
            ``{
                "trades": [RebalanceTrade, ...],
                "turnover": float,
                "estimated_tax_impact": float,
                "drift_report": {symbol: DriftReport},
                "summary": str,
            }``
        """
        drift = self.calculate_drift(current_weights, target_weights)

        # Calculate raw trades
        raw_trades: list[RebalanceTrade] = []
        for sym, report in drift.items():
            if report.action == "HOLD":
                continue

            trade_value = abs(report.absolute_drift) * total_value
            price = prices.get(sym, 1.0) if prices else 1.0
            quantity = trade_value / price if price > 0 else 0

            side = "BUY" if report.absolute_drift > 0 else "SELL"
            priority = "REQUIRED" if abs(report.relative_drift_pct) > 15 else "RECOMMENDED"
            reason = f"Drift {report.relative_drift_pct:+.1f}% from target"

            # Tax-aware analysis for SELLs
            tax_impact = 0.0
            days_to_ltcg = -1
            if side == "SELL" and self.tax_aware and holdings_days:
                held = holdings_days.get(sym, 0)
                days_to_ltcg = max(0, 365 - held)
                if held <= 365:
                    # Would trigger STCG at 20%
                    tax_impact = trade_value * 0.20
                    if days_to_ltcg <= 30:
                        reason += f" (Warning: STCG — only {days_to_ltcg}d to LTCG eligibility)"
                        priority = "OPTIONAL"
                else:
                    # LTCG with possible exemption
                    taxable = max(0, trade_value - 125_000)
                    tax_impact = taxable * 0.125

            raw_trades.append(RebalanceTrade(
                symbol=sym,
                side=side,
                quantity=quantity,
                price=price,
                value=trade_value,
                weight_change=abs(report.absolute_drift),
                priority=priority,
                reason=reason,
                tax_impact=tax_impact,
                days_to_ltcg=days_to_ltcg,
            ))

        # Apply turnover constraint
        final_trades, total_turnover = self._apply_turnover_limit(raw_trades, total_value)

        # Summary
        sell_trades = [t for t in final_trades if t.side == "SELL"]
        buy_trades = [t for t in final_trades if t.side == "BUY"]
        total_tax = sum(t.tax_impact for t in sell_trades)
        sell_value = sum(t.value for t in sell_trades)
        buy_value = sum(t.value for t in buy_trades)

        summary = (
            f"Rebalancing: {len(final_trades)} trades "
            f"({len(sell_trades)} sells, {len(buy_trades)} buys). "
            f"Turnover: {total_turnover:.1%}. "
            f"Estimated tax: ₹{total_tax:,.0f}."
        )

        return {
            "trades": final_trades,
            "turnover": round(total_turnover, 6),
            "estimated_tax_impact": round(total_tax, 2),
            "sell_value": round(sell_value, 2),
            "buy_value": round(buy_value, 2),
            "drift_report": drift,
            "summary": summary,
        }

    def _apply_turnover_limit(
        self,
        trades: list[RebalanceTrade],
        total_value: float,
    ) -> tuple[list[RebalanceTrade], float]:
        """Apply maximum turnover constraint to the trade list.

        Prioritises REQUIRED trades over RECOMMENDED over OPTIONAL.
        """
        if not trades:
            return trades, 0.0

        # Sort by priority (REQUIRED first) then by absolute value
        priority_order = {"REQUIRED": 0, "RECOMMENDED": 1, "OPTIONAL": 2}
        sorted_trades = sorted(
            trades,
            key=lambda t: (priority_order.get(t.priority, 99), -t.value),
        )

        remaining_turnover = self.max_turnover * total_value
        accepted: list[RebalanceTrade] = []
        used_turnover = 0.0

        for trade in sorted_trades:
            if used_turnover + trade.value <= remaining_turnover:
                accepted.append(trade)
                used_turnover += trade.value
            else:
                # Partial fill
                remaining = remaining_turnover - used_turnover
                if remaining > 0:
                    partial_ratio = remaining / trade.value
                    partial_trade = RebalanceTrade(
                        symbol=trade.symbol,
                        side=trade.side,
                        quantity=trade.quantity * partial_ratio,
                        price=trade.price,
                        value=remaining,
                        weight_change=trade.weight_change * partial_ratio,
                        priority=trade.priority,
                        reason=f"{trade.reason} (partial — turnover limit)",
                        tax_impact=trade.tax_impact * partial_ratio,
                        days_to_ltcg=trade.days_to_ltcg,
                    )
                    accepted.append(partial_trade)
                    used_turnover += remaining
                break

        return accepted, used_turnover / total_value if total_value > 0 else 0.0

    # ------------------------------------------------------------------
    # Minimise turnover
    # ------------------------------------------------------------------

    def minimize_turnover(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        max_turnover: float = 0.20,
        prices: Optional[dict[str, float]] = None,
        holdings_days: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        """Find the minimum set of trades to get closest to target weights.

        Uses a greedy approach: only trade assets where the drift-adjusted
        tax cost is lowest.

        Parameters
        ----------
        current_weights : dict[str, float]
            Current portfolio weights.
        target_weights : dict[str, float]
            Target portfolio weights.
        max_turnover : float
            Maximum turnover to use.
        prices, holdings_days :
            Same as :meth:`generate_rebalancing_trades`.

        Returns
        -------
        dict
            Same format as :meth:`generate_rebalancing_trades`.
        """
        drift = self.calculate_drift(current_weights, target_weights)

        # Score each potential trade by: benefit per unit of cost (tax + turnover)
        scored_trades: list[tuple[float, str, float]] = []  # (score, side, symbol)
        for sym, report in drift.items():
            if report.action == "HOLD":
                continue
            abs_drift = abs(report.absolute_drift)
            tax_cost = 0.0

            if report.action == "SELL" and holdings_days:
                held = holdings_days.get(sym, 0)
                if held <= 365:
                    tax_cost = 0.20  # STCG rate
                else:
                    tax_cost = 0.125  # LTCG rate

            # Score: benefit = drift reduction, cost = tax + proportional turnover
            cost = tax_cost + 0.01  # Small base cost for any trade
            score = abs_drift / cost if cost > 0 else abs_drift * 100
            side = "BUY" if report.absolute_drift > 0 else "SELL"
            scored_trades.append((score, side, sym))

        # Sort by score (highest benefit/cost first)
        scored_trades.sort(reverse=True)

        # Greedily select trades until turnover budget is exhausted
        selected_trades: list[RebalanceTrade] = []
        used_turnover = 0.0
        current_total = sum(current_weights.values())
        if current_total <= 0:
            return {
                "trades": [],
                "turnover": 0.0,
                "estimated_tax_impact": 0.0,
                "drift_report": drift,
                "summary": "No trades — current portfolio value is zero.",
            }

        total_value = 1.0  # Working with weights
        turnover_budget = max_turnover * total_value

        for score, side, sym in scored_trades:
            report = drift[sym]
            trade_weight = abs(report.absolute_drift)

            if used_turnover + trade_weight > turnover_budget:
                # Partial fill
                remaining = turnover_budget - used_turnover
                if remaining <= 0:
                    break
                trade_weight = remaining

            price = prices.get(sym, 1.0) if prices else 1.0
            selected_trades.append(RebalanceTrade(
                symbol=sym,
                side=side,
                quantity=trade_weight / price if price > 0 else 0,
                price=price,
                value=trade_weight,
                weight_change=trade_weight,
                priority="RECOMMENDED",
                reason=f"Turnover-minimised: score={score:.2f}",
                tax_impact=trade_weight * (0.20 if (holdings_days and holdings_days.get(sym, 0) <= 365) else 0.125) if side == "SELL" else 0.0,
                days_to_ltcg=max(0, 365 - (holdings_days.get(sym, 0) if holdings_days else 0)) if side == "SELL" else -1,
            ))
            used_turnover += trade_weight

        return {
            "trades": selected_trades,
            "turnover": round(used_turnover, 6),
            "estimated_tax_impact": round(sum(t.tax_impact for t in selected_trades), 2),
            "drift_report": drift,
            "summary": (
                f"Turnover-minimised rebalance: {len(selected_trades)} trades, "
                f"turnover {used_turnover:.1%} (limit {max_turnover:.1%})."
            ),
        }

    # ------------------------------------------------------------------
    # Tax-aware rebalancing helper
    # ------------------------------------------------------------------

    def tax_aware_rebalance_recommendation(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        total_value: float,
        prices: dict[str, float],
        holdings_days: dict[str, int],
    ) -> dict[str, Any]:
        """Generate tax-aware rebalancing recommendations.

        Considers:
        - Positions near LTCG threshold (avoid selling 1-30 days before)
        - Tax-loss harvesting opportunities
        - Net tax impact of the rebalance

        Returns
        -------
        dict
            ``{
                "immediate_trades": [RebalanceTrade, ...],
                "deferred_trades": [RebalanceTrade, ...],
                "tax_loss_opportunities": [RebalanceTrade, ...],
                "estimated_net_tax": float,
                "recommendation": str,
            }``
        """
        full_result = self.generate_rebalancing_trades(
            current_weights=current_weights,
            target_weights=target_weights,
            total_value=total_value,
            prices=prices,
            holdings_days=holdings_days,
        )

        all_trades = full_result["trades"]
        immediate: list[RebalanceTrade] = []
        deferred: list[RebalanceTrade] = []
        tax_loss_ops: list[RebalanceTrade] = []

        for trade in all_trades:
            if trade.side == "BUY":
                immediate.append(trade)
            elif trade.days_to_ltcg >= 0 and trade.days_to_ltcg <= 30:
                # Defer sells that are close to LTCG eligibility
                deferred.append(trade)
            elif trade.tax_impact < 0:
                # Tax-loss harvesting (selling at a loss)
                tax_loss_ops.append(trade)
                immediate.append(trade)
            else:
                immediate.append(trade)

        total_tax = sum(t.tax_impact for t in immediate)
        total_value_deferred = sum(t.value for t in deferred)

        # Build recommendation
        recommendation_parts: list[str] = []
        if immediate:
            recommendation_parts.append(
                f"Execute {len(immediate)} trades now (est. tax: ₹{total_tax:,.0f})."
            )
        if deferred:
            recommendation_parts.append(
                f"Defer {len(deferred)} sells worth ₹{total_value_deferred:,.0f} "
                f"until after LTCG eligibility."
            )
        if tax_loss_ops:
            recommendation_parts.append(
                f"Harvest {len(tax_loss_ops)} tax losses (saving ~₹{abs(sum(t.tax_impact for t in tax_loss_ops)):,.0f})."
            )
        if not recommendation_parts:
            recommendation_parts.append("No rebalancing needed.")

        return {
            "immediate_trades": immediate,
            "deferred_trades": deferred,
            "tax_loss_opportunities": tax_loss_ops,
            "estimated_net_tax": round(total_tax, 2),
            "recommendation": " ".join(recommendation_parts),
            "full_result": full_result,
        }
