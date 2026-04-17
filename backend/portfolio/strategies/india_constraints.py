"""
India-specific portfolio constraints and tax rules.

Encodes SEBI regulations, Income Tax Act provisions for capital gains
(LTCG / STCG), dividend taxation, sector concentration limits, and
tax-loss harvesting rules applicable to Indian resident investors.

All tax figures are based on rules applicable from FY 2024-25
(post Budget 2024 amendments effective 23 July 2024).

Typical usage::

    constraints = IndiaPortfolioConstraints()
    violations = constraints.validate_constraints(weights, holdings)
    tax = constraints.calculate_tax_impact(trades)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tax Rules (FY 2024-25, Assessment Year 2025-26)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaxSlab:
    """A single income tax slab."""

    lower: int
    upper: int  # Use -1 for no upper limit
    rate: float


# New Tax Regime slabs (FY 2024-25)
NEW_REGIME_SLABS: list[TaxSlab] = [
    TaxSlab(0, 300_000, 0.00),
    TaxSlab(300_001, 700_000, 0.05),
    TaxSlab(700_001, 1_000_000, 0.10),
    TaxSlab(1_000_001, 1_200_000, 0.15),
    TaxSlab(1_200_001, 1_500_000, 0.20),
    TaxSlab(1_500_001, -1, 0.30),
]

# Old Tax Regime slabs (FY 2024-25)
OLD_REGIME_SLABS: list[TaxSlab] = [
    TaxSlab(0, 250_000, 0.00),
    TaxSlab(250_001, 500_000, 0.05),
    TaxSlab(500_001, 1_000_000, 0.20),
    TaxSlab(1_000_001, -1, 0.30),
]

# Additional surcharge on income > 50L
SURCHARGE_RATES: list[tuple[int, float]] = [
    (50_00_000, 0.10),
    (1_00_00_000, 0.15),
    (2_00_00_000, 0.25),
    (5_00_00_000, 0.37),  # Super-rich
]

HEALTH_AND_EDUCATION_CESS = 0.04

# Capital gains tax rates
LTCG_EQUITY_RATE = 0.125  # 12.5% from 23 Jul 2024
LTCG_EQUITY_EXEMPTION = 125_000  # Rs 1.25L per year
LTCG_EQUITY_HOLDING_PERIOD_DAYS = 365  # > 12 months

STCG_EQUITY_RATE = 0.20  # 20% from 23 Jul 2024
STCG_EQUITY_HOLDING_PERIOD_DAYS = 365  # <= 12 months

STCG_DEBT_RATE = None  # Taxed at slab rate (no special rate)
STCG_DEBT_HOLDING_PERIOD_DAYS = 1095  # 3 years for debt LTCG threshold

LTCG_OTHER_RATE = 0.125  # 12.5% (no indexation for post-Apr 2023)
LTCG_OTHER_HOLDING_PERIOD_DAYS = 1095  # 3 years

# Dividend taxation
DIVIDEND_TDS_RATE = 0.10  # 10% TDS on dividends > Rs 5,000
DIVIDEND_TAX_THRESHOLD = 5_000  # TDS threshold per payout

# Tax loss harvesting
LOSS_CARRY_FORWARD_YEARS = 8  # Capital losses carry forward max 8 years

# Advance tax installments (by % of total liability)
ADVANCE_TAX_SCHEDULE: list[tuple[str, float]] = [
    ("Jun 15", 0.15),
    ("Sep 15", 0.45),
    ("Dec 15", 0.75),
    ("Mar 15", 1.00),
]


# ---------------------------------------------------------------------------
# SEBI Regulations
# ---------------------------------------------------------------------------

SEBI_MAX_SINGLE_STOCK_WEIGHT = 0.10  # 10% for open-ended mutual funds
SEBI_MAX_TOP_10_HOLDINGS = 0.55  # 55% combined for top 10 stocks
SEBI_MAX_SECTOR_WEIGHT = 0.25  # 25% per sector (varies by AMC)


# ---------------------------------------------------------------------------
# Sector classification for Indian stocks
# ---------------------------------------------------------------------------

SECTOR_LIMITS: dict[str, float] = {
    "BANKING": 0.25,
    "IT": 0.25,
    "OIL_GAS": 0.20,
    "FMCG": 0.20,
    "AUTOMOBILE": 0.20,
    "PHARMA": 0.20,
    "FINANCIAL_SERVICES": 0.25,
    "INFRASTRUCTURE": 0.20,
    "METALS": 0.20,
    "TELECOM": 0.15,
    "CEMENT": 0.15,
    "POWER": 0.20,
    "CONSUMER_GOODS": 0.20,
    "INSURANCE": 0.15,
    "CHEMICALS": 0.15,
    "MINING": 0.15,
    "CONGLOMERATE": 0.20,
}

DEFAULT_SECTOR_LIMIT = 0.25  # Default max weight for unknown sectors


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Holding:
    """Represents a single holding in a portfolio."""

    symbol: str
    quantity: float
    avg_buy_price: float  # Per-unit average buy price (INR)
    current_price: float
    buy_date: date
    sector: str = "UNKNOWN"
    asset_type: str = "equity"  # "equity" or "debt"

    @property
    def current_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def invested_value(self) -> float:
        return self.quantity * self.avg_buy_price

    @property
    def pnl(self) -> float:
        return self.current_value - self.invested_value

    @property
    def pnl_pct(self) -> float:
        if self.invested_value == 0:
            return 0.0
        return (self.pnl / self.invested_value) * 100

    @property
    def holding_days(self) -> int:
        return (date.today() - self.buy_date).days

    @property
    def is_ltcg(self) -> bool:
        if self.asset_type == "equity":
            return self.holding_days > LTCG_EQUITY_HOLDING_PERIOD_DAYS
        return self.holding_days > STCG_DEBT_HOLDING_PERIOD_DAYS


@dataclass
class Trade:
    """Represents a single trade (buy or sell)."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float  # Per-unit price (INR)
    date: date
    sector: str = "UNKNOWN"
    asset_type: str = "equity"
    holding_days_at_sale: int = 0  # Relevant for sells
    buy_price_for_lots: Optional[float] = None  # FIFO buy price for capital gains


@dataclass
class CapitalGain:
    """A computed capital gain from a trade."""

    symbol: str
    gain_type: str  # "LTCG" or "STCG"
    asset_type: str  # "equity" or "debt"
    gain_amount: float  # Positive = profit, negative = loss
    holding_days: int
    tax_rate: float  # Applicable rate
    estimated_tax: float  # Pre-cess tax


# ---------------------------------------------------------------------------
# Main constraints class
# ---------------------------------------------------------------------------

class IndiaPortfolioConstraints:
    """India-specific portfolio constraints and tax rules.

    Provides:
    - SEBI regulatory constraint validation
    - Sector concentration limits
    - Capital gains tax computation (LTCG / STCG)
    - Dividend taxation
    - Tax loss harvesting analysis
    - Advance tax schedule computation
    """

    def __init__(
        self,
        tax_regime: str = "new",
        assessable_income: float = 0.0,
    ) -> None:
        """Initialise with investor-specific parameters.

        Parameters
        ----------
        tax_regime : str
            ``"new"`` or ``"old"`` income tax regime.
        assessable_income : float
            Annual income (excluding capital gains) for surcharge calculation.
        """
        self.tax_regime = tax_regime
        self.assessable_income = assessable_income
        self._slabs = NEW_REGIME_SLABS if tax_regime == "new" else OLD_REGIME_SLABS

    # ------------------------------------------------------------------
    # Constraint validation
    # ------------------------------------------------------------------

    def validate_constraints(
        self,
        weights: dict[str, float],
        holdings: Optional[list[Holding]] = None,
        sector_map: Optional[dict[str, str]] = None,
        max_single_stock: Optional[float] = None,
        max_sector: Optional[dict[str, float]] = None,
    ) -> list[str]:
        """Validate portfolio weights against all India-specific constraints.

        Parameters
        ----------
        weights : dict[str, float]
            Symbol -> weight mapping (should sum to ~1.0).
        holdings : list[Holding] or None
            Current holdings for additional validation.
        sector_map : dict[str, str] or None
            Symbol -> sector mapping.
        max_single_stock : float or None
            Max weight per stock.  Defaults to SEBI limit (10%).
        max_sector : dict[str, float] or None
            Sector -> max weight overrides.

        Returns
        -------
        list[str]
            List of violation descriptions.  Empty list means all constraints pass.
        """
        violations: list[str] = []

        # Weight normalisation
        total = sum(weights.values())
        if total <= 0:
            return ["Total weight is zero or negative."]

        normalised = {s: w / total for s, w in weights.items()}

        # 1. Single-stock concentration
        max_stock = max_single_stock or SEBI_MAX_SINGLE_STOCK_WEIGHT
        for sym, w in normalised.items():
            if w > max_stock:
                violations.append(
                    f"[SEBI] {sym} weight {w:.2%} exceeds {max_stock:.0%} single-stock limit."
                )

        # 2. Top-10 holdings concentration
        sorted_weights = sorted(normalised.values(), reverse=True)
        top_10_sum = sum(sorted_weights[:10])
        if top_10_sum > SEBI_MAX_TOP_10_HOLDINGS:
            violations.append(
                f"[SEBI] Top 10 holdings at {top_10_sum:.2%} exceed "
                f"{SEBI_MAX_TOP_10_HOLDINGS:.0%} combined limit."
            )

        # 3. Sector concentration
        if sector_map:
            effective_limits = {**SECTOR_LIMITS}
            if max_sector:
                effective_limits.update(max_sector)

            sector_weights: dict[str, float] = {}
            for sym, w in normalised.items():
                sec = sector_map.get(sym, "UNKNOWN")
                sector_weights[sec] = sector_weights.get(sec, 0.0) + w

            for sec, sw in sector_weights.items():
                limit = effective_limits.get(sec, DEFAULT_SECTOR_LIMIT)
                if sw > limit:
                    violations.append(
                        f"[SECTOR] {sec} at {sw:.2%} exceeds {limit:.0%} sector limit."
                    )

        # 4. Minimum diversification (at least 10 stocks)
        non_zero = sum(1 for w in normalised.values() if w > 0.005)
        if non_zero < 10:
            violations.append(
                f"[DIVERSIFICATION] Only {non_zero} non-trivial holdings. "
                f"Minimum 10 recommended."
            )

        # 5. Holdings-level checks
        if holdings:
            self._validate_holdings(holdings, violations)

        return violations

    def _validate_holdings(
        self, holdings: list[Holding], violations: list[str]
    ) -> None:
        """Additional checks on actual holdings."""
        total_value = sum(h.current_value for h in holdings)
        if total_value <= 0:
            return

        # Check for concentrated positions
        for h in holdings:
            weight = h.current_value / total_value
            if weight > SEBI_MAX_SINGLE_STOCK_WEIGHT * 1.5:
                violations.append(
                    f"[RISK] {h.symbol} is {weight:.2%} of portfolio — "
                    f"significantly above {SEBI_MAX_SINGLE_STOCK_WEIGHT:.0%} limit. "
                    f"Consider rebalancing."
                )

        # Check for losses that could be harvested
        tax_loss_opportunities = [
            h for h in holdings if h.pnl < 0 and h.holding_days > 1
        ]
        if tax_loss_opportunities:
            total_booked_loss = sum(h.pnl for h in tax_loss_opportunities)
            violations.append(
                f"[TAX_LOSS_HARVEST] {len(tax_loss_opportunities)} holdings with "
                f"unrealised losses totaling ₹{abs(total_booked_loss):,.0f}. "
                f"Consider tax-loss harvesting before 31 March."
            )

    # ------------------------------------------------------------------
    # Tax impact calculation
    # ------------------------------------------------------------------

    def calculate_tax_impact(
        self,
        trades: list[Trade],
        assessable_income: Optional[float] = None,
    ) -> dict[str, Any]:
        """Calculate tax impact of a list of trades.

        Parameters
        ----------
        trades : list[Trade]
            Buy/sell trades to evaluate.
        assessable_income : float or None
            Override instance assessable_income.

        Returns
        -------
        dict
            ``{
                "capital_gains": [CapitalGain, ...],
                "ltcg_total": float,
                "stcg_total": float,
                "ltcg_tax": float,
                "stcg_tax": float,
                "total_tax": float,
                "cess": float,
                "surcharge": float,
                "breakdown": {type: amount, ...},
            }``
        """
        if assessable_income is not None:
            self.assessable_income = assessable_income

        capital_gains: list[CapitalGain] = []
        ltcg_total = 0.0
        stcg_total = 0.0

        for trade in trades:
            if trade.side != "SELL":
                continue

            # Compute gain/loss
            if trade.buy_price_for_lots is not None and trade.buy_price_for_lots > 0:
                cost = trade.buy_price_for_lots * trade.quantity
            else:
                cost = 0.0

            proceeds = trade.price * trade.quantity
            gain = proceeds - cost

            # Determine LTCG vs STCG
            if trade.asset_type == "equity":
                is_ltcg = trade.holding_days_at_sale > LTCG_EQUITY_HOLDING_PERIOD_DAYS
                if is_ltcg:
                    gain_type = "LTCG"
                    rate = LTCG_EQUITY_RATE
                    # Apply exemption for equity LTCG
                    taxable_gain = max(0, gain - LTCG_EQUITY_EXEMPTION)
                    est_tax = taxable_gain * rate
                    ltcg_total += gain
                else:
                    gain_type = "STCG"
                    rate = STCG_EQUITY_RATE
                    est_tax = gain * rate
                    stcg_total += gain
            else:
                # Debt / other assets
                is_ltcg = trade.holding_days_at_sale > STCG_DEBT_HOLDING_PERIOD_DAYS
                if is_ltcg:
                    gain_type = "LTCG"
                    rate = LTCG_OTHER_RATE
                    est_tax = max(0, gain) * rate
                    ltcg_total += gain
                else:
                    gain_type = "STCG"
                    # Taxed at slab rate (approximate using marginal rate)
                    rate = self._get_marginal_rate(self.assessable_income + stcg_total)
                    est_tax = max(0, gain) * rate
                    stcg_total += gain

            capital_gains.append(CapitalGain(
                symbol=trade.symbol,
                gain_type=gain_type,
                asset_type=trade.asset_type,
                gain_amount=gain,
                holding_days=trade.holding_days_at_sale,
                tax_rate=rate,
                estimated_tax=est_tax,
            ))

        # Tax computation
        ltcg_tax = max(0, ltcg_total - LTCG_EQUITY_EXEMPTION) * LTCG_EQUITY_RATE
        stcg_tax = stcg_total * STCG_EQUITY_RATE

        # Surcharge
        total_taxable = self.assessable_income + max(0, stcg_total) + max(0, ltcg_total - LTCG_EQUITY_EXEMPTION)
        surcharge_rate = self._get_surcharge_rate(total_taxable)
        surcharge = (ltcg_tax + stcg_tax) * surcharge_rate

        # Cess
        cess = (ltcg_tax + stcg_tax + surcharge) * HEALTH_AND_EDUCATION_CESS

        total_tax = ltcg_tax + stcg_tax + surcharge + cess

        return {
            "capital_gains": capital_gains,
            "ltcg_total": round(ltcg_total, 2),
            "stcg_total": round(stcg_total, 2),
            "ltcg_tax": round(ltcg_tax, 2),
            "stcg_tax": round(stcg_tax, 2),
            "surcharge": round(surcharge, 2),
            "cess": round(cess, 2),
            "total_tax": round(total_tax, 2),
            "breakdown": {
                "ltcg_taxable": round(max(0, ltcg_total - LTCG_EQUITY_EXEMPTION), 2),
                "ltcg_exemption_applied": round(min(ltcg_total, LTCG_EQUITY_EXEMPTION), 2),
                "surcharge_rate": surcharge_rate,
                "cess_rate": HEALTH_AND_EDUCATION_CESS,
            },
        }

    # ------------------------------------------------------------------
    # Tax loss harvesting
    # ------------------------------------------------------------------

    def analyze_tax_loss_harvest(
        self,
        holdings: list[Holding],
        target_symbol: Optional[str] = None,
    ) -> dict[str, Any]:
        """Identify tax-loss harvesting opportunities.

        Parameters
        ----------
        holdings : list[Holding]
            Current portfolio holdings.
        target_symbol : str or None
            If set, only analyze this specific symbol.

        Returns
        -------
        dict
            ``{
                "harvestable": [Holding, ...],
                "total_loss": float,
                "potential_tax_saving": float,
                "wash_sale_warning": list[str],
                "recommendation": str,
            }``
        """
        candidates = [
            h for h in holdings
            if h.pnl < 0
            and (target_symbol is None or h.symbol == target_symbol)
        ]

        if not candidates:
            return {
                "harvestable": [],
                "total_loss": 0.0,
                "potential_tax_saving": 0.0,
                "wash_sale_warning": [],
                "recommendation": "No tax-loss harvesting opportunities found.",
            }

        total_loss = sum(h.pnl for h in candidates)

        # Estimate tax saving (loss offset against gains at applicable rate)
        # Conservative estimate: assume STCG rate for all
        potential_saving = abs(total_loss) * STCG_EQUITY_RATE

        # Wash sale warning (India doesn't have strict wash sale rules like US,
        # but buying back within 30 days may be scrutinised)
        wash_warnings: list[str] = []
        for h in candidates:
            if h.holding_days > 300:  # Close to 1-year LTCG threshold
                wash_warnings.append(
                    f"⚠️ {h.symbol}: Holding for {h.holding_days} days. "
                    f"Selling now locks in STCG (20%) instead of waiting "
                    f"{LTCG_EQUITY_HOLDING_PERIOD_DAYS - h.holding_days}d more days for LTCG (12.5%)."
                )

        # Recommendation
        if total_loss < -10_000:
            recommendation = (
                f"Found {len(candidates)} loss-making positions with total unrealised "
                f"loss of ₹{abs(total_loss):,.0f}. Estimated tax saving: ₹{potential_saving:,.0f}. "
                f"Consider harvesting before 31 March to offset capital gains this FY."
            )
        else:
            recommendation = (
                f"Found {len(candidates)} positions with small losses (₹{abs(total_loss):,.0f}). "
                f"Tax saving of ₹{potential_saving:,.0f} may not justify transaction costs."
            )

        return {
            "harvestable": candidates,
            "total_loss": round(total_loss, 2),
            "potential_tax_saving": round(potential_saving, 2),
            "wash_sale_warning": wash_warnings,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Advance tax schedule
    # ------------------------------------------------------------------

    def compute_advance_tax_schedule(
        self,
        total_capital_gains: float,
        other_income: float,
    ) -> list[dict[str, Any]]:
        """Compute advance tax installment schedule.

        Parameters
        ----------
        total_capital_gains : float
            Estimated total capital gains for the FY.
        other_income : float
            Other taxable income (salary, business, etc.).

        Returns
        -------
        list[dict]
            Each dict: ``{due_date, cumulative_pct, amount_due, notes}``.
        """
        total_income = other_income + max(0, total_capital_gains)
        estimated_tax = self._compute_income_tax(total_income)

        schedule = []
        for due_date, cum_pct in ADVANCE_TAX_SCHEDULE:
            amount = estimated_tax * cum_pct
            prev_amount = schedule[-1]["amount_due"] if schedule else 0
            installment = amount - prev_amount

            notes = ""
            if due_date == "Jun 15" and total_capital_gains > 0:
                notes = "Include estimated capital gains tax in first installment"

            schedule.append({
                "due_date": due_date,
                "cumulative_pct": cum_pct,
                "cumulative_amount": round(amount, 2),
                "installment_due": round(installment, 2),
                "notes": notes,
            })

        return schedule

    # ------------------------------------------------------------------
    # Dividend taxation
    # ------------------------------------------------------------------

    def calculate_dividend_tax(
        self,
        dividends: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate tax on dividend income.

        Parameters
        ----------
        dividends : list[dict]
            Each dict: ``{symbol, amount, date, asset_type}``.

        Returns
        -------
        dict
            ``{
                "total_dividends": float,
                "taxable_amount": float,
                "estimated_tax": float,
                "tds_applicable": bool,
                "tds_amount": float,
                "breakdown": [dict, ...],
            }``
        """
        total = 0.0
        breakdown = []
        tds_total = 0.0

        for div in dividends:
            amount = float(div.get("amount", 0))
            total += amount

            tds = min(amount * DIVIDEND_TDS_RATE, amount) if amount > DIVIDEND_TAX_THRESHOLD else 0.0
            tds_total += tds

            breakdown.append({
                "symbol": div.get("symbol", "UNKNOWN"),
                "amount": round(amount, 2),
                "tds": round(tds, 2),
                "tds_applicable": amount > DIVIDEND_TAX_THRESHOLD,
                "date": div.get("date", ""),
            })

        # Dividends are taxed at slab rate
        marginal_rate = self._get_marginal_rate(self.assessable_income + total)
        estimated_tax = total * marginal_rate
        tds_credits_against = tds_total

        return {
            "total_dividends": round(total, 2),
            "taxable_amount": round(total, 2),
            "estimated_tax": round(estimated_tax, 2),
            "tds_applicable": total > DIVIDEND_TAX_THRESHOLD,
            "tds_amount": round(tds_total, 2),
            "net_tax_payable": round(max(0, estimated_tax - tds_credits_against), 2),
            "tax_regime": self.tax_regime,
            "breakdown": breakdown,
        }

    # ------------------------------------------------------------------
    # Tax computation helpers (private)
    # ------------------------------------------------------------------

    def _compute_income_tax(self, income: float) -> float:
        """Compute income tax liability for a given total income."""
        tax = 0.0
        remaining = max(0, income)

        for slab in self._slabs:
            if remaining <= 0:
                break
            upper = slab.upper if slab.upper > 0 else remaining + slab.lower
            taxable_in_slab = min(remaining, upper - slab.lower + 1)
            tax += taxable_in_slab * slab.rate
            remaining -= taxable_in_slab

        # Section 87A rebate (new regime only)
        if self.tax_regime == "new" and income <= 700_000:
            tax = 0.0

        return tax

    def _get_marginal_rate(self, income: float) -> float:
        """Get the marginal tax rate for a given income level."""
        if self.tax_regime == "new" and income <= 700_000:
            return 0.0

        for slab in reversed(self._slabs):
            if income >= slab.lower:
                return slab.rate
        return 0.0

    def _get_surcharge_rate(self, income: float) -> float:
        """Get the surcharge rate applicable for given income."""
        for threshold, rate in reversed(SURCHARGE_RATES):
            if income > threshold:
                return rate
        return 0.0
