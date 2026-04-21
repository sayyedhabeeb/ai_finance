# ============================================================
# AI Financial Brain — Shared Pydantic v2 Schemas
# ============================================================
"""
Central schema definitions for the AI Financial Brain system.

All inter-module communication, API payloads, agent messages, financial
domain objects, and persistence models are defined here using Pydantic v2.
Every model includes full type annotations, validators where appropriate,
and descriptive docstrings for auto-generated documentation.
"""

from __future__ import annotations

from datetime import date as date_type, datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# Enumerations
# ============================================================


class AgentType(str, Enum):
    """Types of autonomous agents in the multi-agent system.

    Each agent type corresponds to a specialised role:
    - **PERSONAL_CFO**: Orchestrates the conversation and provides personalised financial advice.
    - **MARKET_ANALYST**: Analyses market data, price trends, and technical indicators.
    - **NEWS_SENTIMENT**: Processes news articles and social media for sentiment signals.
    - **RISK_ANALYST**: Computes risk metrics (VaR, drawdown, volatility) and scenario analyses.
    - **PORTFOLIO_MANAGER**: Handles portfolio construction, optimisation, and rebalancing.
    - **CRITIC**: Evaluates agent outputs for quality before presenting to the user.
    """

    PERSONAL_CFO = "personal_cfo"
    MARKET_ANALYST = "market_analyst"
    NEWS_SENTIMENT = "news_sentiment"
    RISK_ANALYST = "risk_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    CRITIC = "critic"


class RiskLevel(str, Enum):
    """Risk severity levels used across portfolio and threat assessments.

    Maps to quantitative thresholds in the risk engine:
    - **LOW**: Conservative allocation (80 %+ fixed income).
    - **MEDIUM**: Balanced allocation (50/50 equity-debt).
    - **HIGH**: Growth-oriented allocation (80 %+ equity).
    - **VERY_HIGH**: Aggressive allocation (levered / concentrated positions).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SentimentType(str, Enum):
    """Market sentiment classification produced by the news-sentiment agent."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class MarketRegime(str, Enum):
    """Current macro market regime classification.

    Used by the portfolio manager to adjust optimisation constraints
    and the risk analyst to stress-test portfolios.
    """

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class DataType(str, Enum):
    """Categories of financial data consumed by the system.

    Each data type maps to a specific ingestion pipeline and storage backend.
    """

    MARKET_PRICE = "market_price"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    SOCIAL = "social"
    MACRO = "macro"


class QueryType(str, Enum):
    """Classification of user queries for routing to appropriate agents.

    The orchestrator uses this enum to determine which agent(s) should
    be invoked for a given user query.
    """

    MARKET_QUERY = "market_query"
    PORTFOLIO_QUERY = "portfolio_query"
    RISK_QUERY = "risk_query"
    PERSONAL_FINANCE_QUERY = "personal_finance_query"
    GENERAL = "general"


class OptimizationStrategy(str, Enum):
    """Supported portfolio optimization methodologies.

    Each strategy has different assumptions and is suited to different
    market conditions and investor preferences.
    """

    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"


class AssetClass(str, Enum):
    """Standard asset class classifications used in portfolio allocation."""

    EQUITY = "equity"
    DEBT = "debt"
    COMMODITY = "commodity"
    REAL_ESTATE = "real_estate"
    CRYPTO = "crypto"
    CASH = "cash"


class TimeUnit(str, Enum):
    """Time granularities for financial data and reporting.

    Used by the market data service to specify bar intervals and
    by the portfolio manager to define the investment horizon.
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# ============================================================
# Agent Communication Models
# ============================================================


class AgentMessage(BaseModel):
    """Base message envelope for inter-agent communication.

    Every message exchanged between agents wraps in this envelope so that
    the message bus can route, log, and trace the full conversation graph.
    Supports a tree structure via ``parent_message_id`` for threaded
    agent-to-agent discussions.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    sender: AgentType = Field(description="The agent sending this message.")
    recipient: AgentType = Field(description="The intended recipient agent.")
    content: str = Field(min_length=1, description="Free-text body of the message.")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was created (UTC).",
    )
    message_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this message.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached by the sender.",
    )
    parent_message_id: Optional[UUID] = Field(
        default=None,
        description="References the message this is replying to, if any.",
    )


class AgentTask(BaseModel):
    """Task assignment dispatched to a specific agent.

    The orchestrator creates ``AgentTask`` instances and places them on the
    task queue.  Agents pick up tasks whose ``agent_type`` matches their
    role, execute them, and return an ``AgentResult``.

    Priority is 1 (highest) through 10 (lowest).  Tasks with earlier
    deadlines are scheduled first among tasks of equal priority.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    task_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this task.",
    )
    agent_type: AgentType = Field(
        description="The agent type that should handle this task.",
    )
    query: str = Field(
        min_length=1,
        description="The user query or directive to process.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context such as market data, portfolio state, etc.",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority 1 (highest) to 10 (lowest).",
    )
    deadline: Optional[datetime] = Field(
        default=None,
        description="Optional deadline after which the task is considered stale.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the task was created by the orchestrator.",
    )


class AgentResult(BaseModel):
    """Structured result returned by an agent after processing a task.

    Includes a confidence score so the critic agent can judge whether the
    result meets quality thresholds before it is shown to the user.

    The ``output`` field is a dictionary allowing agents to return rich,
    structured data (e.g. portfolio weights, risk metrics, sentiment scores)
    alongside any textual analysis.  The optional ``error`` field captures
    non-fatal warnings that occurred during processing.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    task_id: UUID = Field(
        description="References the originating ``AgentTask.task_id``.",
    )
    agent_type: AgentType = Field(
        description="The agent that produced this result.",
    )
    output: dict[str, Any] | str = Field(
        default_factory=dict,
        description="Structured output — may contain text, data, scores, recommendations, etc.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Agent's self-assessed confidence in the output (0–1).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured metadata such as data sources used, model version, etc.",
    )
    processing_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock seconds the agent spent processing.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Non-fatal error or warning message encountered during processing.",
    )

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v: float) -> float:
        """Validate confidence is in [0, 1] and round to 4 decimal places."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return round(v, 4)


class AgentError(BaseModel):
    """Error envelope returned when an agent fails to process a task.

    Captures the full traceback so that the orchestrator can decide whether
    to retry, escalate, or report to the user.  The ``retry_count`` field
    prevents infinite retry loops — the orchestrator should give up after
    a configurable maximum (default 3).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    task_id: UUID = Field(
        description="References the originating ``AgentTask.task_id``.",
    )
    agent_type: AgentType = Field(
        description="The agent that encountered the error.",
    )
    error_type: str = Field(
        description="Python exception class name, e.g. ``ValueError``.",
    )
    error_message: str = Field(
        min_length=1,
        description="Human-readable error description.",
    )
    stack_trace: Optional[str] = Field(
        default=None,
        description="Full Python traceback string for debugging.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="How many times this task has been retried already.",
    )


# ============================================================
# Financial Domain Models
# ============================================================


class StockInfo(BaseModel):
    """Static reference data for a single tradable instrument.

    Sourced from exchange master data and fundamental databases.  Only
    ``symbol`` is required; all other fields are optional to support
    partial lookups from different data providers.

    This model is read-heavy and cached aggressively.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(
        min_length=1,
        max_length=20,
        description="Ticker symbol, e.g. ``RELIANCE`` or ``AAPL``.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Full legal name of the listed entity.",
    )
    exchange: Optional[str] = Field(
        default=None,
        description="Primary listing exchange, e.g. ``NSE``, ``BSE``, ``NYSE``.",
    )
    sector: Optional[str] = Field(
        default=None,
        description="GICS sector classification.",
    )
    industry: Optional[str] = Field(
        default=None,
        description="GICS sub-industry classification.",
    )
    market_cap: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Market capitalisation in the portfolio's base currency.",
    )
    pe_ratio: Optional[float] = Field(
        default=None,
        description="Trailing price-to-earnings ratio.",
    )
    book_value: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Per-share book value.",
    )
    dividend_yield: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annualised dividend yield as a fraction (0.02 = 2 %).",
    )
    isin: Optional[str] = Field(
        default=None,
        min_length=12,
        max_length=12,
        description="ISO 17442 ISIN code.",
    )


class MarketDataPoint(BaseModel):
    """Single OHLCV candle / bar for a given symbol and timestamp.

    Used as the atomic unit inside ``MarketDataBatch`` and for streaming
    real-time price updates.  The model validator ensures that the
    high/low bounds are consistent with the open and close prices.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Ticker symbol.")
    timestamp: datetime = Field(
        description="Beginning of the bar interval (UTC).",
    )
    open: float = Field(gt=0.0, description="Opening price.")
    high: float = Field(gt=0.0, description="Highest price during the interval.")
    low: float = Field(gt=0.0, description="Lowest price during the interval.")
    close: float = Field(gt=0.0, description="Closing price (last trade of the interval).")
    volume: int = Field(ge=0, description="Total shares / contracts traded.")
    vwap: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Volume-weighted average price.",
    )

    @model_validator(mode="after")
    def high_low_consistency(self) -> MarketDataPoint:
        """Ensure high >= max(open, close) and low <= min(open, close)."""
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"high ({self.high}) must be >= max(open, close) "
                f"({max(self.open, self.close)})"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"low ({self.low}) must be <= min(open, close) "
                f"({min(self.open, self.close)})"
            )
        return self


class MarketDataBatch(BaseModel):
    """A batch of OHLCV bars for a single symbol over a contiguous time window.

    Typically returned by the market data service for charting, back-testing,
    or feeding into ML models.  Data points must be in chronological order
    (oldest first).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    symbol: str = Field(min_length=1, description="Ticker symbol.")
    timeframe: TimeUnit = Field(
        default=TimeUnit.DAILY,
        description="Time granularity of the data points.",
    )
    interval: str = Field(
        default="1d",
        description="Bar interval — ``1m``, ``5m``, ``1h``, ``1d``, ``1wk``, ``1mo``.",
    )
    data_points: list[MarketDataPoint] = Field(
        default_factory=list,
        description="Ordered (oldest → newest) list of candles.",
    )

    @field_validator("data_points")
    @classmethod
    def data_points_must_be_chronological(
        cls,
        v: list[MarketDataPoint],
    ) -> list[MarketDataPoint]:
        """Ensure data points are ordered by ascending timestamp."""
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].timestamp < v[i - 1].timestamp:
                    raise ValueError(
                        f"data_points[{i}] timestamp {v[i].timestamp} is earlier "
                        f"than data_points[{i - 1}] timestamp {v[i - 1].timestamp}"
                    )
        return v


class FinancialMetrics(BaseModel):
    """Key fundamental financial metrics for a company in a given reporting period.

    Numbers are in the reporting currency unless noted otherwise.
    The ``period`` field is required and must identify the reporting period
    (e.g. ``Q3 FY2024`` or ``FY2024``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Ticker symbol.")
    revenue: Optional[float] = Field(
        default=None,
        description="Total revenue for the period.",
    )
    net_income: Optional[float] = Field(
        default=None,
        description="Net profit after tax.",
    )
    ebitda: Optional[float] = Field(
        default=None,
        description="Earnings before interest, taxes, depreciation & amortisation.",
    )
    roe: Optional[float] = Field(
        default=None,
        description="Return on equity (0.15 = 15 %).",
    )
    roa: Optional[float] = Field(
        default=None,
        description="Return on assets (0.10 = 10 %).",
    )
    debt_to_equity: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Debt-to-equity ratio.",
    )
    current_ratio: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Current assets / current liabilities.",
    )
    free_cash_flow: Optional[float] = Field(
        default=None,
        description="Free cash flow for the period.",
    )
    period: str = Field(
        min_length=1,
        description="Reporting period label, e.g. ``Q3 FY2024`` or ``FY2024``.",
    )


class PortfolioHolding(BaseModel):
    """A single position inside a user's portfolio.

    All monetary values are in the portfolio's base currency (default INR).

    A model validator automatically re-derives ``current_value``,
    ``unrealized_pnl``, and ``pnl_pct`` from the primitive fields
    ``quantity``, ``avg_cost``, and ``current_price`` to ensure internal
    consistency.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    symbol: str = Field(min_length=1, description="Ticker symbol of the holding.")
    quantity: float = Field(gt=0.0, description="Number of shares / units held.")
    avg_cost: float = Field(
        gt=0.0,
        description="Volume-weighted average purchase price per unit.",
    )
    current_price: float = Field(
        gt=0.0,
        description="Latest available market price per unit.",
    )
    current_value: float = Field(
        ge=0.0,
        description="Market value of the position (quantity × current_price).",
    )
    unrealized_pnl: float = Field(
        description="Unrealised profit / loss in base currency.",
    )
    pnl_pct: float = Field(
        description="Unrealised P&L as a percentage (0.10 = +10 %).",
    )
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of total portfolio value (0.25 = 25 %).",
    )

    @model_validator(mode="after")
    def derived_fields_consistency(self) -> PortfolioHolding:
        """Re-derive current_value, unrealized_pnl, and pnl_pct from primitives."""
        expected_value = round(self.quantity * self.current_price, 2)
        if abs(self.current_value - expected_value) > 0.01:
            self.current_value = expected_value
        expected_pnl = round(
            self.current_value - (self.quantity * self.avg_cost), 2
        )
        if abs(self.unrealized_pnl - expected_pnl) > 0.01:
            self.unrealized_pnl = expected_pnl
        cost_basis = self.quantity * self.avg_cost
        if cost_basis > 0:
            expected_pct = round(self.unrealized_pnl / cost_basis, 6)
            if abs(self.pnl_pct - expected_pct) > 1e-4:
                self.pnl_pct = expected_pct
        return self


class Portfolio(BaseModel):
    """Aggregated view of a user's investment portfolio.

    Computed on-the-fly from individual holdings or materialised in the
    database for fast dashboard loads.  A model validator recalculates
    ``total_value``, ``total_investment``, and ``total_pnl`` from the
    holdings list to guard against stale cached values.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    holdings: list[PortfolioHolding] = Field(
        default_factory=list,
        description="Ordered list of portfolio positions.",
    )
    total_value: float = Field(
        default=0.0,
        ge=0.0,
        description="Sum of all holding current values.",
    )
    total_investment: float = Field(
        default=0.0,
        ge=0.0,
        description="Sum of all cost bases.",
    )
    total_pnl: float = Field(
        default=0.0,
        description="Absolute P&L across the portfolio.",
    )
    daily_pnl: float = Field(
        default=0.0,
        description="Change in total_value since the previous trading day.",
    )
    sharpe_ratio: float = Field(
        default=0.0,
        description="Annualised Sharpe ratio of the portfolio.",
    )
    beta: float = Field(
        default=1.0,
        description="Portfolio beta w.r.t. the benchmark index.",
    )
    alpha: float = Field(
        default=0.0,
        description="Annualised alpha over the benchmark.",
    )
    sector_allocation: dict[str, float] = Field(
        default_factory=dict,
        description="Map of sector → weight (sums to 1).",
    )
    asset_allocation: dict[str, float] = Field(
        default_factory=dict,
        description="Map of asset class → weight (sums to 1).",
    )

    @model_validator(mode="after")
    def recalculate_aggregates(self) -> Portfolio:
        """Recalculate total_value and total_investment from holdings."""
        if self.holdings:
            self.total_value = round(
                sum(h.current_value for h in self.holdings), 2
            )
            self.total_investment = round(
                sum(h.quantity * h.avg_cost for h in self.holdings), 2
            )
            self.total_pnl = round(self.total_value - self.total_investment, 2)
        return self


class RiskMetrics(BaseModel):
    """Comprehensive risk analytics for a portfolio or single position.

    Produced by the risk analyst agent and consumed by the portfolio
    manager agent and the personal CFO agent.  All VaR/CVaR values are
    expressed as positive numbers representing potential losses in the
    portfolio's base currency.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    var_95: Optional[float] = Field(
        default=None,
        description="Value-at-risk at 95 % confidence (positive number = loss).",
    )
    var_99: Optional[float] = Field(
        default=None,
        description="Value-at-risk at 99 % confidence.",
    )
    cvar_95: Optional[float] = Field(
        default=None,
        description="Conditional VaR (expected shortfall) at 95 % confidence.",
    )
    cvar_99: Optional[float] = Field(
        default=None,
        description="Conditional VaR (expected shortfall) at 99 % confidence.",
    )
    volatility: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annualised portfolio volatility (standard deviation of returns).",
    )
    max_drawdown: Optional[float] = Field(
        default=None,
        le=0.0,
        description="Maximum observed drawdown as a fraction (−0.25 = −25 %).",
    )
    sharpe_ratio: Optional[float] = Field(
        default=None,
        description="Annualised Sharpe ratio.",
    )
    sortino_ratio: Optional[float] = Field(
        default=None,
        description="Sortino ratio (downside-deviation denominator).",
    )
    treynor_ratio: Optional[float] = Field(
        default=None,
        description="Treynor ratio (beta denominator).",
    )
    beta: Optional[float] = Field(
        default=None,
        description="Portfolio beta w.r.t. benchmark.",
    )
    correlation_matrix: Optional[dict[str, dict[str, float]]] = Field(
        default=None,
        description="Symbol→Symbol→Pearson correlation. Symmetric matrix.",
    )
    sector_concentration: Optional[dict[str, float]] = Field(
        default=None,
        description="Sector→Herfindahl-style concentration weight.",
    )


class PortfolioConstraints(BaseModel):
    """Typed constraints applied during portfolio optimisation.

    All weight fields are fractions in [0, 1].  This model can be
    serialised to/from the ``constraints`` dict on
    ``PortfolioOptimizationRequest`` for use by agents that need
    structured access to constraint fields.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    min_sector_weights: dict[str, float] = Field(
        default_factory=dict,
        description='Minimum required weight per sector (e.g. {"BANKING": 0.05}).',
    )
    max_sector_weights: dict[str, float] = Field(
        default_factory=dict,
        description='Maximum allowed weight per sector (e.g. {"IT": 0.30}).',
    )
    max_single_stock_weight: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="Hard cap on the weight of any single position.",
    )
    tax_loss_harvesting_enabled: bool = Field(
        default=False,
        description="Whether the optimiser should consider tax-loss harvesting.",
    )

    @field_validator("min_sector_weights", "max_sector_weights")
    @classmethod
    def weights_must_be_valid_fractions(
        cls,
        v: dict[str, float],
    ) -> dict[str, float]:
        """Validate that all sector weights are in [0, 1]."""
        for sector, weight in v.items():
            if not (0.0 <= weight <= 1.0):
                raise ValueError(
                    f"sector weight for '{sector}' must be between 0.0 and 1.0, "
                    f"got {weight}"
                )
        return v


class PortfolioOptimizationRequest(BaseModel):
    """Input to the portfolio optimisation pipeline.

    The portfolio manager agent constructs this from the user's current
    holdings and risk preferences, then invokes the optimiser backend.

    The ``constraints`` field is a flexible dict that typically contains:
    - ``min_sector_weights`` (dict[str, float]): minimum weight per sector.
    - ``max_sector_weights`` (dict[str, float]): maximum weight per sector.
    - ``max_single_stock_weight`` (float): hard cap on any single position.
    - ``tax_loss_harvesting_enabled`` (bool): enable tax-loss harvesting.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    holdings: list[PortfolioHolding] = Field(
        min_length=1,
        description="Current portfolio positions.",
    )
    risk_tolerance: RiskLevel = Field(
        default=RiskLevel.MEDIUM,
        description="User's self-reported risk tolerance.",
    )
    investment_horizon: str = Field(
        default="1Y",
        description="Expected holding period, e.g. ``6M``, ``1Y``, ``5Y``.",
    )
    target_return: float = Field(
        default=0.10,
        ge=0.0,
        description="Desired annualised return (e.g. 0.12 for 12 %).",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optimisation constraints — min/max sector weights, "
            "max single stock weight, tax loss harvesting flag, etc."
        ),
    )


class RebalancingAction(BaseModel):
    """A single recommended trade to rebalance the portfolio.

    Can be serialised to/from dicts when used inside
    ``PortfolioOptimizationResult.rebalancing_actions``.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    symbol: str = Field(min_length=1, description="Ticker symbol to trade.")
    action: str = Field(description="``BUY``, ``SELL``, or ``HOLD``.")
    shares: float = Field(
        description="Number of shares to buy (positive) or sell (negative).",
    )
    current_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Current portfolio weight of this symbol.",
    )
    target_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Target portfolio weight after rebalancing.",
    )


class PortfolioOptimizationResult(BaseModel):
    """Output of the portfolio optimisation pipeline.

    Includes the optimal weight vector, projected risk/return metrics,
    and a concrete list of rebalancing trades (as dicts for flexibility).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    optimal_weights: dict[str, float] = Field(
        description="Symbol → target weight. Weights sum to 1.0.",
    )
    expected_return: float = Field(
        description="Projected annualised return of the optimal portfolio.",
    )
    expected_risk: float = Field(
        ge=0.0,
        description="Projected annualised volatility (standard deviation of returns).",
    )
    sharpe_ratio: float = Field(
        description="Projected Sharpe ratio of the optimal portfolio.",
    )
    rebalancing_actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Concrete trade recommendations to reach the optimal weights.",
    )
    turnover: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Portfolio turnover fraction — sum of absolute weight changes / 2.",
    )

    @field_validator("optimal_weights")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that optimal weights approximately sum to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"optimal_weights must sum to 1.0 (got {total:.4f})"
            )
        return v


# ============================================================
# RAG (Retrieval-Augmented Generation) Models
# ============================================================


class RAGDocument(BaseModel):
    """A full document stored or referenced in the RAG system.

    The ``metadata`` dict should include at least:
    - ``source`` (str): origin, e.g. ``sec_filing``, ``news_article``.
    - ``doc_type`` (str): MIME-like type hint, e.g. ``text``, ``pdf``, ``html``.
    - ``date`` (str | None): publication or ingestion date.
    - ``author`` (str | None): author or publishing organisation.
    - ``ticker_symbols`` (list[str]): equity tickers referenced in the document.

    The ``embedding`` field is populated only when the document is stored
    in a vector database that supports whole-document embeddings.  In chunk-
    based retrieval systems, embeddings live on ``RAGChunk`` instead.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    doc_id: str = Field(
        min_length=1,
        description="Unique document identifier (typically a UUID or hash).",
    )
    content: str = Field(
        min_length=1,
        description="Full text content of the document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured metadata: source, doc_type, date, author, ticker_symbols, "
            "and any additional key-value pairs."
        ),
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Pre-computed embedding vector (dimension depends on the model).",
    )


class RAGChunk(BaseModel):
    """A semantic chunk extracted from a ``RAGDocument`` for retrieval.

    Chunks are the atomic retrieval unit — vector search returns chunks,
    not full documents.  Each chunk carries its own embedding and
    optional metadata (e.g. page number, section heading).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    chunk_id: str = Field(
        min_length=1,
        description="Unique chunk identifier.",
    )
    doc_id: str = Field(
        min_length=1,
        description="Parent ``RAGDocument.doc_id``.",
    )
    content: str = Field(
        min_length=1,
        description="Text content of this chunk.",
    )
    chunk_index: int = Field(
        ge=0,
        description="Ordinal position of this chunk within the parent document.",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Embedding vector for this chunk.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk-level metadata (e.g. page number, section heading).",
    )


class RetrievalResult(BaseModel):
    """Result set returned by the RAG retrieval pipeline.

    Includes the ranked chunks, their similarity scores, the original
    query, and a flag indicating whether re-ranking was applied.
    A model validator ensures that ``scores`` and ``chunks`` are always
    the same length.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    chunks: list[RAGChunk] = Field(
        default_factory=list,
        description="Retrieved chunks, ordered by relevance.",
    )
    scores: list[float] = Field(
        default_factory=list,
        description="Similarity / relevance score for each chunk (same order).",
    )
    query: str = Field(
        min_length=1,
        description="The query that produced this result set.",
    )
    reranked: bool = Field(
        default=False,
        description="Whether a cross-encoder or LLM re-ranker was applied.",
    )

    @model_validator(mode="after")
    def scores_chunks_aligned(self) -> RetrievalResult:
        """Ensure scores and chunks lists are the same length."""
        if len(self.scores) != len(self.chunks):
            raise ValueError(
                f"scores length ({len(self.scores)}) must match "
                f"chunks length ({len(self.chunks)})"
            )
        return self


class MarketRAGQuery(BaseModel):
    """Structured query targeting market-related documents in the RAG store.

    Allows scoping by ticker symbols, date range, and document source type
    to narrow the retrieval space.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    query: str = Field(
        min_length=1,
        description="Natural-language query string.",
    )
    tickers: Optional[list[str]] = Field(
        default=None,
        description="Optional list of ticker symbols to scope the search.",
    )
    date_range: Optional[tuple[date_type, date_type]] = Field(
        default=None,
        description="Optional (start, end) date filter.",
    )
    document_types: list[str] = Field(
        default_factory=list,
        description='Allowed document source types, e.g. ["sec_filing", "earnings_call"].',
    )


class PersonalFinanceRAGQuery(BaseModel):
    """Structured query targeting personal finance and tax documents.

    Includes jurisdiction awareness so the system returns results relevant
    to the user's country (e.g. Indian Income Tax Act vs. US IRS rules).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    query: str = Field(
        min_length=1,
        description="Natural-language query string.",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Document categories to search: ``tax``, ``insurance``, "
        "``investment``, ``retirement``, ``loan``.",
    )
    jurisdiction: str = Field(
        default="India",
        description="Legal jurisdiction for the query (``India`` or ``US``).",
    )


# ============================================================
# Memory Models
# ============================================================


class MemoryEntry(BaseModel):
    """A single memory record stored in the long-term memory system.

    Memories are classified into three types:
    - **session**: short-lived, scoped to the current conversation.
    - **semantic**: general knowledge facts extracted from conversations.
    - **episodic**: specific events or decisions the user made.

    The ``importance_score`` drives eviction policies and retrieval ranking
    — higher-scoring memories are retained longer and surfaced first.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    memory_id: str = Field(
        min_length=1,
        description="Unique identifier for this memory.",
    )
    user_id: str = Field(
        min_length=1,
        description="Owner of this memory.",
    )
    content: str = Field(
        min_length=1,
        description="Textual content of the memory.",
    )
    memory_type: str = Field(
        default="semantic",
        description="One of ``session``, ``semantic``, or ``episodic``.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this memory was created.",
    )
    importance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative importance used for eviction and retrieval ranking.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description='Free-text tags for filtering (e.g. ["tax", "FY2024"]).',
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata.",
    )

    @field_validator("memory_type")
    @classmethod
    def memory_type_must_be_valid(cls, v: str) -> str:
        """Validate memory_type is one of the allowed values."""
        allowed = {"session", "semantic", "episodic"}
        if v not in allowed:
            raise ValueError(
                f"memory_type must be one of {allowed}, got '{v}'"
            )
        return v


class ConversationContext(BaseModel):
    """Full conversational state for a user session.

    Maintained by the orchestrator and passed to every agent invocation so
    that agents can reference prior turns, extracted entities, and the
    user's current financial context.

    The ``messages`` field stores the raw conversation history as a list
    of dicts, each containing at minimum ``role`` and ``content`` keys.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    user_id: str = Field(
        min_length=1,
        description="Identifier of the user whose conversation this is.",
    )
    session_id: str = Field(
        min_length=1,
        description="Unique session identifier.",
    )
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered history of all messages in this session.",
    )
    current_intent: Optional[str] = Field(
        default=None,
        description="The system's best guess at the user's current intent.",
    )
    entities_mentioned: list[str] = Field(
        default_factory=list,
        description="Extracted entity mentions (tickers, amounts, dates, etc.).",
    )
    financial_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of the user's portfolio, risk profile, and recent transactions.",
    )


class UserProfile(BaseModel):
    """Persistent user profile used for personalization and compliance checks.

    Fields like ``kyc_status`` gate access to certain features (e.g.
    trading recommendations are only shown to verified users).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    user_id: str = Field(min_length=1, description="Unique user identifier.")
    name: str = Field(default="", description="Display name.")
    age: Optional[int] = Field(
        default=None,
        ge=0,
        le=150,
        description="User's age in years.",
    )
    income_range: Optional[str] = Field(
        default=None,
        description="Income bracket label, e.g. ``10L-25L``, ``25L-50L``, ``>$200K``.",
    )
    risk_profile: RiskLevel = Field(
        default=RiskLevel.MEDIUM,
        description="Assessed or self-reported risk profile.",
    )
    investment_style: Optional[str] = Field(
        default=None,
        description="Style label: ``value``, ``growth``, ``balanced``, "
        "``conservative``, ``aggressive``.",
    )
    financial_goals: list[str] = Field(
        default_factory=list,
        description='Free-text list of goals, e.g. ["retirement", "child education"].',
    )
    tax_jurisdiction: str = Field(
        default="India",
        description="Primary tax jurisdiction.",
    )
    kyc_status: str = Field(
        default="pending",
        description="KYC verification status: ``pending``, ``verified``, ``rejected``.",
    )


# ============================================================
# Critic Models
# ============================================================


class CritiqueResult(BaseModel):
    """Structured output of the critic agent's quality assessment.

    The overall ``score`` is a weighted summary of the individual dimension
    scores stored in ``dimensions``.  If ``revision_needed`` is ``True``,
    the producing agent is asked to revise its output before it is shown
    to the user.

    The ``dimensions`` dict should contain the following keys, each scored
    on a [0, 1] scale:
    - ``relevance``: How well the output addresses the user's query.
    - ``accuracy``: Factual correctness of claims and numbers.
    - ``completeness``: Whether all aspects of the query are covered.
    - ``coherence``: Logical consistency and readability.
    - ``actionability``: Whether the output provides clear next steps.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall weighted quality score (0–1).",
    )
    dimensions: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-dimension quality scores. Expected keys: relevance, accuracy, "
            "completeness, coherence, actionability. Each in [0, 1]."
        ),
    )
    feedback: str = Field(
        default="",
        description="Free-text feedback from the critic.",
    )
    revision_needed: bool = Field(
        default=False,
        description="Whether the producing agent should revise the output.",
    )
    suggested_improvements: list[str] = Field(
        default_factory=list,
        description="Actionable suggestions for improving the output.",
    )

    @field_validator("dimensions")
    @classmethod
    def dimension_scores_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that all dimension scores are in [0, 1]."""
        for key, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"dimension '{key}' score must be between 0.0 and 1.0, "
                    f"got {score}"
                )
        return v


class QualityRubric(BaseModel):
    """A rubric entry defining how a single quality dimension should be scored.

    Used to prompt the critic LLM and to standardise evaluation across
    sessions.  The ``weight`` field determines how much each dimension
    contributes to the overall ``CritiqueResult.score``.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    dimension: str = Field(
        min_length=1,
        description="Name of the quality dimension this rubric defines.",
    )
    weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Relative weight in the overall score computation.",
    )
    criteria: str = Field(
        min_length=1,
        description="Natural-language description of what this dimension measures.",
    )
    scoring_guide: str = Field(
        default="",
        description="Detailed scoring guide mapping score ranges "
        "(e.g. 0.0–0.3, 0.3–0.7, 0.7–1.0) to qualitative levels.",
    )


# ============================================================
# Query / Response Models
# ============================================================


class UserQuery(BaseModel):
    """Canonical representation of a user's incoming query.

    Created by the API layer after initial intent classification and entity
    extraction, then routed to the appropriate agent(s).

    The ``entities`` dict holds extracted structured data such as ticker
    symbols, monetary amounts, dates, and named entities discovered by
    the NLU pipeline.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    query_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this query.",
    )
    user_id: str = Field(
        min_length=1,
        description="Identifier of the user who submitted the query.",
    )
    text: str = Field(
        min_length=1,
        description="Raw query text as typed by the user.",
    )
    query_type: QueryType = Field(
        default=QueryType.GENERAL,
        description="Classified query type for agent routing.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Fine-grained intent label from the classifier, "
        "e.g. ``compare_stocks``.",
    )
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities such as tickers, amounts, dates, etc.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context from the conversation history or session.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the query was received (UTC).",
    )


class AgentResponse(BaseModel):
    """A single agent's contribution to the overall system response.

    Used internally by the orchestrator before consolidation into a
    ``SystemResponse``.  Can also be serialised to a dict for storage
    in ``SystemResponse.agent_responses``.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    agent_type: AgentType = Field(
        description="Which agent produced this response.",
    )
    response_text: str = Field(
        min_length=1,
        description="The agent's analysis or recommendation.",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed confidence.",
    )
    data_sources: list[str] = Field(
        default_factory=list,
        description="Data sources consulted by the agent.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent-specific metadata.",
    )


class SystemResponse(BaseModel):
    """Final consolidated response delivered to the user.

    Assembles individual agent contributions (as dicts for flexibility)
    into a single coherent answer, optionally ranked and synthesised by
    the personal CFO agent.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    query_id: UUID = Field(
        description="References the originating ``UserQuery.query_id``.",
    )
    agent_responses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Individual agent contributions (serialised as dicts).",
    )
    final_answer: str = Field(
        min_length=1,
        description="The consolidated answer shown to the user.",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall system confidence in the answer.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of data sources and references backing the answer.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations extracted from the answer.",
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions the user might want to ask.",
    )


# ============================================================
# Utility / Helper Models
# ============================================================


class DateRange(BaseModel):
    """A generic date-range filter used across queries.

    Enforces that ``start`` is not after ``end``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: date_type = Field(description="Inclusive start date.")
    end: date_type = Field(description="Inclusive end date.")

    @model_validator(mode="after")
    def start_before_end(self) -> DateRange:
        """Ensure the start date is not after the end date."""
        if self.start > self.end:
            raise ValueError(
                f"start date ({self.start}) must be <= end date ({self.end})"
            )
        return self


class Pagination(BaseModel):
    """Pagination parameters for list endpoints."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    page: int = Field(
        default=1,
        ge=1,
        description="1-indexed page number.",
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page.",
    )

    @property
    def offset(self) -> int:
        """Zero-based offset for database queries."""
        return (self.page - 1) * self.page_size


class HealthCheckResponse(BaseModel):
    """Response model for the health-check endpoint.

    Aggregates subsystem status into a single response so that load
    balancers and monitoring tools can determine overall service health.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    status: str = Field(
        default="healthy",
        description="``healthy``, ``degraded``, or ``unhealthy``.",
    )
    version: str = Field(
        default="1.0.0",
        description="Application version string.",
    )
    uptime_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds since application start.",
    )
    checks: dict[str, str] = Field(
        default_factory=dict,
        description='Individual subsystem health: {"db": "ok", "redis": "ok", ...}.',
    )
