"""Query Router — classifies user queries and determines agent activation."""
from __future__ import annotations

import json
import re
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from backend.config.schemas import QueryType

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────
# Routing table
# ────────────────────────────────────────────────────────────────

# Each entry maps a QueryType to:
#   (sequential_agents_after_parallel, parallel_groups)
#
# *parallel_groups* is a list of lists — agents within the same inner
# list execute concurrently; groups themselves execute sequentially.
_ROUTING_TABLE: dict[QueryType, tuple[list[str], list[list[str]]]] = {
    QueryType.MARKET_QUERY: (
        ["portfolio_manager"],
        [
            ["market_analyst", "news_sentiment", "risk_analyst"],
        ],
    ),
    QueryType.PORTFOLIO_QUERY: (
        [],
        [
            ["portfolio_manager", "risk_analyst"],
        ],
    ),
    QueryType.RISK_QUERY: (
        [],
        [
            ["risk_analyst"],
        ],
    ),
    QueryType.PERSONAL_FINANCE_QUERY: (
        [],
        [
            ["personal_cfo"],
        ],
    ),
    QueryType.GENERAL: (
        [],
        [
            ["personal_cfo", "market_analyst"],
        ],
    ),
}

# ── Classification prompt ───────────────────────────────────────

_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial query classifier.  Analyse the user query and "
            "return **only** a JSON object with a single key \"query_type\" whose "
            "value is one of: {valid_types}.\n\n"
            "Guidelines:\n"
            "- market_query: questions about stocks, indices, sectors, economy, "
            "price targets, earnings.\n"
            "- portfolio_query: questions about building, optimising, or "
            "rebalancing a portfolio; asset allocation.\n"
            "- risk_query: questions about VaR, drawdown, volatility, "
            "stress testing, hedging.\n"
            "- personal_finance_query: budgeting, savings, insurance, tax "
            "planning, retirement, EMIs.\n"
            "- general: anything else or ambiguous queries.\n\n"
            "Return ONLY the JSON object, nothing else.",
        ),
        ("human", "{query}"),
    ]
)


# ────────────────────────────────────────────────────────────────
# Entity extraction patterns
# ────────────────────────────────────────────────────────────────

_TICKER_RE = re.compile(
    r"\b([A-Z]{2,5}(?:\.[A-Z]{1,2})?)\b"
)
_AMOUNT_RE = re.compile(
    r"(?:₹|INR|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?(?:\s?(?:cr|lakh|L|K|M|B))?)"
    r"|(?:\$|USD)\s?([\d,]+(?:\.\d{1,2})?(?:\s?(?:K|M|B))?)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b"
    r"|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"(?:\s+\d{1,2})?(?:\s+\d{2,4})?)\b",
    re.IGNORECASE,
)
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "BANKING": ["bank", "banking", "hdfc", "sbi", "icici", "kotak", "axis"],
    "IT": ["it ", "infotech", "infosys", "tcs", "wipro", "hcl", "tech"],
    "PHARMA": ["pharma", "pharmaceutical", "sun pharma", "cipra", "dr reddy"],
    "OIL_GAS": ["oil", "gas", "reliance", "ongc", "petroleum", "energy"],
    "FMCG": ["fmcg", "itc", "hindustan", "nestle", "britannia", "consumer"],
    "AUTO": ["auto", "maruti", "tata motors", "mahindra", "vehicle", "ev"],
    "REAL_ESTATE": ["real estate", "dlf", "godrej prop", "housing", "construction"],
}


# ────────────────────────────────────────────────────────────────
# QueryRouter
# ────────────────────────────────────────────────────────────────


class QueryRouter:
    """Classifies user queries and determines which agents should handle them.

    Supports both LLM-based classification (when an LLM is provided) and a
    keyword-heuristic fallback for environments without an LLM.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    # ── Public API ──────────────────────────────────────────────

    def classify_query(self, query: str) -> QueryType:
        """Return the :class:`QueryType` for *query*.

        Uses LLM-based classification when an LLM is available, otherwise
        falls back to a keyword heuristic.
        """
        if self._llm is not None:
            return self._classify_via_llm(query)
        return self._classify_via_keywords(query)

    def determine_active_agents(
        self, query_type: QueryType | str, query: str
    ) -> tuple[list[str], list[list[str]]]:
        """Return ``(sequential_agents, parallel_groups)`` for *query_type*.

        *sequential_agents* run **after** all parallel groups finish.  They
        receive the merged results of every parallel agent as additional
        context, enabling downstream refinement (e.g. portfolio_manager
        seeing market_analyst output).

        *parallel_groups* is a list of agent-name lists; agents within the
        same inner list execute concurrently.  Multiple groups run one after
        another.
        """
        # Normalise query_type to enum
        if isinstance(query_type, str):
            try:
                query_type = QueryType(query_type)
            except ValueError:
                query_type = QueryType.GENERAL

        sequential, parallel = _ROUTING_TABLE.get(
            query_type, _ROUTING_TABLE[QueryType.GENERAL]
        )

        logger.info(
            "query_router.determine_active_agents",
            query_type=query_type.value,
            sequential=sequential,
            parallel_groups=parallel,
        )

        return list(sequential), [list(g) for g in parallel]

    def extract_entities(self, query: str) -> dict[str, Any]:
        """Extract financial entities from *query*.

        Returns a dict with keys: ``tickers``, ``amounts``, ``dates``, ``sectors``.
        """
        return {
            "tickers": self._extract_tickers(query),
            "amounts": self._extract_amounts(query),
            "dates": self._extract_dates(query),
            "sectors": self._extract_sectors(query),
        }

    # ── LLM classification ──────────────────────────────────────

    def _classify_via_llm(self, query: str) -> QueryType:
        valid_types = ", ".join(t.value for t in QueryType)
        chain = _CLASSIFICATION_PROMPT | self._llm

        try:
            response = chain.invoke({"query": query, "valid_types": valid_types})
            text = response.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)
            parsed = json.loads(text)
            raw_type = parsed.get("query_type", "").strip().lower()
            return QueryType(raw_type)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "query_router.llm_classification_failed",
                error=str(exc),
                query=query[:80],
            )
            return self._classify_via_keywords(query)

    # ── Keyword heuristic fallback ──────────────────────────────

    @staticmethod
    def _classify_via_keywords(query: str) -> QueryType:
        """Rule-based classification when no LLM is available."""
        q = query.lower()

        risk_keywords = [
            "risk", "var", "drawdown", "volatility", "hedge", "stress test",
            "sharpe", "beta", "correlation", "max loss", "downside",
        ]
        if any(kw in q for kw in risk_keywords):
            return QueryType.RISK_QUERY

        portfolio_keywords = [
            "portfolio", "allocation", "rebalance", "weight", "diversif",
            "optimize", "optimise", "asset mix", "construct",
        ]
        if any(kw in q for kw in portfolio_keywords):
            return QueryType.PORTFOLIO_QUERY

        personal_keywords = [
            "budget", "save", "savings", "emi", "insurance", "tax",
            "retire", "mutual fund sip", "fd", "fixed deposit", "expense",
            "income", "salary", "loan", "mortgage",
        ]
        if any(kw in q for kw in personal_keywords):
            return QueryType.PERSONAL_FINANCE_QUERY

        market_keywords = [
            "stock", "share", "nifty", "sensex", "index", "price",
            "earnings", "dividend", "ipo", "market", "sector", "gdp",
            "inflation", "rbi", "fed", "interest rate", "bull", "bear",
            "rally", "crash",
        ]
        if any(kw in q for kw in market_keywords):
            return QueryType.MARKET_QUERY

        return QueryType.GENERAL

    # ── Entity extraction helpers ───────────────────────────────

    @staticmethod
    def _extract_tickers(query: str) -> list[str]:
        """Return uppercase ticker-like tokens (2-5 all-caps letters)."""
        matches = _TICKER_RE.findall(query)
        # Filter out common English words that look like tickers
        stopwords = {
            "THE", "AND", "FOR", "NOT", "YOU", "ALL", "CAN", "HER",
            "WAS", "ONE", "OUR", "OUT", "ARE", "HAS", "HIS", "HOW",
            "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO",
            "DID", "GET", "HIM", "LET", "SAY", "SHE", "TOO",
            "USE", "CEO", "CFO", "CTO", "ROI", "EPS", "PE", "GDP",
            "USA", "UK", "EU", "INR", "USD",
        }
        return sorted({m for m in matches if m not in stopwords})

    @staticmethod
    def _extract_amounts(query: str) -> list[str]:
        matches = _AMOUNT_RE.findall(query)
        amounts: list[str] = []
        for group in matches:
            for part in group:
                if part:
                    amounts.append(part.replace(",", "").strip())
        return amounts

    @staticmethod
    def _extract_dates(query: str) -> list[str]:
        return _DATE_RE.findall(query)

    @staticmethod
    def _extract_sectors(query: str) -> list[str]:
        q_lower = query.lower()
        found: list[str] = []
        for sector, keywords in _SECTOR_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                found.append(sector)
        return found

