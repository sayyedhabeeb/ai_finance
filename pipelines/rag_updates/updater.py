"""
RAG document update pipeline.

Keeps the vector stores (Weaviate for market data, pgvector for personal
finance) in sync with fresh documents:

  - **Market RAG**: Research reports, NSE/BSE filings, annual reports,
    earnings call transcripts, analyst notes.
  - **Personal Finance RAG**: Updated tax rules (IT Act), RBI regulations,
    SEBI investor guidelines, GST notifications.

Also handles staleness detection and embedding rebuilds for documents
whose content or underlying facts may have changed.

Typical usage::

    updater = RAGUpdatePipeline()
    updater.update_market_rag()
    updater.update_personal_finance_rag()
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sources for market research documents
# ---------------------------------------------------------------------------

NSE_FILING_URLS: list[str] = [
    "https://www.nseindia.com/companies-listing/corporate-filings",
    "https://www.nseindia.com/market-data/equities",
]

BSE_FILING_URLS: list[str] = [
    "https://www.bseindia.com/corporates/corporate.html",
]

RESEARCH_SOURCES: list[dict[str, str]] = [
    # Screener.in financial data
    {"name": "Screener.in", "url": "https://www.screener.in/company/{symbol}/consolidated/"},
    # Moneycontrol financials
    {"name": "Moneycontrol", "url": "https://www.moneycontrol.com/financials/{symbol}/results/yearly/{symbol}#results"},
    # Trendlyne
    {"name": "Trendlyne", "url": "https://trendlyne.com/equity/{symbol}/"},
]

# Symbols to track for corporate filings
_TRACKED_SYMBOLS: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE",
    "HINDUNILVR", "MARUTI", "SUNPHARMA", "TATAMOTORS", "WIPRO",
]

# Personal finance regulation sources
REGULATION_SOURCES: list[dict[str, str]] = [
    {"name": "Income Tax India", "url": "https://www.incometax.gov.in/iec/foportal/help/article/section-wise-summary"},
    {"name": "CBDT Circulars", "url": "https://www.incometax.gov.in/iec/foportal/pages/common/circulars.html"},
    {"name": "SEBI Investor", "url": "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes&intmId=13"},
    {"name": "RBI Notifications", "url": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?Id=22602"},
    {"name": "GST Council Updates", "url": "https://www.gst.gov.in"},
    {"name": "PFRDA", "url": "https://www.pfrda.org.in"},
]


class RAGUpdatePipeline:
    """Keeps RAG systems updated with fresh data.

    Parameters
    ----------
    weaviate_url : str or None
        Weaviate URL for market documents.
    pgvector_url : str or None
        PostgreSQL + pgvector URL for personal finance documents.
    embedder : Any or None
        Pre-initialised :class:`EmbeddingService` instance.
    """

    def __init__(
        self,
        weaviate_url: Optional[str] = None,
        pgvector_url: Optional[str] = None,
        embedder: Any = None,
    ) -> None:
        self._weaviate_url = weaviate_url
        self._pgvector_url = pgvector_url
        self._embedder = embedder
        self._http_client = httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": "AI-Financial-Brain-RAGUpdater/1.0"},
        )

    def close(self) -> None:
        """Close HTTP client."""
        self._http_client.close()

    def __enter__(self) -> "RAGUpdatePipeline":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Market RAG Updates
    # ------------------------------------------------------------------

    def update_market_rag(
        self,
        symbols: Optional[list[str]] = None,
        document_types: Optional[list[str]] = None,
    ) -> dict[str, int]:
        """Ingest new market research reports, filings, and articles.

        Parameters
        ----------
        symbols : list[str] or None
            NSE symbols to check for new filings.  Defaults to
            :data:`_TRACKED_SYMBOLS`.
        document_types : list[str] or None
            Types of documents to fetch.  Defaults to
            ``["annual_report", "quarterly_report", "research_report"]``.

        Returns
        -------
        dict
            Counts: ``{fetched: int, ingested: int, errors: int}``.
        """
        target_symbols = symbols or _TRACKED_SYMBOLS
        doc_types = document_types or ["annual_report", "quarterly_report", "research_report"]

        stats = {"fetched": 0, "ingested": 0, "errors": 0}

        # Use the existing MarketDocumentIngestionPipeline if available
        try:
            from backend.rag.market.weaviate_client import MarketWeaviateClient
            from backend.rag.embeddings.embedder import EmbeddingService
            from backend.rag.chunking.strategies import DocumentChunker
            from backend.rag.market.ingestion_pipeline import MarketDocumentIngestionPipeline

            wv_client = MarketWeaviateClient(url=self._weaviate_url or "http://localhost:8080")
            embedder = self._embedder or EmbeddingService()
            chunker = DocumentChunker()

            with MarketDocumentIngestionPipeline(
                weaviate_client=wv_client,
                embedder=embedder,
                chunker=chunker,
            ) as pipeline:
                # 1. Ingest from research source URLs
                for source in RESEARCH_SOURCES:
                    for sym in target_symbols[:5]:  # Limit to avoid rate limits
                        try:
                            url = source["url"].format(symbol=sym.lower())
                            chunks = pipeline.ingest_url(
                                url,
                                metadata={
                                    "doc_type": "research_report",
                                    "source_name": source["name"],
                                    "ticker_symbols": [sym],
                                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                },
                            )
                            stats["fetched"] += 1
                            stats["ingested"] += chunks
                        except Exception as exc:
                            stats["errors"] += 1
                            logger.debug("Failed to ingest %s for %s: %s", source["name"], sym, exc)

                # 2. Ingest from NSE/BSE filing pages
                for url in NSE_FILING_URLS + BSE_FILING_URLS:
                    try:
                        chunks = pipeline.ingest_url(
                            url,
                            metadata={
                                "doc_type": "nse_filing",
                                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                            },
                        )
                        stats["fetched"] += 1
                        stats["ingested"] += chunks
                    except Exception as exc:
                        stats["errors"] += 1
                        logger.debug("Failed to ingest filing page %s: %s", url, exc)

            logger.info("Market RAG update: %s", stats)
            return stats

        except ImportError as exc:
            logger.error("Cannot import RAG components: %s", exc)
            return stats

    # ------------------------------------------------------------------
    # Personal Finance RAG Updates
    # ------------------------------------------------------------------

    def update_personal_finance_rag(
        self,
        topics: Optional[list[str]] = None,
    ) -> dict[str, int]:
        """Update personal finance RAG with new tax rules and regulations.

        Parameters
        ----------
        topics : list[str] or None
            Specific topics to update.  Defaults to a comprehensive set
            covering Indian tax, investment, insurance, and retirement rules.

        Returns
        -------
        dict
            Counts: ``{fetched: int, ingested: int, errors: int}``.
        """
        default_topics = [
            "income_tax_slabs_2024",
            "ltcg_tax_rules_equity",
            "stcg_tax_rules",
            "section_80c_deductions",
            "section_80d_health_insurance",
            "section_80ccd_nps",
            "section_24_home_loan_interest",
            "dividend_taxation_2024",
            "capital_gains_indexation",
            "nps_withdrawal_rules",
            "ppf_withdrawal_rules",
            "epf_withdrawal_rules",
            "gst_rates_financial_services",
            "tds_rates_2024",
            "advance_tax_dates",
            "itr_filing_deadlines",
            "mutual_fund_taxation",
            "sip_withdrawal_tax",
            "crypto_taxation_india",
            "international_investing_tax",
        ]
        target_topics = topics or default_topics

        stats = {"fetched": 0, "ingested": 0, "errors": 0}

        try:
            from backend.rag.personal_finance.pgvector_client import PgvectorRAGClient
            from backend.rag.embeddings.embedder import EmbeddingService

            pg_client = PgvectorRAGClient(
                connection_string=self._pgvector_url
                or "postgresql://postgres:postgres@localhost:5432/financial_brain_rag"
            )
            embedder = self._embedder or EmbeddingService()

            # 1. Fetch and ingest from regulation source pages
            for source in REGULATION_SOURCES:
                try:
                    content = self._fetch_web_content(source["url"])
                    if content and len(content) > 100:
                        # Split into manageable chunks
                        chunks = self._split_content_for_ingestion(
                            content,
                            title=source["name"],
                            source_url=source["url"],
                        )

                        for chunk_data in chunks:
                            embedding = embedder.embed_text(chunk_data["content"])
                            pg_client.upsert_document(
                                doc_id=chunk_data["doc_id"],
                                content=chunk_data["content"],
                                title=chunk_data["title"],
                                source=chunk_data["source"],
                                doc_type="regulation",
                                jurisdiction="IN",
                                category=chunk_data.get("category", "regulation"),
                                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                embedding=embedding,
                            )
                            stats["ingested"] += 1

                        stats["fetched"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    logger.debug("Failed to fetch %s: %s", source["name"], exc)

            # 2. Ingest topic-specific content
            for topic in target_topics:
                try:
                    topic_content = self._generate_topic_content(topic)
                    if topic_content:
                        embedding = embedder.embed_text(topic_content["content"])
                        pg_client.upsert_document(
                            doc_id=topic_content["doc_id"],
                            content=topic_content["content"],
                            title=topic_content["title"],
                            source="ai_financial_brain_knowledge_base",
                            doc_type="knowledge_base",
                            jurisdiction="IN",
                            category=topic_content.get("category", "tax"),
                            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                            embedding=embedding,
                        )
                        stats["ingested"] += 1
                        stats["fetched"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    logger.debug("Failed to ingest topic %s: %s", topic, exc)

            logger.info("Personal Finance RAG update: %s", stats)
            return stats

        except ImportError as exc:
            logger.error("Cannot import personal finance RAG components: %s", exc)
            return stats

    def _fetch_web_content(self, url: str) -> str:
        """Fetch and extract text content from a URL."""
        try:
            response = self._http_client.get(url)
            response.raise_for_status()

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()
        except Exception as exc:
            logger.debug("Failed to fetch URL %s: %s", url, exc)
            return ""

    def _split_content_for_ingestion(
        self,
        content: str,
        title: str,
        source_url: str,
        max_chunk_size: int = 2000,
    ) -> list[dict[str, str]]:
        """Split content into chunks suitable for ingestion."""
        # Split by sections (double newline or headings)
        sections = re.split(r"\n{2,}", content)
        chunks: list[dict[str, str]] = []
        current = ""
        chunk_index = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(current) + len(section) + 2 > max_chunk_size and current:
                doc_id = hashlib.sha256(
                    f"{source_url}:{title}:{chunk_index}".encode()
                ).hexdigest()[:24]
                chunks.append({
                    "doc_id": doc_id,
                    "content": current.strip(),
                    "title": title,
                    "source": source_url,
                    "category": self._detect_category(current),
                })
                chunk_index += 1
                current = section
            else:
                current = f"{current}\n\n{section}" if current else section

        if current.strip():
            doc_id = hashlib.sha256(
                f"{source_url}:{title}:{chunk_index}".encode()
            ).hexdigest()[:24]
            chunks.append({
                "doc_id": doc_id,
                "content": current.strip(),
                "title": title,
                "source": source_url,
                "category": self._detect_category(current),
            })

        return chunks

    @staticmethod
    def _detect_category(text: str) -> str:
        """Detect the regulatory category from text content."""
        text_lower = text.lower()
        category_keywords: dict[str, list[str]] = {
            "tax": ["income tax", "tax slab", "tds", "itr", "advance tax", "section 80"],
            "insurance": ["health insurance", "life insurance", "premium", "claim"],
            "investment": ["mutual fund", "sip", "equity", "bond", "fd", "ppf", "nps"],
            "retirement": ["retirement", "pension", "epf", "vpf", "gratuity"],
            "regulation": ["sebi", "rbi", "regulation", "compliance", "guidelines"],
            "gst": ["gst", "goods and services tax", "igst", "cgst", "sgst"],
        }
        for category, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "general"

    def _generate_topic_content(self, topic: str) -> Optional[dict[str, str]]:
        """Generate structured knowledge-base content for a personal finance topic.

        Returns a dict with doc_id, title, content, and category, or
        ``None`` if the topic is unrecognised.
        """
        # Knowledge base for key Indian personal finance topics
        knowledge: dict[str, dict[str, str]] = {
            "income_tax_slabs_2024": {
                "title": "India Income Tax Slabs FY 2024-25 (New Regime)",
                "category": "tax",
                "content": (
                    "Income Tax Slabs under New Tax Regime FY 2024-25 (Assessment Year 2025-26):\n\n"
                    "0 - 3,00,000: Nil\n"
                    "3,00,001 - 7,00,000: 5%\n"
                    "7,00,001 - 10,00,000: 10%\n"
                    "10,00,001 - 12,00,000: 15%\n"
                    "12,00,001 - 15,00,000: 20%\n"
                    "Above 15,00,000: 30%\n\n"
                    "Standard deduction: Rs 75,000\n"
                    "Rebate under section 87A: Up to Rs 25,000 (taxable income up to Rs 7,00,000)\n"
                    "Surcharge: 10% for income above Rs 50L, 15% above Rs 1Cr, 25% above Rs 2Cr\n"
                    "Health & Education Cess: 4% on total tax + surcharge\n"
                    "Note: New regime is default from FY 2023-24. Opt-out requires filing Form 10-IEA."
                ),
            },
            "ltcg_tax_rules_equity": {
                "title": "Long-Term Capital Gains (LTCG) Tax on Equity - India",
                "category": "tax",
                "content": (
                    "Long-Term Capital Gains (LTCG) on Equity Instruments:\n\n"
                    "Applicable to: Listed equity shares, equity-oriented mutual funds, business trusts\n"
                    "Holding period: More than 12 months\n"
                    "Tax rate: 12.5% (w.e.f. 23 July 2024, earlier 10%)\n"
                    "Exemption limit: Rs 1,25,000 per financial year (increased from Rs 1,00,000)\n"
                    "Cost of acquisition: Higher of actual cost or lower of FMV on 31 Jan 2018 / actual cost\n"
                    "Grandfathering: Gains up to 31 Jan 2018 FMV are exempt\n\n"
                    "Example: If LTCG is Rs 2,00,000, taxable amount = Rs 2,00,000 - Rs 1,25,000 = Rs 75,000\n"
                    "Tax = 12.5% of Rs 75,000 = Rs 9,375 (plus 4% cess = Rs 9,750)\n\n"
                    "Note: Equity LTCG tax is payable under section 112A."
                ),
            },
            "stcg_tax_rules": {
                "title": "Short-Term Capital Gains (STCG) Tax Rules - India",
                "category": "tax",
                "content": (
                    "Short-Term Capital Gains (STCG) Tax Rules:\n\n"
                    "Equity Instruments (holding <= 12 months):\n"
                    "- Listed equity shares, equity mutual funds: 20% (w.e.f. 23 July 2024, earlier 15%)\n\n"
                    "Debt Instruments:\n"
                    "- Listed debt securities, non-equity mutual funds (holding <= 36 months):\n"
                    "  Taxed at individual's slab rate\n\n"
                    "Other Assets (real estate, gold, unlisted shares):\n"
                    "- Holding period <= 36 months (24 months for real estate)\n"
                    "- Taxed at individual's slab rate\n\n"
                    "Key Points:\n"
                    "- STCG is added to total income for tax calculation\n"
                    "- No indexation benefit for STCG\n"
                    "- Securities Transaction Tax (STT) may apply on sale\n"
                    "- Set-off: STCG loss can be set off against STCG gain, then LTCG gain (equity)\n"
                    "- Carry-forward: Unabsorbed STCG loss can be carried forward for 8 years"
                ),
            },
            "section_80c_deductions": {
                "title": "Section 80C Deductions - India Income Tax",
                "category": "tax",
                "content": (
                    "Section 80C - Deductions for Investments and Expenses:\n\n"
                    "Maximum deduction: Rs 1,50,000 per financial year\n\n"
                    "Eligible Investments:\n"
                    "- PPF (Public Provident Fund): Min Rs 500, max Rs 1.5L per year\n"
                    "- ELSS (Equity Linked Savings Scheme): 3-year lock-in\n"
                    "- NSC (National Savings Certificate)\n"
                    "- 5-year Fixed Deposits (post office/bank)\n"
                    "- Senior Citizens Savings Scheme (SCSS)\n"
                    "- NPS (National Pension System) - Tier 1 (up to Rs 50K additional under 80CCD(1B))\n\n"
                    "Eligible Expenses:\n"
                    "- Life Insurance Premium\n"
                    "- Children's Tuition Fee (max 2 children)\n"
                    "- Home Loan Principal Repayment\n"
                    "- Sukanya Samriddhi Account\n\n"
                    "Note: Section 80C deductions are available only under the OLD tax regime.\n"
                    "New tax regime does not allow most 80C deductions (except NPS employer contribution)."
                ),
            },
            "section_80d_health_insurance": {
                "title": "Section 80D - Health Insurance Deduction",
                "category": "tax",
                "content": (
                    "Section 80D - Health Insurance Premium Deduction:\n\n"
                    "Self, Spouse, Children:\n"
                    "- Below 60 years: Rs 25,000\n"
                    "- 60 years and above: Rs 50,000\n\n"
                    "Parents (additional deduction):\n"
                    "- Parents below 60 years: Rs 25,000\n"
                    "- Parents 60+ years: Rs 50,000\n\n"
                    "Maximum Deduction:\n"
                    "- Self + Parents (both below 60): Rs 50,000\n"
                    "- Self below 60 + Parents 60+: Rs 75,000\n"
                    "- Self 60+ + Parents 60+: Rs 1,00,000\n\n"
                    "Preventive Health Check-up:\n"
                    "- Up to Rs 5,000 within the overall limit\n\n"
                    "Eligible for: Medical insurance premiums, CGHS contribution, Ayush policy\n"
                    "Available under: Both Old and New tax regimes"
                ),
            },
            "dividend_taxation_2024": {
                "title": "Dividend Taxation Rules India 2024",
                "category": "tax",
                "content": (
                    "Dividend Taxation in India (Post April 2020):\n\n"
                    "Since Finance Act 2020, dividends are taxable in the hands of the recipient\n"
                    "at their applicable income tax slab rate (removed DDT concept).\n\n"
                    "Key Rules:\n"
                    "- Dividends above Rs 5,000: Company deducts TDS at 10%\n"
                    "- For resident individuals below taxable limit: Submit Form 15G/15H to avoid TDS\n"
                    "- Dividend income is added to total income and taxed at slab rate\n"
                    "- Foreign dividends may be taxed separately or as part of total income\n\n"
                    "Mutual Fund Dividends:\n"
                    "- Equity MF dividends: Taxed at slab rate\n"
                    "- Debt MF dividends: Taxed at slab rate\n\n"
                    "Note: Under the new tax regime, dividend income has no special exemption\n"
                    "and is fully taxable at slab rates."
                ),
            },
            "crypto_taxation_india": {
                "title": "Cryptocurrency Taxation Rules in India",
                "category": "tax",
                "content": (
                    "Cryptocurrency / Virtual Digital Asset (VDA) Taxation India:\n\n"
                    "Since Finance Act 2022 (effective 1 April 2022):\n\n"
                    "Tax on Gains:\n"
                    "- 30% flat tax on transfer of VDA (regardless of holding period)\n"
                    "- No distinction between short-term and long-term\n"
                    "- No indexation benefit\n"
                    "- 1% TDS on transactions above Rs 10,000 (Rs 50,000 in some cases)\n\n"
                    "Key Points:\n"
                    "- Loss from VDA cannot be set off against any other income\n"
                    "- Loss cannot be carried forward\n"
                    "- Gifting of VDA is taxable in recipient's hands\n"
                    "- Mining income is taxed as income from other sources\n"
                    "- Cost of acquisition includes purchase price + charges\n\n"
                    "VDA includes: Crypto, NFTs, and similar digital assets\n"
                    "Excludes: RBI Digital Rupee (CBDC), foreign currency"
                ),
            },
            "mutual_fund_taxation": {
                "title": "Mutual Fund Taxation Rules India 2024",
                "category": "tax",
                "content": (
                    "Mutual Fund Taxation Rules India (post July 2024 amendments):\n\n"
                    "Equity-Oriented Funds (>65% domestic equity):\n"
                    "- STCG (<=12 months): 20%\n"
                    "- LTCG (>12 months): 12.5% above Rs 1,25,000 exemption\n\n"
                    "Debt Funds (<=35% domestic equity):\n"
                    "- From 1 April 2023: All gains taxed at slab rate\n"
                    "- No indexation benefit for investments after 1 April 2023\n"
                    "- Pre-1 Apr 2023 investments: Indexation applies if held >3 years\n\n"
                    "Gold / International Funds:\n"
                    "- STCG (<=3 years): Slab rate\n"
                    "- LTCG (>3 years): 12.5% (no indexation for post-Apr 2023)\n\n"
                    "SIP Taxation:\n"
                    "- Each SIP installment is a separate purchase\n"
                    "- Holding period calculated from each installment's date\n"
                    "- FIFO method for redemption\n\n"
                    "Dividends: Taxed at slab rate (since April 2020)\n"
                    "Switches: Treated as redemption + fresh purchase"
                ),
            },
        }

        entry = knowledge.get(topic)
        if entry is None:
            # Generate a generic stub for unrecognised topics
            return None

        doc_id = hashlib.sha256(
            f"knowledge_base:{topic}:{entry['title']}".encode()
        ).hexdigest()[:24]

        return {
            "doc_id": doc_id,
            "title": entry["title"],
            "content": entry["content"],
            "category": entry.get("category", "general"),
        }

    # ------------------------------------------------------------------
    # Staleness detection
    # ------------------------------------------------------------------

    def detect_stale_documents(
        self,
        collection: str = "market",
        max_age_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Find documents that are older than *max_age_days*.

        Parameters
        ----------
        collection : str
            ``"market"`` or ``"personal_finance"``.
        max_age_days : int
            Maximum document age in days.

        Returns
        -------
        list[dict]
            Stale document metadata: ``{doc_id, title, source, age_days, doc_type}``.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        stale_docs: list[dict[str, Any]] = []

        if collection == "market" and self._weaviate_url:
            stale_docs = self._detect_stale_weaviate(cutoff)
        elif collection == "personal_finance" and self._pgvector_url:
            stale_docs = self._detect_stale_pgvector(cutoff)

        logger.info(
            "Found %d stale documents in '%s' (max_age=%d days).",
            len(stale_docs),
            collection,
            max_age_days,
        )
        return stale_docs

    def _detect_stale_weaviate(self, cutoff: datetime) -> list[dict[str, Any]]:
        """Query Weaviate for stale market documents."""
        try:
            import weaviate

            client = weaviate.connect_to_url(self._weaviate_url)
            try:
                stale: list[dict[str, Any]] = []
                # Check MarketDocuments collection
                try:
                    col = client.collections.get("MarketDocuments")
                    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
                    # Weaviate doesn't have built-in date comparison,
                    # so we fetch recent and filter in Python
                    result = col.query.fetch_objects(limit=500)
                    for obj in result.objects:
                        date_str = obj.properties.get("date", "")
                        if date_str and date_str < cutoff_str:
                            stale.append({
                                "doc_id": obj.properties.get("docId", obj.uuid),
                                "title": obj.properties.get("title", ""),
                                "source": obj.properties.get("source", ""),
                                "age_days": (datetime.now(timezone.utc) - datetime.fromisoformat(date_str.replace("Z", "+00:00"))).days,
                                "doc_type": obj.properties.get("docType", ""),
                            })
                except Exception:
                    pass
                return stale
            finally:
                client.close()
        except Exception as exc:
            logger.error("Weaviate staleness check failed: %s", exc)
            return []

    def _detect_stale_pgvector(self, cutoff: datetime) -> list[dict[str, Any]]:
        """Query pgvector for stale personal finance documents."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self._pgvector_url)
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT doc_id, title, source, doc_type, date,
                               EXTRACT(DAY FROM NOW() - date) as age_days
                        FROM personal_finance_docs
                        WHERE date < :cutoff
                        ORDER BY date ASC
                        LIMIT 500
                    """),
                    {"cutoff": cutoff.strftime("%Y-%m-%d")},
                )
                return [
                    {
                        "doc_id": row.doc_id,
                        "title": row.title,
                        "source": row.source,
                        "age_days": int(row.age_days),
                        "doc_type": row.doc_type,
                    }
                    for row in result
                ]
        except Exception as exc:
            logger.error("pgvector staleness check failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Embedding rebuild
    # ------------------------------------------------------------------

    def rebuild_embeddings(
        self,
        doc_ids: Optional[list[str]] = None,
        collection: str = "market",
        batch_size: int = 50,
    ) -> int:
        """Re-embed stale documents with the latest embedding model.

        Parameters
        ----------
        doc_ids : list[str] or None
            Specific document IDs to rebuild.  If ``None``, rebuilds
            all stale documents detected by :meth:`detect_stale_documents`.
        collection : str
            ``"market"`` or ``"personal_finance"``.
        batch_size : int
            Number of documents to process per batch.

        Returns
        -------
        int
            Number of documents re-embedded.
        """
        if doc_ids is None:
            stale = self.detect_stale_documents(collection)
            doc_ids = [d["doc_id"] for d in stale]

        if not doc_ids:
            logger.info("No documents to re-embed.")
            return 0

        from backend.rag.embeddings.embedder import EmbeddingService

        embedder = self._embedder or EmbeddingService()
        rebuilt = 0

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i : i + batch_size]
            for doc_id in batch_ids:
                try:
                    self._reembed_single(doc_id, collection, embedder)
                    rebuilt += 1
                except Exception as exc:
                    logger.error("Failed to re-embed %s: %s", doc_id, exc)

        logger.info("Re-embedded %d/%d documents.", rebuilt, len(doc_ids))
        return rebuilt

    def _reembed_single(
        self,
        doc_id: str,
        collection: str,
        embedder: Any,
    ) -> None:
        """Re-embed a single document."""
        if collection == "market" and self._weaviate_url:
            self._reembed_weaviate(doc_id, embedder)
        elif collection == "personal_finance" and self._pgvector_url:
            self._reembed_pgvector(doc_id, embedder)

    def _reembed_weaviate(self, doc_id: str, embedder: Any) -> None:
        """Re-embed a document in Weaviate."""
        import weaviate

        client = weaviate.connect_to_url(self._weaviate_url)
        try:
            col = client.collections.get("MarketDocuments")
            obj = col.query.fetch_object_by_id(doc_id)
            if obj is None:
                return

            content = obj.properties.get("content", "")
            if not content:
                return

            new_embedding = embedder.embed_text(content)
            col.data.update(
                uuid=obj.uuid,
                properties={
                    "content": content,
                    "vector": new_embedding,
                },
            )
            logger.debug("Re-embedded Weaviate document %s.", doc_id)
        finally:
            client.close()

    def _reembed_pgvector(self, doc_id: str, embedder: Any) -> None:
        """Re-embed a document in pgvector."""
        from sqlalchemy import create_engine, text

        engine = create_engine(self._pgvector_url)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT content FROM personal_finance_docs WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )
            row = result.fetchone()
            if row is None:
                return

            content = row[0]
            if not content:
                return

            new_embedding = embedder.embed_text(content)
            conn.execute(
                text(
                    "UPDATE personal_finance_docs SET embedding = :embedding WHERE doc_id = :doc_id"
                ),
                {"doc_id": doc_id, "embedding": str(new_embedding)},
            )
            conn.commit()
            logger.debug("Re-embedded pgvector document %s.", doc_id)
