"""
Pipeline scheduler using Prefect 3.x.

Defines Prefect flows for each data pipeline, configures cron-like
schedules, and handles error notifications.

Schedules:
  - Market data:  every 15 minutes during NSE trading hours (Mon–Fri 09:15–15:30 IST)
  - News ingestion: every 30 minutes during market hours, hourly otherwise
  - RAG updates: daily at 02:00 IST
  - Staleness detection: weekly on Sunday at 03:00 IST
  - Embedding rebuild: weekly on Sunday at 04:00 IST

Usage (development)::

    python -m pipelines.scheduled.scheduler

Usage (production / Prefect server)::

    prefect deploy pipelines/scheduled/scheduler.py:market_data_flow
    prefect deploy pipelines/scheduled/scheduler.py:news_ingestion_flow
    prefect deploy pipelines/scheduled/scheduler.py:rag_update_flow
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

from prefect import flow, task, get_run_logger
from prefect.tasks import exponential_backoff

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Prefect API URL (set via env var or in settings)
PREFECT_API_URL = os.environ.get(
    "PREFECT_API_URL", "http://localhost:4200"
)

# Notification settings
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "")

# DB URLs
TIMESCALEDB_URL = os.environ.get(
    "TIMESCALEDB_URL",
    "postgresql://postgres:postgres@localhost:5432/financial_brain",
)
WEAVIATE_URL = os.environ.get(
    "WEAVIATE_URL", "http://localhost:8080"
)
PGVECTOR_URL = os.environ.get(
    "PGVECTOR_URL",
    "postgresql://postgres:postgres@localhost:5432/financial_brain_rag",
)

# NewsAPI key
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------

def send_error_notification(
    flow_name: str,
    error_message: str,
    run_id: Optional[str] = None,
) -> None:
    """Send an email notification on pipeline failure.

    Falls back silently if SMTP is not configured.
    """
    if not SMTP_USER or not NOTIFICATION_EMAIL:
        logger.warning(
            "SMTP not configured. Skipping error notification for flow '%s'.",
            flow_name,
        )
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ALERT] Pipeline Failure: {flow_name}"
        msg["From"] = SMTP_USER
        msg["To"] = NOTIFICATION_EMAIL

        timestamp = datetime.now(timezone.utc).isoformat()
        run_info = f"\nRun ID: {run_id}" if run_id else ""
        body = (
            f"A pipeline failure occurred:\n\n"
            f"Flow: {flow_name}\n"
            f"Timestamp (UTC): {timestamp}{run_info}\n\n"
            f"Error:\n{error_message}\n\n"
            f"Check the Prefect UI for details: {PREFECT_API_URL}"
        )
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, NOTIFICATION_EMAIL, msg.as_string())

        logger.info("Error notification sent for flow '%s'.", flow_name)
    except Exception as exc:
        logger.error("Failed to send error notification: %s", exc)


# ---------------------------------------------------------------------------
# Market Data Flow
# ---------------------------------------------------------------------------

@task(
    name="fetch_market_data",
    retries=3,
    retry_delay=exponential_backoff(backoff_factor=10, max_delay=300),
)
def fetch_market_data_task(
    period: str = "1d",
    interval: str = "15m",
) -> dict[str, int]:
    """Fetch latest market data for all tracked Indian symbols."""
    log = get_run_logger()
    log.info("Starting market data fetch (period=%s, interval=%s).", period, interval)

    from pipelines.market_data.ingestion import MarketDataIngestionPipeline

    pipeline = MarketDataIngestionPipeline(
        db_url=TIMESCALEDB_URL,
        adjust_prices=True,
    )
    rows = pipeline.scheduled_ingest_trading_hours(period=period)

    log.info("Market data fetch complete. Rows ingested: %d", rows)
    return {"rows_ingested": rows}


@task(name="daily_market_snapshot")
def daily_market_snapshot_task() -> dict[str, int]:
    """Take a daily snapshot of all tracked symbols at end of day."""
    log = get_run_logger()
    log.info("Taking daily market snapshot.")

    from pipelines.market_data.ingestion import MarketDataIngestionPipeline

    pipeline = MarketDataIngestionPipeline(
        db_url=TIMESCALEDB_URL,
        adjust_prices=True,
    )

    data = pipeline.fetch_all_indian_symbols(period="1d", interval="1d", parallel=True)
    rows = pipeline.bulk_ingest_to_timescaledb(data) if pipeline._engine else sum(len(df) for df in data.values())

    log.info("Daily snapshot complete. Rows: %d", rows)
    return {"rows_ingested": rows}


@flow(
    name="Market Data Ingestion",
    description="Fetches and stores market data for Indian equities during trading hours.",
    log_prints=True,
)
def market_data_flow(
    period: str = "1d",
    interval: str = "15m",
) -> dict[str, Any]:
    """Main market data ingestion flow.

    Designed to run every 15 minutes during NSE trading hours.
    """
    result = fetch_market_data_task(period=period, interval=interval)
    return result


@flow(
    name="Daily Market Snapshot",
    description="End-of-day market data snapshot for all tracked symbols.",
    log_prints=True,
)
def daily_snapshot_flow() -> dict[str, Any]:
    """Daily EOD snapshot flow. Run at 16:00 IST."""
    return daily_market_snapshot_task()


# ---------------------------------------------------------------------------
# News Ingestion Flow
# ---------------------------------------------------------------------------

@task(
    name="fetch_newsapi",
    retries=2,
    retry_delay=exponential_backoff(backoff_factor=15, max_delay=120),
)
def fetch_newsapi_task(queries: list[str]) -> list[dict]:
    """Fetch articles from NewsAPI."""
    from pipelines.news_ingestion.fetcher import NewsIngestionPipeline

    pipeline = NewsIngestionPipeline(
        newsapi_key=NEWSAPI_KEY,
        db_url=TIMESCALEDB_URL,
        weaviate_url=WEAVIATE_URL,
    )
    all_articles = []
    for query in queries:
        articles = pipeline.fetch_from_newsapi(query, page_size=20)
        all_articles.extend(articles)
    return all_articles


@task(
    name="fetch_rss_feeds",
    retries=2,
    retry_delay=exponential_backoff(backoff_factor=10, max_delay=60),
)
def fetch_rss_task(feeds: list[str]) -> list[dict]:
    """Fetch articles from RSS feeds."""
    from pipelines.news_ingestion.fetcher import NewsIngestionPipeline

    pipeline = NewsIngestionPipeline(
        newsapi_key=NEWSAPI_KEY,
        db_url=TIMESCALEDB_URL,
        weaviate_url=WEAVIATE_URL,
    )
    return pipeline.fetch_from_rss(feeds=feeds, max_articles_per_feed=30)


@task(name="analyze_and_store_news")
def analyze_and_store_news_task(articles: list[dict]) -> dict[str, int]:
    """Run sentiment analysis and store articles."""
    from pipelines.news_ingestion.fetcher import NewsIngestionPipeline

    pipeline = NewsIngestionPipeline(
        newsapi_key=NEWSAPI_KEY,
        db_url=TIMESCALEDB_URL,
        weaviate_url=WEAVIATE_URL,
    )
    articles = pipeline.analyze_sentiment_batch(articles)
    articles = pipeline._deduplicate_articles(articles)
    stored_db = pipeline.store_articles_db(articles)
    stored_wv = pipeline.store_articles_weaviate(articles)
    return {"analyzed": len(articles), "stored_db": stored_db, "stored_weaviate": stored_wv}


@flow(
    name="News Ingestion",
    description="Fetches financial news from NewsAPI and RSS feeds, runs sentiment analysis.",
    log_prints=True,
)
def news_ingestion_flow(
    queries: Optional[list[str]] = None,
    rss_feeds: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Main news ingestion flow.

    Runs every 30 minutes during market hours, hourly otherwise.
    """
    default_queries = [
        "Nifty 50 Sensex today",
        "RBI monetary policy",
        "Indian stock market earnings",
        "SEBI regulations",
        "India banking sector",
        "Indian IT companies",
        "Reliance TCS HDFC Bank",
    ]

    target_queries = queries or default_queries
    target_feeds = rss_feeds  # Will use defaults if None

    # Fetch from both sources
    newsapi_articles = fetch_newsapi_task(target_queries)
    rss_articles = fetch_rss_task(target_feeds) if target_feeds else []

    # Combine and process
    all_articles = newsapi_articles + rss_articles

    if not all_articles:
        return {"newsapi": 0, "rss": len(rss_articles), "total": 0, "analyzed": 0}

    result = analyze_and_store_news_task(all_articles)
    result["newsapi"] = len(newsapi_articles)
    result["rss"] = len(rss_articles)
    result["total"] = len(all_articles)

    return result


# ---------------------------------------------------------------------------
# RAG Update Flow
# ---------------------------------------------------------------------------

@task(
    name="update_market_rag",
    retries=2,
    retry_delay=exponential_backoff(backoff_factor=30, max_delay=600),
)
def update_market_rag_task() -> dict[str, int]:
    """Update market RAG with new research reports and filings."""
    from pipelines.rag_updates.updater import RAGUpdatePipeline

    with RAGUpdatePipeline(weaviate_url=WEAVIATE_URL) as updater:
        return updater.update_market_rag()


@task(
    name="update_personal_finance_rag",
    retries=2,
    retry_delay=exponential_backoff(backoff_factor=30, max_delay=600),
)
def update_personal_finance_rag_task() -> dict[str, int]:
    """Update personal finance RAG with new tax rules and regulations."""
    from pipelines.rag_updates.updater import RAGUpdatePipeline

    with RAGUpdatePipeline(pgvector_url=PGVECTOR_URL) as updater:
        return updater.update_personal_finance_rag()


@task(name="detect_stale_documents")
def detect_stale_task(collection: str = "market", max_age_days: int = 30) -> list[dict]:
    """Detect stale documents in a collection."""
    from pipelines.rag_updates.updater import RAGUpdatePipeline

    with RAGUpdatePipeline(weaviate_url=WEAVIATE_URL, pgvector_url=PGVECTOR_URL) as updater:
        return updater.detect_stale_documents(collection=collection, max_age_days=max_age_days)


@task(name="rebuild_embeddings")
def rebuild_embeddings_task(doc_ids: list[str], collection: str = "market") -> int:
    """Rebuild embeddings for specified documents."""
    from pipelines.rag_updates.updater import RAGUpdatePipeline

    with RAGUpdatePipeline(weaviate_url=WEAVIATE_URL, pgvector_url=PGVECTOR_URL) as updater:
        return updater.rebuild_embeddings(doc_ids=doc_ids, collection=collection)


@flow(
    name="RAG Update",
    description="Daily update of RAG documents for market and personal finance.",
    log_prints=True,
)
def rag_update_flow() -> dict[str, Any]:
    """Main RAG update flow.  Run daily at 02:00 IST."""
    # Update market RAG
    market_stats = update_market_rag_task()

    # Update personal finance RAG
    pf_stats = update_personal_finance_rag_task()

    return {
        "market": market_stats,
        "personal_finance": pf_stats,
    }


@flow(
    name="Staleness Detection & Rebuild",
    description="Weekly staleness detection and embedding rebuild.",
    log_prints=True,
)
def staleness_rebuild_flow(
    max_age_days: int = 30,
) -> dict[str, Any]:
    """Weekly flow to detect stale documents and rebuild their embeddings."""
    # Detect stale market documents
    stale_market = detect_stale_task(collection="market", max_age_days=max_age_days)
    market_rebuilt = 0
    if stale_market:
        market_ids = [d["doc_id"] for d in stale_market]
        market_rebuilt = rebuild_embeddings_task(doc_ids=market_ids, collection="market")

    # Detect stale personal finance documents
    stale_pf = detect_stale_task(collection="personal_finance", max_age_days=max_age_days)
    pf_rebuilt = 0
    if stale_pf:
        pf_ids = [d["doc_id"] for d in stale_pf]
        pf_rebuilt = rebuild_embeddings_task(doc_ids=pf_ids, collection="personal_finance")

    return {
        "stale_market": len(stale_market),
        "rebuilt_market": market_rebuilt,
        "stale_personal_finance": len(stale_pf),
        "rebuilt_personal_finance": pf_rebuilt,
    }


# ---------------------------------------------------------------------------
# Master Orchestrator Flow
# ---------------------------------------------------------------------------

@flow(
    name="Master Pipeline Orchestrator",
    description="Orchestrates all data pipelines with dependency management.",
    log_prints=True,
)
def master_orchestrator_flow() -> dict[str, Any]:
    """Master flow that orchestrates all sub-flows.

    Dependencies:
    1. Market data ingestion (parallel with news)
    2. News ingestion (parallel with market data)
    3. RAG update (depends on market data completion)
    4. Staleness detection (depends on RAG update)
    """
    import asyncio

    # Run market data and news ingestion in parallel
    market_future = market_data_flow()
    news_future = news_ingestion_flow()

    # Wait for both to complete
    market_result = market_future
    news_result = news_future

    # Then run RAG update
    rag_result = rag_update_flow()

    return {
        "market_data": market_result,
        "news": news_result,
        "rag_update": rag_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Schedule Builder & CLI Entry Point
# ---------------------------------------------------------------------------

class PipelineScheduler:
    """Schedules all data pipelines using Prefect.

    Provides methods to:
    - Deploy all flows to a Prefect server
    - Configure cron-like schedules
    - Set up flow dependencies

    Parameters
    ----------
    prefect_api_url : str or None
        Prefect server API URL.  Defaults to the ``PREFECT_API_URL``
        environment variable.
    """

    def __init__(
        self,
        prefect_api_url: Optional[str] = None,
    ) -> None:
        self._api_url = prefect_api_url or PREFECT_API_URL

    # NSE trading hours in IST: Mon–Fri 09:15 – 15:30
    # Cron format: minute hour day-of-month month day-of-week
    # We use multiple 15-min intervals during trading hours
    _MARKET_CRON = "*/15 9-15 * * 1-5"  # Every 15 min, 9am-3pm, Mon-Fri
    _SNAPSHOT_CRON = "0 16 * * 1-5"  # 4:00 PM IST, Mon-Fri
    _NEWS_CRON = "*/30 6-23 * * *"  # Every 30 min, 6am-11pm, all days
    _RAG_CRON = "0 2 * * *"  # 2:00 AM IST daily
    _STALENESS_CRON = "0 3 * * 0"  # 3:00 AM IST, every Sunday

    def get_all_flows(self) -> dict[str, Any]:
        """Return all flow objects with their configurations.

        Returns
        -------
        dict
            Mapping of flow name -> ``{flow, cron, description, tags}``.
        """
        return {
            "Market Data Ingestion": {
                "flow": market_data_flow,
                "cron": self._MARKET_CRON,
                "description": "Fetch market data every 15 min during NSE trading hours",
                "tags": ["market-data", "india", "scheduled"],
            },
            "Daily Market Snapshot": {
                "flow": daily_snapshot_flow,
                "cron": self._SNAPSHOT_CRON,
                "description": "End-of-day daily snapshot of all Indian market symbols",
                "tags": ["market-data", "india", "daily"],
            },
            "News Ingestion": {
                "flow": news_ingestion_flow,
                "cron": self._NEWS_CRON,
                "description": "Fetch financial news every 30 minutes",
                "tags": ["news", "india", "scheduled"],
            },
            "RAG Update": {
                "flow": rag_update_flow,
                "cron": self._RAG_CRON,
                "description": "Daily RAG document update for market and personal finance",
                "tags": ["rag", "daily"],
            },
            "Staleness Detection & Rebuild": {
                "flow": staleness_rebuild_flow,
                "cron": self._STALENESS_CRON,
                "description": "Weekly staleness detection and embedding rebuild",
                "tags": ["rag", "maintenance", "weekly"],
            },
        }

    def deploy_all(self) -> list[dict[str, Any]]:
        """Deploy all flows to the Prefect server with schedules.

        Returns
        -------
        list[dict]
            Deployment results.
        """
        results = []
        for name, config in self.get_all_flows().items():
            try:
                deployment = self._deploy_flow(name, config)
                results.append({
                    "flow_name": name,
                    "status": "deployed",
                    "deployment_id": getattr(deployment, "id", "unknown"),
                    "cron": config["cron"],
                })
                logger.info("Deployed flow '%s' with schedule '%s'.", name, config["cron"])
            except Exception as exc:
                results.append({
                    "flow_name": name,
                    "status": "failed",
                    "error": str(exc),
                })
                logger.error("Failed to deploy flow '%s': %s", name, exc)

        return results

    def _deploy_flow(self, name: str, config: dict[str, Any]) -> Any:
        """Deploy a single flow with its schedule configuration."""
        flow_obj = config["flow"]

        from prefect.deployments import Deployment
        from prefect.schedules import CronSchedule

        deployment = Deployment.build_from_flow(
            flow=flow_obj,
            name=name,
            schedule=CronSchedule(cron=config["cron"], timezone="Asia/Kolkata"),
            tags=config.get("tags", []),
            description=config.get("description", ""),
        )

        return deployment.apply()

    def print_schedule_summary(self) -> None:
        """Print a human-readable summary of all pipeline schedules."""
        print("\n" + "=" * 70)
        print("  AI Financial Brain — Pipeline Schedule Summary")
        print("=" * 70)
        for name, config in self.get_all_flows().items():
            print(f"\n  📊 {name}")
            print(f"     Schedule: {config['cron']}")
            print(f"     Tags: {', '.join(config.get('tags', []))}")
            print(f"     Description: {config.get('description', '')}")
        print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scheduler = PipelineScheduler()
    scheduler.print_schedule_summary()

    # Try to deploy if Prefect server is running
    try:
        results = scheduler.deploy_all()
        print("\nDeployment Results:")
        for r in results:
            status = "✅" if r["status"] == "deployed" else "❌"
            print(f"  {status} {r['flow_name']}: {r['status']}")
            if r["status"] == "failed":
                print(f"     Error: {r.get('error', 'Unknown')}")
    except Exception as exc:
        print(f"\n⚠️  Could not deploy to Prefect server: {exc}")
        print("   Make sure the Prefect server is running: prefect server start")
        print("   Or set PREFECT_API_URL environment variable.")
