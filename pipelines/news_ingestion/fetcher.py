"""
Financial news ingestion pipeline.

Aggregates articles from:
  - NewsAPI.org (requires API key)
  - RSS feeds: Moneycontrol, Economic Times, LiveMint, Reuters India

Applies FinBERT sentiment analysis on ingestion and stores results
in the database + Weaviate for RAG retrieval.

Typical usage::

    pipeline = NewsIngestionPipeline(newsapi_key="...")
    articles = pipeline.fetch_from_newsapi("Reliance Industries Q3 results")
    pipeline.analyze_and_store(articles)
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS feed URLs for Indian financial news
# ---------------------------------------------------------------------------

DEFAULT_RSS_FEEDS: list[str] = [
    # Moneycontrol
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/business.xml",
    # Economic Times
    "https://economictimes.indiatimes.com/rssMarkets.xml",
    "https://economictimes.indiatimes.com/rssfeeds/13358312.cms",
    "https://economictimes.indiatimes.com/rssfeeds/1287361747.cms",
    # LiveMint
    "https://www.livemint.com/rss/market",
    "https://www.livemint.com/rss/companies",
    "https://www.livemint.com/rss/economy",
    # Reuters India
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/INtopNews",
    "https://feeds.reuters.com/reuters/INmarketsNews",
]

# ---------------------------------------------------------------------------
# NSE ticker extraction patterns
# ---------------------------------------------------------------------------

_NSE_TICKER_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(RELIANCE|TCS|INFY|HDFCBANK|ICICIBANK|SBIN|BHARTIARTL|ITC|KOTAKBANK|LT|"
                r"AXISBANK|BAJFINANCE|MARUTI|SUNPHARMA|TATAMOTORS|WIPRO|ULTRACEMCO|TITAN|"
                r"NESTLEIND|NTPC|POWERGRID|ONGC|JSWSTEEL|ADANIENT|TATASTEEL|HCLTECH|"
                r"COALINDIA|BAJAJFINSV|INDUSINDBK|ASIANPAINT|DRREDDY|TECHM|CIPLA|"
                r"EICHERMOT|GRASIM|HEROMOTOCO|DIVISLAB|BRITANNIA|HINDALCO|UPL|BPCL|"
                r"HDFCLIFE|SBILIFE|M_M|SHREECEM)\b", re.IGNORECASE),
]


def _extract_tickers(text: str) -> list[str]:
    """Extract NSE ticker symbols mentioned in text."""
    tickers: set[str] = set()
    for pattern in _NSE_TICKER_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            tickers.add(m.upper())
    return sorted(tickers)


def _clean_html_text(raw_html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _generate_article_id(url: str, title: str) -> str:
    """Deterministic ID from URL + title."""
    raw = f"{url}:{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class NewsIngestionPipeline:
    """Fetches financial news from multiple sources.

    Parameters
    ----------
    newsapi_key : str or None
        NewsAPI.org API key.  If ``None``, NewsAPI fetches are disabled.
    db_url : str or None
        PostgreSQL connection string.  If ``None``, DB storage is disabled.
    weaviate_url : str or None
        Weaviate URL for storing news vectors.  If ``None``, Weaviate storage
        is disabled.
    sentiment_analyzer : Any or None
        Pre-initialised :class:`FinBERTSentimentAnalyzer` instance.
        If ``None``, a new instance will be created on first use.
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        db_url: Optional[str] = None,
        weaviate_url: Optional[str] = None,
        sentiment_analyzer: Any = None,
    ) -> None:
        self._newsapi_key = newsapi_key
        self._db_url = db_url
        self._weaviate_url = weaviate_url
        self._sentiment_analyzer = sentiment_analyzer
        self._db_engine = None
        self._http_client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "AI-Financial-Brain-NewsBot/1.0"},
        )

        if db_url:
            self._init_db(db_url)

    def close(self) -> None:
        """Close HTTP client and DB connections."""
        self._http_client.close()
        if self._db_engine:
            self._db_engine.dispose()

    def __enter__(self) -> "NewsIngestionPipeline":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _init_db(self, db_url: str) -> None:
        """Create SQLAlchemy engine and ensure the news table exists."""
        try:
            from sqlalchemy import create_engine

            self._db_engine = create_engine(
                db_url, pool_size=5, max_overflow=10, pool_recycle=3600
            )
            self._ensure_news_table()
            logger.info("News DB engine initialised.")
        except Exception as exc:
            logger.error("Failed to init news DB: %s", exc)
            self._db_engine = None

    def _ensure_news_table(self) -> None:
        """Create the ``news_articles`` table if it doesn't exist."""
        if self._db_engine is None:
            return
        sql = """
        CREATE TABLE IF NOT EXISTS news_articles (
            article_id   TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            source       TEXT,
            url          TEXT UNIQUE,
            published_at TIMESTAMPTZ,
            content      TEXT,
            summary      TEXT,
            tickers      TEXT[],
            sentiment    TEXT,
            sentiment_score FLOAT,
            source_type  TEXT DEFAULT 'rss',
            ingested_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at DESC);
        CREATE INDEX IF NOT EXISTS idx_news_tickers ON news_articles USING GIN(tickers);
        CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles(sentiment);
        """
        try:
            with self._db_engine.begin() as conn:
                conn.execute(sql)
            logger.info("news_articles table ready.")
        except Exception as exc:
            logger.error("Failed to create news_articles table: %s", exc)

    # ------------------------------------------------------------------
    # NewsAPI fetching
    # ------------------------------------------------------------------

    def fetch_from_newsapi(
        self,
        query: str,
        language: str = "en",
        page_size: int = 100,
        sort_by: str = "publishedAt",
        domains: Optional[list[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Fetch articles from NewsAPI.org.

        Parameters
        ----------
        query : str
            Search query (e.g. ``"Reliance Industries Q3 results"``).
        language : str
            ISO language code (default ``"en"``).
        page_size : int
            Number of results per page (max 100).
        sort_by : str
            ``"relevancy"``, ``"popularity"``, or ``"publishedAt"``.
        domains : list[str] or None
            Restrict to specific domains.
        from_date : str or None
            Start date (``YYYY-MM-DD`` or ISO 8601).
        to_date : str or None
            End date (``YYYY-MM-DD`` or ISO 8601).

        Returns
        -------
        list[dict]
            Normalised article dicts with keys:
            ``article_id, title, source, url, published_at, content,
            summary, tickers``.
        """
        if not self._newsapi_key:
            logger.warning("NewsAPI key not set. Skipping NewsAPI fetch.")
            return []

        from newsapi import NewsApiClient

        client = NewsApiClient(api_key=self._newsapi_key)

        params: dict[str, Any] = {
            "q": query,
            "language": language,
            "page_size": min(page_size, 100),
            "sort_by": sort_by,
        }
        if domains:
            params["domains"] = ",".join(domains)
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            response = client.get_everything(**params)
            if response.get("status") != "ok":
                logger.error(
                    "NewsAPI error: %s", response.get("message", "Unknown error")
                )
                return []

            articles: list[dict[str, Any]] = []
            for item in response.get("articles", []):
                article = self._normalise_newsapi_article(item)
                if article:
                    articles.append(article)

            logger.info(
                "NewsAPI returned %d articles for query '%s'.",
                len(articles),
                query,
            )
            return articles

        except Exception as exc:
            logger.error("NewsAPI fetch failed: %s", exc)
            return []

    def _normalise_newsapi_article(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Normalise a raw NewsAPI article dict."""
        title = (raw.get("title") or "").strip()
        url = raw.get("url") or ""
        content = (raw.get("content") or "").strip()
        description = (raw.get("description") or "").strip()

        if not title and not content:
            return None

        # NewsAPI truncates content with "[+NNN chars]" suffix
        content = re.sub(r"\[\+\d+\s*chars\]", "", content).strip()

        article_id = _generate_article_id(url, title)
        source_name = raw.get("source", {}).get("name", "unknown") if isinstance(raw.get("source"), dict) else "unknown"
        published = raw.get("publishedAt", "")

        # Extract tickers
        full_text = f"{title} {description} {content}"
        tickers = _extract_tickers(full_text)

        return {
            "article_id": article_id,
            "title": title,
            "source": source_name,
            "url": url,
            "published_at": published,
            "content": content,
            "summary": description,
            "tickers": tickers,
            "source_type": "newsapi",
        }

    # ------------------------------------------------------------------
    # RSS feed fetching
    # ------------------------------------------------------------------

    def fetch_from_rss(
        self,
        feeds: Optional[list[str]] = None,
        max_articles_per_feed: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch and parse articles from RSS/Atom feeds.

        Parameters
        ----------
        feeds : list[str] or None
            List of RSS feed URLs.  Defaults to :data:`DEFAULT_RSS_FEEDS`.
        max_articles_per_feed : int
            Maximum number of articles to parse per feed.

        Returns
        -------
        list[dict]
            Normalised article dicts.
        """
        target_feeds = feeds or DEFAULT_RSS_FEEDS
        all_articles: list[dict[str, Any]] = []

        for feed_url in target_feeds:
            try:
                articles = self._parse_rss_feed(feed_url, max_articles_per_feed)
                all_articles.extend(articles)
                logger.info(
                    "RSS feed '%s': %d articles parsed.",
                    feed_url,
                    len(articles),
                )
            except Exception as exc:
                logger.error("Failed to parse RSS feed '%s': %s", feed_url, exc)

        logger.info("Total articles from RSS: %d", len(all_articles))
        return all_articles

    def _parse_rss_feed(
        self, feed_url: str, max_articles: int
    ) -> list[dict[str, Any]]:
        """Parse a single RSS/Atom feed and return normalised articles."""
        response = self._http_client.get(feed_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "xml")
        articles: list[dict[str, Any]] = []

        # Determine feed type (RSS 2.0 vs Atom)
        entries = soup.find_all("item")
        feed_type = "rss"
        if not entries:
            entries = soup.find_all("entry")
            feed_type = "atom"

        source_domain = urlparse(feed_url).netloc.replace("www.", "")

        for entry in entries[:max_articles]:
            try:
                article = self._parse_rss_entry(entry, feed_type, source_domain)
                if article:
                    articles.append(article)
            except Exception as exc:
                logger.debug("Failed to parse RSS entry: %s", exc)
                continue

        return articles

    def _parse_rss_entry(
        self, entry: Any, feed_type: str, source_domain: str
    ) -> dict[str, Any] | None:
        """Parse a single RSS/Atom <item> or <entry> element."""
        # Title
        title_tag = entry.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # URL / Link
        if feed_type == "atom":
            link_tag = entry.find("link")
            url = link_tag.get("href", "") if link_tag else ""
        else:
            link_tag = entry.find("link")
            url = link_tag.get_text(strip=True) if link_tag else ""

        if not url:
            return None

        # Published date
        if feed_type == "atom":
            pub_tag = entry.find("published") or entry.find("updated")
        else:
            pub_tag = entry.find("pubDate") or entry.find("dc:date")
        published = pub_tag.get_text(strip=True) if pub_tag else ""

        # Content / Description
        content_tag = entry.find("content:encoded") or entry.find("content")
        desc_tag = entry.find("description")
        summary_text = ""
        full_content = ""

        if content_tag:
            raw_content = content_tag.get_text() if content_tag.string else str(content_tag)
            full_content = _clean_html_text(raw_content)
            summary_text = full_content[:500]

        if desc_tag and not full_content:
            raw_desc = desc_tag.get_text() if desc_tag.string else str(desc_tag)
            summary_text = _clean_html_text(raw_desc)[:500]
            full_content = summary_text

        if not title and not full_content:
            return None

        article_id = _generate_article_id(url, title)
        tickers = _extract_tickers(f"{title} {full_content}")

        # Map source domain to friendly name
        source_map = {
            "moneycontrol.com": "Moneycontrol",
            "economictimes.indiatimes.com": "Economic Times",
            "livemint.com": "LiveMint",
            "feeds.reuters.com": "Reuters India",
        }
        source_name = source_map.get(source_domain, source_domain)

        return {
            "article_id": article_id,
            "title": title,
            "source": source_name,
            "url": url,
            "published_at": published,
            "content": full_content,
            "summary": summary_text,
            "tickers": tickers,
            "source_type": "rss",
        }

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    def _get_sentiment_analyzer(self) -> Any:
        """Lazy-initialise the FinBERT sentiment analyzer."""
        if self._sentiment_analyzer is not None:
            return self._sentiment_analyzer

        try:
            from backend.ml_models.sentiment.finbert import FinBERTSentimentAnalyzer

            self._sentiment_analyzer = FinBERTSentimentAnalyzer()
            logger.info("FinBERT sentiment analyzer initialised.")
            return self._sentiment_analyzer
        except Exception as exc:
            logger.error(
                "Failed to initialise FinBERT: %s. "
                "Sentiment analysis will use simple heuristic.",
                exc,
            )
            return None

    def analyze_sentiment_batch(
        self, articles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run FinBERT sentiment analysis on a batch of articles.

        Modifies each article dict in-place, adding ``sentiment`` and
        ``sentiment_score`` fields.

        Parameters
        ----------
        articles : list[dict]
            Normalised article dicts.

        Returns
        -------
        list[dict]
            The same list, enriched with sentiment data.
        """
        analyzer = self._get_sentiment_analyzer()

        if analyzer is None:
            # Fallback: simple keyword-based heuristic
            return self._heuristic_sentiment(articles)

        # Build text for each article (title + summary for speed)
        texts = []
        for art in articles:
            combined = f"{art.get('title', '')} {art.get('summary', '')}"
            texts.append(combined.strip())

        if not texts:
            return articles

        # Batch analyze
        try:
            results = analyzer.analyze_batch(texts)
            for article, result in zip(articles, results):
                article["sentiment"] = result["label"]
                article["sentiment_score"] = result["normalized_score"]
        except Exception as exc:
            logger.error("FinBERT batch analysis failed: %s", exc)
            articles = self._heuristic_sentiment(articles)

        return articles

    def _heuristic_sentiment(
        self, articles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fallback sentiment using simple keyword matching."""
        bullish_words = {
            "surge", "jump", "rally", "gain", "profit", "growth", "beat",
            "upgrade", "outperform", "strong", "record", "bullish", "soar",
            "boost", "improve", "exceed", "positive", "optimistic",
        }
        bearish_words = {
            "drop", "fall", "decline", "loss", "crash", "slump", "miss",
            "downgrade", "underperform", "weak", "bearish", "plunge",
            "cut", "reduce", "warning", "negative", "risk", "debt",
        }

        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')}".lower()
            words = set(re.findall(r"\b\w+\b", text))
            bull_count = len(words & bullish_words)
            bear_count = len(words & bearish_words)
            total = bull_count + bear_count

            if total == 0:
                article["sentiment"] = "neutral"
                article["sentiment_score"] = 0.5
            elif bull_count > bear_count:
                article["sentiment"] = "positive"
                article["sentiment_score"] = min(1.0, 0.5 + (bull_count - bear_count) * 0.1)
            elif bear_count > bull_count:
                article["sentiment"] = "negative"
                article["sentiment_score"] = max(0.0, 0.5 - (bear_count - bull_count) * 0.1)
            else:
                article["sentiment"] = "neutral"
                article["sentiment_score"] = 0.5

        return articles

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_articles_db(self, articles: list[dict[str, Any]]) -> int:
        """Insert articles into the PostgreSQL ``news_articles`` table.

        Uses upsert to handle re-ingestion gracefully.

        Returns
        -------
        int
            Number of articles stored.
        """
        if self._db_engine is None or not articles:
            return 0

        stored = 0
        for article in articles:
            try:
                with self._db_engine.begin() as conn:
                    conn.execute(
                        """
                        INSERT INTO news_articles
                            (article_id, title, source, url, published_at,
                             content, summary, tickers, sentiment,
                             sentiment_score, source_type)
                        VALUES
                            (:article_id, :title, :source, :url, :published_at,
                             :content, :summary, :tickers, :sentiment,
                             :sentiment_score, :source_type)
                        ON CONFLICT (article_id) DO UPDATE SET
                            sentiment = EXCLUDED.sentiment,
                            sentiment_score = EXCLUDED.sentiment_score,
                            content = COALESCE(EXCLUDED.content, news_articles.content),
                            summary = COALESCE(EXCLUDED.summary, news_articles.summary);
                        """,
                        {
                            "article_id": article["article_id"],
                            "title": article.get("title", "")[:500],
                            "source": article.get("source", ""),
                            "url": article.get("url", "")[:1000],
                            "published_at": article.get("published_at"),
                            "content": article.get("content", ""),
                            "summary": article.get("summary", ""),
                            "tickers": article.get("tickers", []),
                            "sentiment": article.get("sentiment", "neutral"),
                            "sentiment_score": float(article.get("sentiment_score", 0.5)),
                            "source_type": article.get("source_type", "rss"),
                        },
                    )
                stored += 1
            except Exception as exc:
                logger.error(
                    "Failed to store article '%s': %s",
                    article.get("article_id", "?"),
                    exc,
                )

        logger.info("Stored %d/%d articles in DB.", stored, len(articles))
        return stored

    def store_articles_weaviate(self, articles: list[dict[str, Any]]) -> int:
        """Store articles in Weaviate for RAG retrieval.

        Returns
        -------
        int
            Number of articles stored.
        """
        if not self._weaviate_url or not articles:
            return 0

        try:
            import weaviate

            client = weaviate.connect_to_url(self._weaviate_url)
            try:
                collection = client.collections.get("NewsDocuments")
                stored = 0
                with collection.batch.dynamic() as batch:
                    for article in articles:
                        content = article.get("content") or article.get("summary", "")
                        if not content.strip():
                            continue
                        batch.add_object(
                            properties={
                                "title": article.get("title", ""),
                                "source": article.get("source", ""),
                                "publishDate": article.get("published_at", ""),
                                "text": content[:5000],
                                "tickers": article.get("tickers", []),
                                "sentiment": float(article.get("sentiment_score", 0.5)),
                                "summary": article.get("summary", ""),
                            },
                        )
                        stored += 1
                logger.info("Stored %d articles in Weaviate.", stored)
                return stored
            finally:
                client.close()
        except Exception as exc:
            logger.error("Weaviate storage failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Combined pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        queries: Optional[list[str]] = None,
        rss_feeds: Optional[list[str]] = None,
    ) -> dict[str, int]:
        """Run the full ingestion pipeline: fetch → analyze → store.

        Parameters
        ----------
        queries : list[str] or None
            NewsAPI search queries.  Defaults to a set of Indian market queries.
        rss_feeds : list[str] or None
            RSS feeds to scrape.  Defaults to :data:`DEFAULT_RSS_FEEDS`.

        Returns
        -------
        dict
            Counts: ``{newsapi: int, rss: int, analyzed: int,
            stored_db: int, stored_weaviate: int}``.
        """
        all_articles: list[dict[str, Any]] = []

        # 1. Fetch from NewsAPI
        newsapi_count = 0
        if self._newsapi_key:
            default_queries = [
                "Nifty 50 India stock market",
                "Sensex Bombay Stock Exchange",
                "RBI monetary policy India",
                "SEBI regulations India",
                "India GDP growth",
                "Indian IT sector earnings",
                "Indian banking sector",
                "Reliance Industries",
                "TCS Infosys Wipro",
                "HDFC Bank ICICI Bank",
            ]
            target_queries = queries or default_queries
            for query in target_queries:
                articles = self.fetch_from_newsapi(query, page_size=20)
                all_articles.extend(articles)
                newsapi_count += len(articles)
            # Deduplicate
            all_articles = self._deduplicate_articles(all_articles)

        # 2. Fetch from RSS
        rss_count = 0
        rss_articles = self.fetch_from_rss(feeds=rss_feeds, max_articles_per_feed=30)
        all_articles.extend(rss_articles)
        all_articles = self._deduplicate_articles(all_articles)
        rss_count = len(rss_articles)

        # 3. Sentiment analysis
        all_articles = self.analyze_sentiment_batch(all_articles)

        # 4. Store in DB
        stored_db = self.store_articles_db(all_articles)

        # 5. Store in Weaviate
        stored_wv = self.store_articles_weaviate(all_articles)

        logger.info(
            "Full pipeline: newsapi=%d, rss=%d, total=%d, db=%d, weaviate=%d",
            newsapi_count,
            rss_count,
            len(all_articles),
            stored_db,
            stored_wv,
        )

        return {
            "newsapi": newsapi_count,
            "rss": rss_count,
            "total": len(all_articles),
            "analyzed": len(all_articles),
            "stored_db": stored_db,
            "stored_weaviate": stored_wv,
        }

    @staticmethod
    def _deduplicate_articles(
        articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deduplicate articles by article_id, keeping the one with most content."""
        seen: dict[str, dict[str, Any]] = {}
        for art in articles:
            aid = art.get("article_id", "")
            if not aid:
                continue
            existing = seen.get(aid)
            if existing is None or len(art.get("content", "")) > len(
                existing.get("content", "")
            ):
                seen[aid] = art
        return list(seen.values())
