"""
Pipeline for ingesting market documents into Weaviate.

Supports multiple source types:
  - Web URLs (articles, blog posts)
  - Raw text content
  - Batch document uploads
  - SEC / NSE filings (PDF, HTML, XML)
  - Earnings call transcripts

Pipeline stages: **fetch → clean → chunk → embed → store**
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from backend.rag.chunking.strategies import DocumentChunker
from backend.rag.embeddings.embedder import EmbeddingService
from backend.rag.market.weaviate_client import MarketWeaviateClient
from backend.rag.models import ChunkedDocument, RAGDocument, RetrievalResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Text cleaning utilities
# ──────────────────────────────────────────────────────────────

def clean_html(html: str) -> str:
    """Extract visible text from HTML, stripping tags, scripts, and styles."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove excessive whitespace within lines
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    return text.strip()


def extract_metadata_from_html(html: str) -> dict[str, str]:
    """Extract title, description, and other meta tags from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    metadata: dict[str, str] = {}

    title_tag = soup.find("title")
    if title_tag:
        metadata["title"] = title_tag.get_text(strip=True)

    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        content = meta.get("content", "")
        if name in ("description", "author", "date", "og:title", "og:description"):
            key = name.replace("og:", "")
            metadata[key] = content

    # Try h1 if no title
    if "title" not in metadata:
        h1 = soup.find("h1")
        if h1:
            metadata["title"] = h1.get_text(strip=True)

    return metadata


def detect_doc_type_from_url(url: str) -> str:
    """Heuristic doc_type detection from URL patterns."""
    url_lower = url.lower()
    patterns: list[tuple[str, str]] = [
        (r"sec\.gov|\.gov/edgar|10-[kq]", "sec_filing"),
        (r"nseindia|bseindia|\.pdf.*filing|regulation", "nse_filing"),
        (r"transcript|earnings.call|conference.call", "earnings_transcript"),
        (r"annual.?report|annual.?review", "annual_report"),
        (r"quarterly|q[1234].?result", "quarterly_report"),
        (r"news|article|blog|press.?release", "news_article"),
        (r"research|analysis|report|note", "research_report"),
    ]
    for pattern, doc_type in patterns:
        if re.search(pattern, url_lower):
            return doc_type
    return "web_article"


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

class MarketDocumentIngestionPipeline:
    """Pipeline for ingesting market documents into Weaviate.

    Stages: **fetch → clean → chunk → embed → store**

    Parameters
    ----------
    weaviate_client:
        Connected :class:`MarketWeaviateClient`.
    embedder:
        :class:`EmbeddingService` for computing embeddings.
    chunker:
        :class:`DocumentChunker` instance.
    chunking_strategy:
        Default chunking strategy (``"recursive"``, ``"semantic"``,
        ``"financial_report"``, ``"news"``).
    chunk_size:
        Default max characters per chunk.
    chunk_overlap:
        Default overlap between chunks.
    """

    def __init__(
        self,
        weaviate_client: MarketWeaviateClient,
        embedder: EmbeddingService,
        chunker: DocumentChunker | None = None,
        chunking_strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._wv = weaviate_client
        self._embedder = embedder
        self._chunker = chunker or DocumentChunker()
        self._strategy = chunking_strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._seen_checksums: set[str] = set()
        self._http_client = httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": "AI-Financial-Brain/1.0"},
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()

    def __enter__(self) -> "MarketDocumentIngestionPipeline":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Public API ──────────────────────────────────────────

    def ingest_url(self, url: str, metadata: dict[str, Any] | None = None) -> int:
        """Fetch a URL, clean the content, chunk, embed, and store.

        Parameters
        ----------
        url:
            URL to fetch and ingest.
        metadata:
            Additional metadata to attach to the document.

        Returns
        -------
        int
            Number of chunks stored in Weaviate.
        """
        metadata = metadata or {}

        # 1. Fetch
        logger.info("Fetching URL: %s", url)
        html = self._fetch_url(url)

        # 2. Clean & extract
        text = clean_html(html)
        if not text or len(text) < 50:
            logger.warning("URL produced too little text (< 50 chars): %s", url)
            return 0

        extracted_meta = extract_metadata_from_html(html)
        meta = {**extracted_meta, **metadata}

        doc_type = meta.pop("doc_type", detect_doc_type_from_url(url))
        title = meta.pop("title", "")

        # 3. Create RAGDocument
        doc = RAGDocument(
            content=text,
            title=title,
            source=url,
            doc_type=doc_type,  # type: ignore[arg-type]
            metadata=meta,
        )

        # 4. Dedup check
        if doc.checksum in self._seen_checksums:
            logger.info("Skipping duplicate URL: %s", url)
            return 0
        self._seen_checksums.add(doc.checksum)

        return self._process_document(doc)

    def ingest_text(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest raw text content directly.

        Parameters
        ----------
        content:
            Full text content to ingest.
        metadata:
            Additional metadata.  May include ``title``, ``source``,
            ``doc_type``, ``ticker_symbols``, ``sector``, ``date``, etc.

        Returns
        -------
        int
            Number of chunks stored.
        """
        metadata = metadata or {}

        if not content or len(content.strip()) < 10:
            logger.warning("Content too short to ingest (< 10 chars).")
            return 0

        doc = RAGDocument(
            content=content.strip(),
            title=metadata.pop("title", ""),
            source=metadata.pop("source", ""),
            doc_type=metadata.pop("doc_type", "text_document"),  # type: ignore[arg-type]
            ticker_symbols=metadata.pop("ticker_symbols", []),
            sector=metadata.pop("sector", ""),
            date=metadata.pop("date", ""),
            sentiment_score=metadata.pop("sentiment_score", 0.0),
            metadata=metadata,
        )

        if doc.checksum in self._seen_checksums:
            logger.info("Skipping duplicate text (checksum: %s).", doc.checksum[:12])
            return 0
        self._seen_checksums.add(doc.checksum)

        return self._process_document(doc)

    def ingest_batch(self, documents: list[dict[str, Any]]) -> int:
        """Ingest a batch of document dicts.

        Each dict should have at least ``content`` and may include
        any fields accepted by :meth:`ingest_text`.

        Returns
        -------
        int
            Total number of chunks stored across all documents.
        """
        total_chunks = 0
        for doc_dict in documents:
            try:
                total_chunks += self.ingest_text(
                    content=doc_dict.pop("content", ""),
                    metadata=doc_dict,
                )
            except Exception as exc:
                logger.error("Failed to ingest batch document: %s", exc)
        return total_chunks

    def process_filing(
        self,
        filing_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Process an SEC/NSE filing file.

        Supports:
          - PDF files (extracted via text extraction)
          - HTML/XML filings
          - Plain text filings

        Parameters
        ----------
        filing_path:
            Path to the filing file (local path or URL).
        metadata:
            Additional metadata.

        Returns
        -------
        int
            Number of chunks stored.
        """
        metadata = metadata or {}

        # Check if it's a URL
        if filing_path.startswith(("http://", "https://")):
            return self.ingest_url(filing_path, metadata)

        # Local file
        try:
            with open(filing_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            logger.error("Filing file not found: %s", filing_path)
            return 0
        except Exception as exc:
            logger.error("Failed to read filing file %s: %s", filing_path, exc)
            return 0

        if not content.strip():
            logger.warning("Filing file is empty: %s", filing_path)
            return 0

        # Detect if it's HTML
        if content.strip().startswith("<") and "</" in content:
            content = clean_html(content)

        metadata.setdefault("doc_type", "regulatory_filing")
        metadata.setdefault("source", filing_path)

        return self.ingest_text(content, metadata)

    # ── Internal processing ─────────────────────────────────

    def _process_document(self, doc: RAGDocument) -> int:
        """Run the full chunk → embed → store pipeline on a document.

        Returns the number of chunks stored.
        """
        # Select appropriate chunking strategy
        strategy = self._select_strategy(doc)
        logger.info(
            "Chunking doc '%s' (%d chars) with strategy '%s'.",
            doc.title or doc.doc_id,
            len(doc.content),
            strategy,
        )

        if strategy == "news":
            chunks = self._chunker.news_chunk(doc.content, max_chars=self._chunk_size)
        elif strategy == "financial_report":
            chunks = self._chunker.financial_report_chunk(
                doc.content, max_chars=self._chunk_size, overlap=self._chunk_overlap
            )
        elif strategy == "semantic":
            chunks = self._chunker.semantic_chunk(doc.content, max_tokens=self._chunk_size // 4)
        else:
            chunks = self._chunker.recursive_chunk(
                doc.content, chunk_size=self._chunk_size, overlap=self._chunk_overlap
            )

        if not chunks:
            logger.warning("No chunks produced for doc '%s'.", doc.title or doc.doc_id)
            return 0

        logger.info("Produced %d chunks.", len(chunks))

        # Embed
        embeddings = self._embedder.embed_batch(chunks)

        # Determine collection
        is_news = doc.doc_type.value in ("news_article", "market_commentary")
        collection_name = "NewsDocuments" if is_news else "MarketDocuments"

        # Build chunk objects for Weaviate
        chunk_dicts: list[dict[str, Any]] = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_dict = self._chunk_to_weaviate_dict(
                chunk_text=chunk_text,
                embedding=embedding,
                doc=doc,
                chunk_index=idx,
                is_news=is_news,
            )
            chunk_dicts.append(chunk_dict)

        # Store
        stored = self._wv.batch_ingest_chunks(chunk_dicts, collection_name=collection_name)
        logger.info("Stored %d/%d chunks in %s.", stored, len(chunks), collection_name)
        return stored

    def _select_strategy(self, doc: RAGDocument) -> str:
        """Choose the best chunking strategy based on document type."""
        type_strategies: dict[str, str] = {
            "news_article": "news",
            "market_commentary": "news",
            "annual_report": "financial_report",
            "quarterly_report": "financial_report",
            "research_report": "financial_report",
            "earnings_transcript": "financial_report",
            "sec_filing": "financial_report",
            "nse_filing": "financial_report",
            "regulatory_filing": "financial_report",
        }
        return type_strategies.get(doc.doc_type.value, self._strategy)

    @staticmethod
    def _chunk_to_weaviate_dict(
        chunk_text: str,
        embedding: list[float],
        doc: RAGDocument,
        chunk_index: int,
        is_news: bool,
    ) -> dict[str, Any]:
        """Convert a chunk into a Weaviate-compatible property dict."""
        base: dict[str, Any] = {
            "content": chunk_text,
            "text": chunk_text,
            "docId": doc.doc_id,
            "chunkIndex": chunk_index,
            "checksum": hashlib.sha256(
                f"{chunk_text}:{doc.doc_id}:{chunk_index}".encode("utf-8")
            ).hexdigest(),
            "vector": embedding,
        }

        if is_news:
            base.update({
                "title": doc.title,
                "source": doc.source,
                "publishDate": doc.date,
                "tickers": doc.ticker_symbols,
                "sentiment": doc.sentiment_score,
                "summary": doc.metadata.get("summary", ""),
            })
        else:
            base.update({
                "title": doc.title,
                "source": doc.source,
                "docType": doc.doc_type.value,
                "date": doc.date,
                "tickerSymbols": doc.ticker_symbols,
                "sector": doc.sector,
                "sentimentScore": doc.sentiment_score,
            })

        return base

    # ── HTTP fetching ───────────────────────────────────────

    def _fetch_url(self, url: str) -> str:
        """Fetch URL content with error handling and retries."""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._http_client.get(url)
                response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError as exc:
                last_error = exc
                logger.warning(
                    "HTTP %d fetching %s (attempt %d/3).",
                    exc.response.status_code,
                    url,
                    attempt + 1,
                )
                if exc.response.status_code >= 500:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    break
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning("Request error fetching %s (attempt %d/3): %s", url, attempt + 1, exc)
                import time
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to fetch URL after 3 attempts: {url}") from last_error
