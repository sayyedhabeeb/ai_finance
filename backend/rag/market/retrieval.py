"""
Market document retrieval with reranking and contextual compression.

Uses Weaviate hybrid search as the primary retrieval mechanism,
with optional cross-encoder or Cohere reranking for improved relevance.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

import numpy as np

from backend.rag.embeddings.embedder import EmbeddingService
from backend.rag.market.weaviate_client import MarketWeaviateClient
from backend.rag.models import RetrievalResult, RetrievalResponse

logger = logging.getLogger(__name__)

# Default rerank cutoff – results below this score are dropped
_DEFAULT_SCORE_THRESHOLD = 0.15


class MarketRetriever:
    """Retrieves and reranks market documents from Weaviate.

    Parameters
    ----------
    weaviate_client:
        Connected :class:`MarketWeaviateClient`.
    embedder:
        :class:`EmbeddingService` for computing query embeddings.
    cohere_api_key:
        Optional Cohere API key for reranking.  If provided, the
        ``cohere`` package must be installed.
    rerank_fn:
        Optional custom reranking callable
        ``(query: str, results: list[RetrievalResult]) -> list[RetrievalResult]``.
    """

    def __init__(
        self,
        weaviate_client: MarketWeaviateClient,
        embedder: EmbeddingService,
        cohere_api_key: str | None = None,
        rerank_fn: Callable[[str, list[RetrievalResult]], list[RetrievalResult]] | None = None,
    ) -> None:
        self._wv = weaviate_client
        self._embedder = embedder
        self._cohere_key = cohere_api_key
        self._rerank_fn = rerank_fn
        self._cohere_client: Any | None = None

    # ── Main retrieval API ──────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection_name: str = "MarketDocuments",
        alpha: float = 0.7,
        enable_rerank: bool = True,
        enable_compression: bool = True,
        min_score: float = _DEFAULT_SCORE_THRESHOLD,
    ) -> RetrievalResponse:
        """Full retrieval pipeline: hybrid search → rerank → compress.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Maximum number of results to return.
        filters:
            Optional Weaviate-compatible filters.
        collection_name:
            Target Weaviate collection.
        alpha:
            Hybrid search alpha (0 = BM25, 1 = vector).
        enable_rerank:
            Whether to apply reranking.
        enable_compression:
            Whether to apply contextual compression.
        min_score:
            Minimum relevance score to keep a result.

        Returns
        -------
        RetrievalResponse
        """
        start_time = time.time()

        # 1. Compute query embedding
        query_vector = self._embedder.embed_text(query)

        # 2. Also search NewsDocuments if we're searching MarketDocuments
        all_results: list[RetrievalResult] = []

        # Search the primary collection
        primary_results = self._wv.hybrid_search(
            query=query,
            collection_name=collection_name,
            query_vector=query_vector,
            alpha=alpha,
            filters=filters,
            limit=top_k * 3,  # over-fetch for reranking
        )
        all_results.extend(primary_results)

        # If searching MarketDocuments, also cross-search NewsDocuments
        if collection_name == "MarketDocuments":
            news_results = self._wv.hybrid_search(
                query=query,
                collection_name="NewsDocuments",
                query_vector=query_vector,
                alpha=alpha,
                filters=filters,
                limit=top_k,
            )
            all_results.extend(news_results)

        logger.info("Retrieved %d raw results for query: %s", len(all_results), query[:80])

        # 3. Rerank
        if enable_rerank and all_results:
            all_results = self._rerank(query, all_results)

        # 4. Filter by score
        all_results = [r for r in all_results if r.score >= min_score]

        # 5. Contextual compression
        if enable_compression and all_results:
            all_results = self._contextual_compress(query, all_results)

        # 6. Deduplicate by content (within the same doc_id)
        all_results = self._deduplicate(all_results)

        # 7. Sort by score and truncate
        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[:top_k]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "Returning %d results in %.1fms for query: %s",
            len(all_results),
            elapsed_ms,
            query[:80],
        )

        return RetrievalResponse(
            query=query,
            results=all_results,
            total_found=len(all_results),
            retrieval_time_ms=elapsed_ms,
        )

    # ── Reranking ───────────────────────────────────────────

    def _rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Apply reranking using Cohere or a custom reranking function."""
        # Prefer custom rerank function
        if self._rerank_fn is not None:
            try:
                return self._rerank_fn(query, results)
            except Exception as exc:
                logger.warning("Custom rerank failed (%s). Falling back.", exc)

        # Try Cohere rerank
        if self._cohere_key:
            try:
                return self._cohere_rerank(query, results)
            except Exception as exc:
                logger.warning("Cohere rerank failed (%s). Falling back to score-based.", exc)

        # Fallback: cross-encoder-like scoring using embedding similarity
        return self._embedding_rerank(query, results)

    def _cohere_rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Rerank using the Cohere Rerank API."""
        import cohere

        if self._cohere_client is None:
            self._cohere_client = cohere.ClientV2(self._cohere_key)

        documents = [r.content for r in results]
        response = self._cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=len(results),
        )

        reranked: list[RetrievalResult] = []
        for item in response.results:
            original = results[item.index]
            reranked.append(
                RetrievalResult(
                    content=original.content,
                    doc_id=original.doc_id,
                    chunk_id=original.chunk_id,
                    title=original.title,
                    source=original.source,
                    score=item.relevance_score,
                    doc_type=original.doc_type,
                    metadata=original.metadata,
                    highlight=item.document.get("text", "")[:200] if hasattr(item.document, "get") else "",
                )
            )

        return reranked

    def _embedding_rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Fallback reranking using query-document embedding cosine similarity."""
        query_embedding = self._embedder.embed_text(query)
        query_vec = np.array(query_embedding, dtype=np.float32)

        for r in results:
            doc_vec = np.array(self._embedder.embed_text(r.content), dtype=np.float32)
            norm_q = np.linalg.norm(query_vec)
            norm_d = np.linalg.norm(doc_vec)
            if norm_q > 0 and norm_d > 0:
                cos_sim = float(np.dot(query_vec, doc_vec) / (norm_q * norm_d))
            else:
                cos_sim = 0.0
            # Blend original score with cosine similarity
            r.score = 0.4 * r.score + 0.6 * cos_sim

        return results

    # ── Contextual compression ──────────────────────────────

    def _contextual_compress(
        self,
        query: str,
        results: list[RetrievalResult],
        max_chars: int = 1500,
    ) -> list[RetrievalResult]:
        """Compress retrieved context to remove irrelevant passages.

        Uses a simple extractive approach: splits content into sentences
        and keeps only those that contain terms overlapping with the query.
        """
        query_terms = set(re.findall(r"\b\w+\b", query.lower()))

        # Remove common stop words from query terms
        stop_words = {
            "what", "the", "is", "a", "an", "of", "in", "to", "for", "and",
            "how", "are", "was", "were", "be", "been", "being", "do", "does",
            "did", "will", "would", "could", "should", "can", "may", "might",
            "this", "that", "these", "those", "it", "its", "with", "on", "at",
            "by", "from", "or", "not", "no", "but", "about", "which", "who",
            "when", "where", "why", "all", "each", "every", "both", "any",
        }
        query_terms -= stop_words

        if not query_terms:
            return results

        compressed: list[RetrievalResult] = []
        for result in results:
            sentences = re.split(r"(?<=[.!?])\s+", result.content)
            relevant_sentences: list[str] = []

            for sent in sentences:
                sent_terms = set(re.findall(r"\b\w+\b", sent.lower()))
                overlap = len(query_terms & sent_terms)
                # Keep sentences with at least 1 overlapping term
                if overlap > 0 or len(sent) < 50:
                    relevant_sentences.append(sent)
                elif not relevant_sentences:
                    # Always keep at least the first sentence
                    relevant_sentences.append(sent)

            compressed_text = " ".join(relevant_sentences)
            if len(compressed_text) > max_chars:
                compressed_text = compressed_text[:max_chars] + "..."

            if compressed_text.strip():
                compressed.append(
                    RetrievalResult(
                        content=compressed_text,
                        doc_id=result.doc_id,
                        chunk_id=result.chunk_id,
                        title=result.title,
                        source=result.source,
                        score=result.score,
                        doc_type=result.doc_type,
                        metadata=result.metadata,
                        highlight=self._extract_highlight(compressed_text, query_terms),
                    )
                )

        return compressed

    @staticmethod
    def _extract_highlight(text: str, query_terms: set[str], max_length: int = 250) -> str:
        """Extract a short highlight snippet from text based on query terms."""
        sentences = re.split(r"(?<=[.!?])\s+", text)

        best_sentence = ""
        best_overlap = 0
        for sent in sentences:
            sent_terms = set(re.findall(r"\b\w+\b", sent.lower()))
            overlap = len(query_terms & sent_terms)
            if overlap > best_overlap and len(sent) > 30:
                best_overlap = overlap
                best_sentence = sent

        if not best_sentence:
            best_sentence = sentences[0] if sentences else ""

        if len(best_sentence) > max_length:
            # Trim to a reasonable window around the first query term match
            first_match = len(best_sentence)
            for term in query_terms:
                idx = best_sentence.lower().find(term)
                if idx != -1 and idx < first_match:
                    first_match = idx
            start = max(0, first_match - 50)
            end = min(len(best_sentence), start + max_length)
            best_sentence = "..." + best_sentence[start:end] + "..."

        return best_sentence

    # ── Deduplication ───────────────────────────────────────

    @staticmethod
    def _deduplicate(results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Remove duplicate content chunks, keeping the highest-scoring one."""
        seen_content: dict[str, RetrievalResult] = {}

        for r in results:
            # Normalise content for comparison
            key = r.content.strip()[:200]  # first 200 chars as fingerprint
            if key not in seen_content or r.score > seen_content[key].score:
                seen_content[key] = r

        # Sort by score descending
        deduped = list(seen_content.values())
        deduped.sort(key=lambda r: r.score, reverse=True)
        return deduped

    # ── Metadata extraction ─────────────────────────────────

    @staticmethod
    def extract_metadata(results: list[RetrievalResult]) -> dict[str, Any]:
        """Extract aggregate metadata from a set of retrieval results.

        Returns a dict with:
          - ``tickers``: set of all ticker symbols found
          - ``sectors``: set of sectors
          - ``doc_types``: set of document types
          - ``date_range``: (earliest, latest) dates
          - ``avg_sentiment``: average sentiment score
        """
        tickers: set[str] = set()
        sectors: set[str] = set()
        doc_types: set[str] = set()
        dates: list[str] = []
        sentiments: list[float] = []

        for r in results:
            ts = r.metadata.get("tickerSymbols") or r.metadata.get("tickers") or []
            if isinstance(ts, list):
                tickers.update(ts)
            elif isinstance(ts, str):
                tickers.add(ts)

            sector = r.metadata.get("sector", "")
            if sector:
                sectors.add(sector)

            if r.doc_type:
                doc_types.add(r.doc_type)

            date_str = r.metadata.get("date") or r.metadata.get("publishDate", "")
            if date_str:
                dates.append(date_str)

            sent = r.metadata.get("sentimentScore") or r.metadata.get("sentiment", 0.0)
            if sent:
                sentiments.append(float(sent))

        date_range = ("", "")
        if dates:
            dates_sorted = sorted(dates)
            date_range = (dates_sorted[0], dates_sorted[-1])

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        return {
            "tickers": sorted(tickers),
            "sectors": sorted(sectors),
            "doc_types": sorted(doc_types),
            "date_range": date_range,
            "avg_sentiment": avg_sentiment,
            "result_count": len(results),
        }

    # ── Format for agent consumption ────────────────────────

    @staticmethod
    def format_for_agent(results: list[RetrievalResult], max_results: int = 5) -> str:
        """Format retrieval results as a structured text block for LLM agents.

        Each result is prefixed with metadata (source, type, score) and
        truncated to keep the total context manageable.
        """
        if not results:
            return "No relevant documents found."

        blocks: list[str] = []
        for i, r in enumerate(results[:max_results], 1):
            meta_parts = [f"Type: {r.doc_type}"]
            if r.title:
                meta_parts.append(f"Title: {r.title}")
            if r.source:
                meta_parts.append(f"Source: {r.source}")
            meta_parts.append(f"Relevance: {r.score:.2%}")

            tickers = r.metadata.get("tickerSymbols") or r.metadata.get("tickers") or []
            if tickers:
                if isinstance(tickers, list):
                    tickers_str = ", ".join(tickers)
                else:
                    tickers_str = str(tickers)
                meta_parts.append(f"Tickers: {tickers_str}")

            header = f"[Document {i}] {' | '.join(meta_parts)}"
            content = r.content[:2000]
            if len(r.content) > 2000:
                content += "\n... [truncated]"

            blocks.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(blocks)
