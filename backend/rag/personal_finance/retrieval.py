"""
Personal finance document retrieval with sub-question decomposition.

Complex financial queries (e.g. "How can I save tax if I earn ₹15L?") are
decomposed into simpler sub-questions, each retrieved independently, and
results are merged and reranked for the final answer.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from backend.rag.embeddings.embedder import EmbeddingService
from backend.rag.models import RetrievalResult, RetrievalResponse
from backend.rag.personal_finance.pgvector_client import PersonalFinanceVectorStore

logger = logging.getLogger(__name__)

# Complexity heuristics for sub-question decomposition
_COMPLEXITY_PATTERNS = re.compile(
    r"\b(?:"
    r"how (?:can|do|should|to)|"
    r"what (?:is|are|will|would|should)|"
    r"compare|comparison|"
    r"difference (?:between)|vs\.?|versus|"
    r"(?:both|all|multiple|various|several)|"
    r"(?:and|also|additionally|plus|along with)|"
    r"(?:if|whether|either|or)\b"
    r")",
    re.IGNORECASE,
)


class PersonalFinanceRetriever:
    """Retrieves personal finance documents with sub-question decomposition.

    Parameters
    ----------
    vector_store:
        Connected :class:`PersonalFinanceVectorStore`.
    embedder:
        :class:`EmbeddingService` for computing query embeddings.
    enable_decomposition:
        Whether to decompose complex queries into sub-questions.
    """

    def __init__(
        self,
        vector_store: PersonalFinanceVectorStore,
        embedder: EmbeddingService,
        enable_decomposition: bool = True,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self._enable_decomposition = enable_decomposition

    # ── Main retrieval API ──────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool = True,
        alpha: float = 0.7,
    ) -> RetrievalResponse:
        """Retrieve documents for a query, with optional sub-question decomposition.

        Parameters
        ----------
        query:
            Natural-language query about personal finance / tax.
        top_k:
            Maximum number of results.
        filters:
            Optional filters (e.g. ``{"category": "deductions"}``).
        use_hybrid:
            Whether to use hybrid (vector + full-text) search.
        alpha:
            Hybrid search alpha weight.

        Returns
        -------
        RetrievalResponse
        """
        start_time = time.time()

        # Decide whether to decompose
        sub_questions = []
        if self._enable_decomposition and self._is_complex_query(query):
            sub_questions = self.decompose_query(query)
            logger.info(
                "Decomposed query into %d sub-questions: %s",
                len(sub_questions),
                sub_questions,
            )
            results = await self._retrieve_with_decomposition(
                sub_questions, top_k, filters, use_hybrid, alpha
            )
        else:
            results = await self._single_retrieve(
                query, top_k=top_k, filters=filters, use_hybrid=use_hybrid, alpha=alpha
            )

        elapsed_ms = (time.time() - start_time) * 1000

        return RetrievalResponse(
            query=query,
            results=results,
            total_found=len(results),
            retrieval_time_ms=elapsed_ms,
            sub_questions=sub_questions,
        )

    # ── Sub-question decomposition ──────────────────────────

    @staticmethod
    def is_complex_query(query: str) -> bool:
        """Determine if a query is complex enough for decomposition."""
        # Check for multiple clauses
        and_positions = [m.start() for m in re.finditer(r"\band\b", query, re.IGNORECASE)]
        question_marks = query.count("?")

        # Check for comparative language
        has_comparison = bool(re.search(
            r"\b(?:compare|difference|vs\.?|versus|between|better|cheaper)\b",
            query,
            re.IGNORECASE,
        ))

        # Check for multiple distinct topics
        financial_keywords = re.findall(
            r"\b(?:tax|deduction|section|80C|80D|gst|insurance|mutual fund|fd|ppf|nps|sip|loan|emi|home loan|car loan|education loan|retirement|pension|salary|income|investment|savings)\b",
            query,
            re.IGNORECASE,
        )
        unique_keywords = set(kw.lower() for kw in financial_keywords)

        score = 0
        score += min(len(and_positions), 2)  # 0-2
        score += min(question_marks, 2)  # 0-2
        score += int(has_comparison)  # 0-1
        score += min(len(unique_keywords), 3)  # 0-3

        return score >= 2

    def _is_complex_query(self, query: str) -> bool:
        """Instance method wrapper for complexity check."""
        return self.is_complex_query(query)

    @staticmethod
    def decompose_query(query: str) -> list[str]:
        """Break a complex query into simpler sub-questions.

        Strategies:
          1. Split on conjunctions (and, also, plus, along with)
          2. Split on comparative patterns (vs, versus, difference between)
          3. If no clean split, generate focused sub-questions based on
             detected financial keywords.

        Returns
        -------
        list[str]
            List of sub-question strings (always at least 1).
        """
        # Strategy 1: Split on "and", "also", "plus", "along with"
        conj_pattern = re.compile(
            r"\s+(?:and|also|plus|along with|as well as)\s+",
            re.IGNORECASE,
        )
        parts = conj_pattern.split(query)
        if len(parts) >= 2:
            # Verify each part is meaningful (> 10 chars)
            meaningful = [p.strip() for p in parts if len(p.strip()) > 10]
            if len(meaningful) >= 2:
                return meaningful

        # Strategy 2: Split on comparative patterns
        comp_pattern = re.compile(
            r"\b(?:vs\.?|versus|difference between|compared to|compared with)\b",
            re.IGNORECASE,
        )
        comp_match = comp_pattern.search(query)
        if comp_match:
            left = query[: comp_match.start()].strip()
            right = query[comp_match.end():].strip()
            if left and right:
                sub_questions = []
                if left:
                    sub_questions.append(f"What is {left}?")
                if right:
                    sub_questions.append(f"What is {right}?")
                sub_questions.append(f"How does {left} compare with {right}?")
                return sub_questions

        # Strategy 3: Keyword-based sub-question generation
        financial_topics = re.findall(
            r"\b(?:tax|deduction|section \w[\w()]*|gst|insurance|mutual fund|fd|ppf|nps|sip|loan|emi|home loan|car loan|education loan|retirement|pension|salary|income|investment|savings|ltcg|stcg|capital gain|nifty|sensex|equity|debt|bond|fixed deposit|recurring deposit|gold|real estate|crypto|sip|elss|ulip)\b",
            query,
            re.IGNORECASE,
        )
        unique_topics = list(dict.fromkeys(t.lower() for t in financial_topics))

        if len(unique_topics) >= 2:
            sub_questions = []
            for topic in unique_topics:
                # Rebuild a focused sub-question
                sub = f"What are the rules and benefits related to {topic}?"
                sub_questions.append(sub)
            # Add a combined question
            sub_questions.append(query)
            return sub_questions

        # Fallback: return original query
        return [query]

    # ── Multi-sub-question retrieval ────────────────────────

    async def retrieve_for_subquestions(
        self,
        subquestions: list[str],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool = True,
        alpha: float = 0.7,
    ) -> dict[str, list[RetrievalResult]]:
        """Retrieve documents for each sub-question independently.

        Returns
        -------
        dict[str, list[RetrievalResult]]
            Map from sub-question to its retrieval results.
        """
        results: dict[str, list[RetrievalResult]] = {}

        # Execute sub-question retrievals concurrently
        tasks = [
            self._single_retrieve(sq, top_k=top_k, filters=filters, use_hybrid=use_hybrid, alpha=alpha)
            for sq in subquestions
        ]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for sq, result in zip(subquestions, task_results):
            if isinstance(result, Exception):
                logger.error("Failed to retrieve for sub-question '%s': %s", sq, result)
                results[sq] = []
            else:
                results[sq] = result

        return results

    @staticmethod
    def merge_results(results: dict[str, list[RetrievalResult]]) -> list[RetrievalResult]:
        """Merge results from multiple sub-questions, deduplicate, and rerank.

        The merge strategy:
          1. Collect all unique results across sub-questions
          2. If a chunk appears in multiple sub-question results, boost its score
          3. Sort by final score and return top results
        """
        # Collect results with their source sub-questions
        content_map: dict[str, tuple[RetrievalResult, set[str]]] = {}

        for sub_q, res_list in results.items():
            for r in res_list:
                key = r.content.strip()[:300]  # fingerprint
                if key in content_map:
                    # Boost score and record the additional sub-question
                    existing_result, sub_qs = content_map[key]
                    existing_result.score = min(1.0, existing_result.score + 0.05)
                    sub_qs.add(sub_q)
                else:
                    # Clone the result to avoid mutating the original
                    new_result = RetrievalResult(
                        content=r.content,
                        doc_id=r.doc_id,
                        chunk_id=r.chunk_id,
                        title=r.title,
                        source=r.source,
                        score=r.score,
                        doc_type=r.doc_type,
                        metadata={**r.metadata},
                        highlight=r.highlight,
                    )
                    content_map[key] = (new_result, {sub_q})

        # Sort by score descending
        merged = sorted(
            [r for r, _ in content_map.values()],
            key=lambda r: r.score,
            reverse=True,
        )
        return merged

    # ── Single retrieval ────────────────────────────────────

    async def _single_retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool = True,
        alpha: float = 0.7,
    ) -> list[RetrievalResult]:
        """Retrieve documents for a single query."""
        # Compute query embedding
        query_embedding = self._embedder.embed_text(query)

        # Search
        if use_hybrid:
            raw_results = await self._vs.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                alpha=alpha,
                filters=filters,
            )
        else:
            raw_results = await self._vs.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )

        # Convert to RetrievalResult
        results = []
        for row in raw_results:
            results.append(
                RetrievalResult(
                    content=row["content"],
                    doc_id=row.get("doc_id", ""),
                    chunk_id=row.get("chunk_id", ""),
                    title=row.get("title", ""),
                    source=row.get("source", ""),
                    score=row.get("score", 0.0),
                    doc_type=row.get("doc_type", ""),
                    metadata=row.get("metadata", {}),
                )
            )

        return results

    async def _retrieve_with_decomposition(
        self,
        sub_questions: list[str],
        top_k: int,
        filters: dict[str, Any] | None,
        use_hybrid: bool,
        alpha: float,
    ) -> list[RetrievalResult]:
        """Retrieve using sub-question decomposition."""
        sub_results = await self.retrieve_for_subquestions(
            sub_questions, top_k=top_k, filters=filters, use_hybrid=use_hybrid, alpha=alpha
        )
        merged = self.merge_results(sub_results)
        return merged[:top_k]

    # ── Context formatting ──────────────────────────────────

    @staticmethod
    def format_context(results: list[RetrievalResult], max_results: int = 8, max_chars: int = 2000) -> str:
        """Format retrieval results into a context string for LLM consumption.

        Includes document metadata (type, category, source) and truncates
        long content to keep the context window manageable.
        """
        if not results:
            return "No relevant personal finance documents found."

        blocks: list[str] = []

        for i, r in enumerate(results[:max_results], 1):
            meta_parts = [f"Type: {r.doc_type}"]
            category = r.metadata.get("category", "")
            if category:
                meta_parts.append(f"Category: {category}")
            if r.title:
                meta_parts.append(f"Title: {r.title}")
            meta_parts.append(f"Relevance: {r.score:.2%}")

            header = f"[Document {i}] {' | '.join(meta_parts)}"
            content = r.content[:max_chars]
            if len(r.content) > max_chars:
                content += "\n... [truncated]"

            blocks.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(blocks)

    # ── Convenience sync wrapper ────────────────────────────

    def retrieve_sync(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Synchronous wrapper around :meth:`retrieve` for use in non-async contexts."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async event loop – create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.retrieve(query, top_k=top_k, filters=filters),
                )
                return future.result()
        else:
            return asyncio.run(self.retrieve(query, top_k=top_k, filters=filters))
