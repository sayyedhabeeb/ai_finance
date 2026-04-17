"""
Document chunking strategies for the RAG system.

Provides multiple chunking approaches suited for different document types:
  - Recursive character chunking (general purpose)
  - Semantic chunking (embedding-based boundary detection)
  - Financial report chunking (section-aware)
  - News article chunking (paragraph + headline preservation)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.rag.models import ChunkedDocument, RAGDocument


# ──────────────────────────────────────────────────────────────
# Text splitters
# ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A raw text chunk before being wrapped in ChunkedDocument."""

    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_by_separator(text: str, separator: str) -> list[str]:
    """Split text on a separator, filtering out empty strings."""
    parts = text.split(separator)
    return [p.strip() for p in parts if p.strip()]


def _merge_short_chunks(chunks: list[str], min_size: int) -> list[str]:
    """Merge consecutive chunks that fall below *min_size*."""
    if not chunks:
        return []
    merged: list[str] = []
    buffer = chunks[0]
    for ch in chunks[1:]:
        if len(buffer) < min_size:
            buffer = buffer + "\n" + ch
        else:
            merged.append(buffer)
            buffer = ch
    if buffer:
        merged.append(buffer)
    return merged


# ──────────────────────────────────────────────────────────────
# Main chunker
# ──────────────────────────────────────────────────────────────

class DocumentChunker:
    """Multiple chunking strategies for different document types.

    Usage::

        chunker = DocumentChunker()
        chunks = chunker.recursive_chunk(some_text, chunk_size=1000, overlap=200)
    """

    # ── Recursive character chunking ─────────────────────────

    def recursive_chunk(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: list[str] | None = None,
    ) -> list[str]:
        """Split *text* recursively using a hierarchy of separators.

        Parameters
        ----------
        text:
            Full document text.
        chunk_size:
            Maximum number of characters per chunk.
        overlap:
            Number of overlapping characters between consecutive chunks.
        separators:
            Ordered list of separators to try.  Defaults to
            ``["\\n\\n", "\\n", ". ", " ", ""]``.

        Returns
        -------
        list[str]
            Ordered list of text chunks.
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        final_chunks: list[str] = []
        # Find the best separator (largest that actually splits the text)
        for sep in separators:
            if sep == "":
                # No separator works – just split by character count
                return self._split_by_char_count(text, chunk_size, overlap)
            if sep in text:
                splits = _split_by_separator(text, sep)
                good_splits: list[str] = []
                for s in splits:
                    if len(s) <= chunk_size:
                        good_splits.append(s)
                    else:
                        # Recurse
                        good_splits.extend(
                            self.recursive_chunk(s, chunk_size, overlap, separators[separators.index(sep) + 1 :])
                        )
                return self._add_overlap(good_splits, overlap)
        return self._split_by_char_count(text, chunk_size, overlap)

    def _split_by_char_count(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Brute-force split by character count with overlap."""
        if len(text) <= chunk_size:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Try to break at a word boundary
            if end < len(text):
                # Look back up to 100 chars for whitespace
                look_back = min(100, end - start)
                space_idx = text.rfind(" ", end - look_back, end)
                if space_idx != -1:
                    end = space_idx
            chunks.append(text[start:end].strip())
            start = end - overlap
        return [c for c in chunks if c]

    @staticmethod
    def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
        """Add overlapping prefix from previous chunk to the next."""
        if overlap <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = result[-1]
            # Take the last `overlap` chars from the previous chunk
            ov = prev[-overlap:] if len(prev) >= overlap else prev
            result.append(ov + "\n" + chunks[i])
        return result

    # ── Semantic chunking ────────────────────────────────────

    def semantic_chunk(
        self,
        text: str,
        max_tokens: int = 512,
        buffer_size: int = 3,
        similarity_threshold: float = 0.5,
        embedding_fn: Any | None = None,
    ) -> list[str]:
        """Split *text* at natural semantic boundaries.

        Splits text into sentences, computes cosine similarity between
        consecutive sentence groups, and breaks where similarity drops
        below *similarity_threshold*.

        Parameters
        ----------
        text:
            Full document text.
        max_tokens:
            Approximate maximum tokens per chunk (chars estimated at ~4/token).
        embedding_fn:
            Callable ``(texts: list[str]) -> list[list[float]]``.
            If ``None``, falls back to a simple Jaccard similarity heuristic.
        buffer_size:
            Number of sentences to average when computing group embeddings.
        similarity_threshold:
            Minimum cosine/Jaccard similarity to keep merging.

        Returns
        -------
        list[str]
            Semantically coherent chunks.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text]

        if embedding_fn is None:
            return self._semantic_chunk_jaccard(sentences, max_tokens, buffer_size, similarity_threshold)

        return self._semantic_chunk_embedding(sentences, max_tokens, buffer_size, similarity_threshold, embedding_fn)

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations and numbered lists
        text = re.sub(r"(?<!\w)(?:(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|Corp|vs|etc|e\.g|i\.e)\.)\s", lambda m: m.group(0).replace(".", "§§PERIOD§§"), text)
        # Split on sentence boundaries
        sentence_pattern = r"(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<![§§])\.\s|\n\n|\?\s|!\s"
        raw = re.split(sentence_pattern, text)
        raw = [r.replace("§§PERIOD§§", ".") for r in raw]
        return [s.strip() for s in raw if s.strip()]

    def _semantic_chunk_jaccard(
        self, sentences: list[str], max_tokens: int, buffer_size: int, threshold: float
    ) -> list[str]:
        """Semantic chunking using Jaccard similarity (no model required)."""
        def _tokens(s: str) -> set[str]:
            return set(re.findall(r"\b\w+\b", s.lower()))

        max_chars = max_tokens * 4
        chunks: list[str] = []
        current_sentences: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            # Compute Jaccard between the last `buffer_size` sentences and current
            window_start = max(0, len(current_sentences) - buffer_size)
            left_text = " ".join(current_sentences[window_start:])
            left_tok = _tokens(left_text)
            right_tok = _tokens(sentences[i])

            if not left_tok or not right_tok:
                sim = 0.0
            else:
                sim = len(left_tok & right_tok) / len(left_tok | right_tok)

            combined = " ".join(current_sentences + [sentences[i]])
            if sim >= threshold and len(combined) <= max_chars:
                current_sentences.append(sentences[i])
            else:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i]]

        if current_sentences:
            chunks.append(" ".join(current_sentences))
        return chunks

    def _semantic_chunk_embedding(
        self,
        sentences: list[str],
        max_tokens: int,
        buffer_size: int,
        threshold: float,
        embed_fn: Any,
    ) -> list[str]:
        """Semantic chunking using sentence embeddings and cosine similarity."""
        max_chars = max_tokens * 4

        # Compute embeddings in batches of buffer_size+1
        all_embeddings: list[list[float]] = []
        batch: list[str] = []
        for s in sentences:
            batch.append(s)
            if len(batch) >= buffer_size + 1:
                embeddings = embed_fn(batch)
                all_embeddings.extend(embeddings)
                batch = []
        if batch:
            embeddings = embed_fn(batch)
            all_embeddings.extend(embeddings)

        chunks: list[str] = []
        current_start = 0

        for i in range(buffer_size, len(sentences)):
            left_emb = np.array(all_embeddings[i - buffer_size], dtype=np.float32)
            right_emb = np.array(all_embeddings[i], dtype=np.float32)

            norm_l = np.linalg.norm(left_emb)
            norm_r = np.linalg.norm(right_emb)
            if norm_l == 0 or norm_r == 0:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(left_emb, right_emb) / (norm_l * norm_r))

            chunk_text = " ".join(sentences[current_start:i + 1])
            if cos_sim < threshold or len(chunk_text) >= max_chars:
                if i > current_start:
                    chunks.append(" ".join(sentences[current_start:i]))
                current_start = i

        if current_start < len(sentences):
            chunks.append(" ".join(sentences[current_start:]))

        return chunks

    # ── Financial report chunking ────────────────────────────

    # Common section headers found in financial reports
    _SECTION_HEADERS = re.compile(
        r"(?:^|\n)"
        r"(?:(?:#{1,3}\s+)|(?:\d+\.?\s+))"
        r"(?:"
        r"Executive Summary|"
        r"Management Discussion|"
        r"MD&A|"
        r"Financial Summary|"
        r"Income Statement|"
        r"Balance Sheet|"
        r"Cash Flow|"
        r"Risk Factors|"
        r"Business Overview|"
        r"Results of Operations|"
        r"Liquidity and Capital Resources|"
        r"Market Risk|"
        r"Outlook|"
        r"Forward[- ]Looking Statements|"
        r"Revenue|"
        r"Expenses|"
        r"Net (?:Income|Profit|Loss)|"
        r"Earnings (?:Per Share|Call|Transcript)|"
        r"Shareholder|"
        r"Dividend|"
        r"Capital Expenditure|"
        r"Segment Performance|"
        r"Quarterly (?:Results|Highlights)|"
        r"Directors.+Report|"
        r"Auditors.+Report"
        r")",
        re.IGNORECASE,
    )

    def financial_report_chunk(self, text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
        """Section-aware chunking for financial reports.

        Splits on section headers first, then falls back to recursive
        chunking for oversized sections.
        """
        # Split by section headers
        sections: list[tuple[str, str]] = []
        matches = list(self._SECTION_HEADERS.finditer(text))

        if not matches:
            # No section headers found – fall back to recursive chunk
            return self.recursive_chunk(text, chunk_size=max_chars, overlap=overlap)

        for idx, match in enumerate(matches):
            header = match.group(0).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((header, section_text))

        # Build chunks
        chunks: list[str] = []
        for header, body in sections:
            # Prepend header to body for context
            full = f"{header}\n{body}"
            if len(full) <= max_chars:
                chunks.append(full)
            else:
                # Split oversized sections, keeping header in first chunk
                sub_chunks = self.recursive_chunk(body, chunk_size=max_chars, overlap=overlap)
                if sub_chunks:
                    sub_chunks[0] = f"{header}\n{sub_chunks[0]}"
                    chunks.extend(sub_chunks)
        return chunks

    # ── News article chunking ────────────────────────────────

    _HEADLINE_PATTERN = re.compile(r"^(?!\n)(.{10,150})\n", re.MULTILINE)

    def news_chunk(self, text: str, max_chars: int = 1500, preserve_headline: bool = True) -> list[str]:
        """Paragraph-based chunking that preserves the headline.

        Tries to keep paragraphs together and prepends the headline
        to every chunk for context.
        """
        lines = text.strip().split("\n")
        if not lines:
            return [text]

        # Extract headline (first non-empty line)
        headline = ""
        body_start = 0
        if preserve_headline:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped:
                    headline = stripped
                    body_start = i + 1
                    break

        # Join remaining body
        body = "\n".join(lines[body_start:]).strip()
        if not body:
            return [headline] if headline else [text]

        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", body)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge short paragraphs and split oversized ones
        chunks: list[str] = []
        current: list[str] = []

        for para in paragraphs:
            candidate = "\n\n".join(current + [para])
            if len(candidate) <= max_chars:
                current.append(para)
            else:
                if current:
                    chunks.append(self._prepend_headline("\n\n".join(current), headline))
                # If a single paragraph is oversized, split it
                if len(para) > max_chars:
                    sub_chunks = self.recursive_chunk(para, chunk_size=max_chars, overlap=150)
                    for sc in sub_chunks[:-1]:
                        chunks.append(self._prepend_headline(sc, headline))
                    current = [sub_chunks[-1]] if sub_chunks else []
                else:
                    current = [para]

        if current:
            chunks.append(self._prepend_headline("\n\n".join(current), headline))

        return chunks if chunks else [headline]

    @staticmethod
    def _prepend_headline(chunk: str, headline: str) -> str:
        """Prepend headline to chunk for context preservation."""
        if not headline:
            return chunk
        return f"[Headline: {headline}]\n\n{chunk}"

    # ── Generic public API ──────────────────────────────────

    def chunk_document(
        self,
        document: RAGDocument,
        strategy: str = "recursive",
        **kwargs: Any,
    ) -> list[ChunkedDocument]:
        """Chunk a :class:`RAGDocument` and wrap results.

        Parameters
        ----------
        document:
            The document to chunk.
        strategy:
            One of ``"recursive"``, ``"semantic"``, ``"financial_report"``, ``"news"``.
        **kwargs:
            Forwarded to the underlying chunk method.

        Returns
        -------
        list[ChunkedDocument]
        """
        dispatch = {
            "recursive": self.recursive_chunk,
            "semantic": self.semantic_chunk,
            "financial_report": self.financial_report_chunk,
            "news": self.news_chunk,
        }

        chunk_fn = dispatch.get(strategy)
        if chunk_fn is None:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose from {list(dispatch.keys())}")

        raw_chunks = chunk_fn(document.content, **kwargs)
        results: list[ChunkedDocument] = []
        for idx, text in enumerate(raw_chunks):
            chunk = ChunkedDocument(
                document_id=document.doc_id,
                content=text,
                chunk_index=idx,
                title=document.title,
                metadata={
                    "doc_type": document.doc_type.value,
                    "source": document.source,
                    "ticker_symbols": document.ticker_symbols,
                    "sector": document.sector,
                    "sentiment_score": document.sentiment_score,
                    "date": document.date,
                    "category": document.category,
                    "jurisdiction": document.jurisdiction,
                    "chunking_strategy": strategy,
                    **document.metadata,
                },
            )
            results.append(chunk)

        return results
