"""
Shared data models for the RAG system.

Defines input documents, retrieval results, and common types used
across both Market RAG (Weaviate) and Personal Finance RAG (pgvector).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    """Types of documents in the RAG system."""

    # Market documents
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    NEWS_ARTICLE = "news_article"
    RESEARCH_REPORT = "research_report"
    SEC_FILING = "sec_filing"
    NSE_FILING = "nse_filing"
    MARKET_COMMENTARY = "market_commentary"
    REGULATORY_FILING = "regulatory_filing"

    # Personal finance documents
    TAX_RULE = "tax_rule"
    TAX_SECTION = "tax_section"
    DEDUCTION_RULE = "deduction_rule"
    TAX_SLAB = "tax_slab"
    GST_RULE = "gst_rule"
    INVESTMENT_RULE = "investment_rule"
    RETIREMENT_RULE = "retirement_rule"
    INSURANCE_RULE = "insurance_rule"
    LOAN_RULE = "loan_rule"

    # General
    WEB_ARTICLE = "web_article"
    TEXT_DOCUMENT = "text_document"
    PDF_DOCUMENT = "pdf_document"
    CSV_DATA = "csv_data"


class SourceType(str, Enum):
    """Origin sources for documents."""

    URL = "url"
    FILE_UPLOAD = "file_upload"
    API_FETCH = "api_fetch"
    MANUAL_ENTRY = "manual_entry"
    SCHEDULED_INGESTION = "scheduled_ingestion"


# ──────────────────────────────────────────────────────────────
# Input Models
# ──────────────────────────────────────────────────────────────

class RAGDocument(BaseModel):
    """Input document for ingestion into the RAG system.

    This is the standard format accepted by all ingestion pipelines.
    """

    content: str = Field(..., min_length=1, description="Full text content of the document.")
    title: str = Field(default="", description="Title of the document.")
    source: str = Field(default="", description="Source URL or file path.")
    doc_type: DocumentType = Field(default=DocumentType.TEXT_DOCUMENT)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata.")

    # Optional fields for market documents
    ticker_symbols: list[str] = Field(default_factory=list, description="Associated ticker symbols.")
    sector: str = Field(default="", description="Industry sector.")
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0, description="Sentiment score [-1, 1].")
    date: str = Field(default="", description="Document date in YYYY-MM-DD format.")

    # Optional fields for personal finance documents
    category: str = Field(default="", description="Category (e.g. 'tax', 'insurance', 'investment').")
    jurisdiction: str = Field(default="IN", description="Jurisdiction code (e.g. 'IN', 'US').")

    # Internal tracking
    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique document ID.")
    checksum: str = Field(default="", description="SHA-256 checksum for deduplication.")

    def model_post_init(self, __context: Any) -> None:
        """Generate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of content + source for dedup."""
        raw = f"{self.content}:{self.source}:{self.title}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ChunkedDocument(BaseModel):
    """A single chunk of a larger document, ready for embedding."""

    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    document_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str = Field(..., min_length=1)
    chunk_index: int = Field(default=0, ge=0)
    title: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Output Models
# ──────────────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    """A single result returned from the RAG retriever."""

    content: str = Field(..., description="Retrieved text content.")
    doc_id: str = Field(default="", description="Source document ID.")
    chunk_id: str = Field(default="", description="Source chunk ID.")
    title: str = Field(default="", description="Document title.")
    source: str = Field(default="", description="Source URL or path.")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score [0, 1].")
    doc_type: str = Field(default="", description="Type of document.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlight: str = Field(default="", description="Highlighted excerpt snippet.")

    def to_context_str(self, max_length: int = 2000) -> str:
        """Format as a context string for LLM prompts."""
        header = f"[{self.doc_type}] {self.title}" if self.title else f"[{self.doc_type}]"
        source_line = f"  Source: {self.source}" if self.source else ""
        return f"{header}\n{source_line}\n  Score: {self.score:.3f}\n  {self.content[:max_length]}"


class RetrievalResponse(BaseModel):
    """Aggregated response from a retrieval operation."""

    query: str = Field(default="")
    results: list[RetrievalResult] = Field(default_factory=list)
    total_found: int = Field(default=0)
    retrieval_time_ms: float = Field(default=0.0)
    sub_questions: list[str] = Field(default_factory=list)

    def get_context_string(self, max_results: int = 10, max_chars_per_result: int = 2000) -> str:
        """Concatenate top results into a single context block for LLM consumption."""
        selected = self.results[:max_results]
        blocks: list[str] = []
        for i, r in enumerate(selected, 1):
            header = f"[Result {i}] Score={r.score:.3f}"
            if r.title:
                header += f" | Title: {r.title}"
            if r.source:
                header += f" | Source: {r.source}"
            blocks.append(f"{header}\n{r.content[:max_chars_per_result]}")
        return "\n\n---\n\n".join(blocks)

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0
