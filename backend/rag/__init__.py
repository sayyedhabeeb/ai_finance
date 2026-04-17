"""
DUAL-LAYER RAG system for AI Financial Brain.

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │                 RAG System                           │
  │                                                      │
  │  ┌─────────────────────┐  ┌───────────────────────┐  │
  │  │   MARKET RAG        │  │  PERSONAL FINANCE RAG  │  │
  │  │   (Weaviate)        │  │  (pgvector)           │  │
  │  │                     │  │                       │  │
  │  │  • News articles    │  │  • Tax rules (80C...) │  │
  │  │  • Research reports │  │  • Tax slabs          │  │
  │  │  • SEC/NSE filings  │  │  • GST rules          │  │
  │  │  • Earnings calls   │  │  • Deductions         │  │
  │  │  • Market data      │  │  • Capital gains      │  │
  │  └─────────────────────┘  └───────────────────────┘  │
  │                                                      │
  │  ┌──────────────────────────────────────────────────┐ │
  │  │  Shared: Chunking + Embeddings + Models          │ │
  │  └──────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────┘

Re-exports:
  - **Models**: ``RAGDocument``, ``RetrievalResult``, ``RetrievalResponse``
  - **Chunking**: ``DocumentChunker``
  - **Embeddings**: ``EmbeddingService``
  - **Market RAG**: ``MarketWeaviateClient``, ``MarketDocumentIngestionPipeline``, ``MarketRetriever``
  - **Personal Finance RAG**: ``PersonalFinanceVectorStore``, ``TaxRuleIngestionPipeline``, ``PersonalFinanceRetriever``
"""

from backend.rag.chunking.strategies import DocumentChunker
from backend.rag.embeddings.embedder import EmbeddingService
from backend.rag.market.ingestion_pipeline import MarketDocumentIngestionPipeline
from backend.rag.market.retrieval import MarketRetriever
from backend.rag.market.weaviate_client import MarketWeaviateClient
from backend.rag.models import (
    ChunkedDocument,
    DocumentType,
    RAGDocument,
    RetrievalResponse,
    RetrievalResult,
    SourceType,
)
from backend.rag.personal_finance.pgvector_client import PersonalFinanceVectorStore
from backend.rag.personal_finance.retrieval import PersonalFinanceRetriever
from backend.rag.personal_finance.tax_rule_ingestion import TaxRuleIngestionPipeline

__all__ = [
    # Models
    "RAGDocument",
    "RetrievalResult",
    "RetrievalResponse",
    "ChunkedDocument",
    "DocumentType",
    "SourceType",
    # Chunking
    "DocumentChunker",
    # Embeddings
    "EmbeddingService",
    # Market RAG (Weaviate)
    "MarketWeaviateClient",
    "MarketDocumentIngestionPipeline",
    "MarketRetriever",
    # Personal Finance RAG (pgvector)
    "PersonalFinanceVectorStore",
    "TaxRuleIngestionPipeline",
    "PersonalFinanceRetriever",
]
