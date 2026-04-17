"""
Market RAG package — Weaviate-based vector store for market documents.

Re-exports:
  - :class:`MarketWeaviateClient`
  - :class:`MarketDocumentIngestionPipeline`
  - :class:`MarketRetriever`
"""

from backend.rag.market.ingestion_pipeline import MarketDocumentIngestionPipeline
from backend.rag.market.retrieval import MarketRetriever
from backend.rag.market.weaviate_client import MarketWeaviateClient

__all__ = [
    "MarketWeaviateClient",
    "MarketDocumentIngestionPipeline",
    "MarketRetriever",
]
