"""
Embedding service for the RAG system.

Re-exports:
  - :class:`EmbeddingService`
"""

from backend.rag.embeddings.embedder import EmbeddingService

__all__ = ["EmbeddingService"]
