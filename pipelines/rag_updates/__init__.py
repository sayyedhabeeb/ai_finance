"""
RAG document update pipeline.

Keeps the vector stores (Weaviate / pgvector) in sync with fresh
research reports, regulatory filings, and tax/regulation updates.
"""

from pipelines.rag_updates.updater import RAGUpdatePipeline

__all__ = ["RAGUpdatePipeline"]
