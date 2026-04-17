"""
Personal Finance RAG package — pgvector-based vector store for tax and
personal finance documents.

Re-exports:
  - :class:`PersonalFinanceVectorStore`
  - :class:`TaxRuleIngestionPipeline`
  - :class:`PersonalFinanceRetriever`
"""

from backend.rag.personal_finance.pgvector_client import PersonalFinanceVectorStore
from backend.rag.personal_finance.retrieval import PersonalFinanceRetriever
from backend.rag.personal_finance.tax_rule_ingestion import TaxRuleIngestionPipeline

__all__ = [
    "PersonalFinanceVectorStore",
    "TaxRuleIngestionPipeline",
    "PersonalFinanceRetriever",
]
