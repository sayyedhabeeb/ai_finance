"""
AI Financial Brain — Data Pipelines

Automated data ingestion pipelines for market data, news, and RAG updates.
Scheduled and orchestrated via Prefect.
"""

from pipelines.market_data.ingestion import MarketDataIngestionPipeline
from pipelines.news_ingestion.fetcher import NewsIngestionPipeline
from pipelines.rag_updates.updater import RAGUpdatePipeline
from pipelines.scheduled.scheduler import PipelineScheduler

__all__ = [
    "MarketDataIngestionPipeline",
    "NewsIngestionPipeline",
    "RAGUpdatePipeline",
    "PipelineScheduler",
]
