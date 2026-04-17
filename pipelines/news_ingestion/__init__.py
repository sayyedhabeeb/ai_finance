"""
Financial news ingestion pipeline.

Aggregates articles from NewsAPI and RSS feeds (Moneycontrol, Economic
Times, LiveMint, Reuters India), runs FinBERT sentiment analysis, and
stores results in the database and Weaviate.
"""

from pipelines.news_ingestion.fetcher import NewsIngestionPipeline

__all__ = ["NewsIngestionPipeline"]
