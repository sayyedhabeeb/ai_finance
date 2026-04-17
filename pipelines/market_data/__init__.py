"""
Market data ingestion pipeline.

Fetches OHLCV data from yfinance and NSEpy for Indian equities,
cleans/normalises it, and writes into TimescaleDB.
"""

from pipelines.market_data.ingestion import MarketDataIngestionPipeline

__all__ = ["MarketDataIngestionPipeline"]
