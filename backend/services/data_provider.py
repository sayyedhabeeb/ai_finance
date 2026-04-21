"""Data Provider - delivers standardized financial data (simulated/DB) to agents."""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)

class DataProvider:
    """Service to provide financial data for agents to perform analysis."""

    @staticmethod
    def get_portfolio_data(user_id: str = "default") -> Dict[str, Any]:
        """Return simulated portfolio holdings and metrics."""
        # Realistic simulated data for India-specific markets
        holdings = [
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "weight": 0.15, "units": 100, "avg_price": 2450.0},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "weight": 0.12, "units": 50, "avg_price": 3200.0},
            {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "weight": 0.18, "units": 200, "avg_price": 1550.0},
            {"symbol": "INFY.NS", "name": "Infosys", "weight": 0.10, "units": 120, "avg_price": 1400.0},
            {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "weight": 0.15, "units": 300, "avg_price": 850.0},
            {"symbol": "NIFTY_BEES", "name": "Nifty BeES ETF", "weight": 0.30, "units": 1500, "avg_price": 210.0},
        ]
        
        total_value = 2500000.0  # ₹25 Lakhs
        
        return {
            "total_value": total_value,
            "currency": "INR",
            "holdings": holdings,
            "risk_free_rate": 0.07,
            "last_updated": datetime.now().isoformat(),
        }

    @staticmethod
    def get_market_returns(symbols: List[str], days: int = 252) -> Dict[str, List[float]]:
        """Return simulated daily percentage returns for given symbols."""
        results = {}
        for sym in symbols:
            # Generate random but somewhat realistic return series
            # Mean daily return ~0.04% (10% annual), Volatility ~1.2% daily
            vol = random.uniform(0.01, 0.02)
            results[sym] = [random.normalvariate(0.0004, vol) for _ in range(days)]
        return results

    @staticmethod
    def get_sector_data() -> Dict[str, Dict[str, float]]:
        """Return sector-wise performance metrics."""
        return {
            "IT": {"perf_1m": 0.042, "vol": 0.18},
            "BANKING": {"perf_1m": -0.015, "vol": 0.22},
            "PHARMA": {"perf_1m": 0.021, "vol": 0.15},
            "AUTO": {"perf_1m": 0.057, "vol": 0.25},
            "ENERGY": {"perf_1m": 0.008, "vol": 0.19},
        }

    @staticmethod
    def get_tax_context(user_id: str = "default") -> Dict[str, Any]:
        """Return tax-related data for the user."""
        return {
            "annual_income": 1200000,
            "tax_regime": "new",
            "deductions_80c": 150000,
            "deductions_80d": 25000,
            "hra_exempt": 80000,
        }
