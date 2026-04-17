"""
Market data ingestion pipeline.

Fetches OHLCV data from yfinance (global + NSE with .NS suffix) and
nsepy (for intraday and historical Indian market data), applies
cleaning / adjustment logic, and bulk-writes into TimescaleDB.

Typical usage::

    pipeline = MarketDataIngestionPipeline()
    df = pipeline.fetch_stock_data("RELIANCE.NS", period="1y")
    pipeline.bulk_ingest_to_timescaledb({"RELIANCE.NS": df})
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Popular NSE / BSE symbols (yfinance format with .NS / .BO suffix)
# ---------------------------------------------------------------------------

INDIAN_MARKET_SYMBOLS: list[str] = [
    # NSE large-cap (Nifty 50 constituents, representative set)
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "WIPRO.NS",
    "ULTRACEMCO.NS",
    "TITAN.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "JSWSTEEL.NS",
    "ADANIENT.NS",
    "TATASTEEL.NS",
    "HCLTECH.NS",
    "COALINDIA.NS",
    "BAJAJFINSV.NS",
    "INDUSINDBK.NS",
    "ASIANPAINT.NS",
    "DRREDDY.NS",
    "TECHM.NS",
    "CIPLA.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HEROMOTOCO.NS",
    "DIVISLAB.NS",
    "BRITANNIA.NS",
    "HINDALCO.NS",
    "M_M.NS",
    "SHREECEM.NS",
    "UPL.NS",
    "BPCL.NS",
    "HDFCLIFE.NS",
    "SBILIFE.NS",
    # BSE large-cap
    "RELIANCE.BO",
    "TCS.BO",
    "HDFCBANK.BO",
]

# Sector mapping for Indian stocks
_SYMBOL_SECTOR_MAP: dict[str, str] = {
    "RELIANCE.NS": "OIL_GAS",
    "RELIANCE.BO": "OIL_GAS",
    "TCS.NS": "IT",
    "TCS.BO": "IT",
    "HDFCBANK.NS": "BANKING",
    "HDFCBANK.BO": "BANKING",
    "INFY.NS": "IT",
    "ICICIBANK.NS": "BANKING",
    "HINDUNILVR.NS": "FMCG",
    "SBIN.NS": "BANKING",
    "BHARTIARTL.NS": "TELECOM",
    "ITC.NS": "FMCG",
    "KOTAKBANK.NS": "BANKING",
    "LT.NS": "INFRASTRUCTURE",
    "AXISBANK.NS": "BANKING",
    "BAJFINANCE.NS": "FINANCIAL_SERVICES",
    "MARUTI.NS": "AUTOMOBILE",
    "SUNPHARMA.NS": "PHARMA",
    "TATAMOTORS.NS": "AUTOMOBILE",
    "WIPRO.NS": "IT",
    "ULTRACEMCO.NS": "CEMENT",
    "TITAN.NS": "CONSUMER_GOODS",
    "NESTLEIND.NS": "FMCG",
    "NTPC.NS": "POWER",
    "POWERGRID.NS": "POWER",
    "ONGC.NS": "OIL_GAS",
    "JSWSTEEL.NS": "STEEL",
    "ADANIENT.NS": "CONGLOMERATE",
    "TATASTEEL.NS": "STEEL",
    "HCLTECH.NS": "IT",
    "COALINDIA.NS": "MINING",
    "BAJAJFINSV.NS": "FINANCIAL_SERVICES",
    "INDUSINDBK.NS": "BANKING",
    "ASIANPAINT.NS": "CONSUMER_GOODS",
    "DRREDDY.NS": "PHARMA",
    "TECHM.NS": "IT",
    "CIPLA.NS": "PHARMA",
    "EICHERMOT.NS": "AUTOMOBILE",
    "GRASIM.NS": "CEMENT",
    "HEROMOTOCO.NS": "AUTOMOBILE",
    "DIVISLAB.NS": "PHARMA",
    "BRITANNIA.NS": "FMCG",
    "HINDALCO.NS": "METALS",
    "M_M.NS": "AUTOMOBILE",
    "SHREECEM.NS": "CEMENT",
    "UPL.NS": "CHEMICALS",
    "BPCL.NS": "OIL_GAS",
    "HDFCLIFE.NS": "INSURANCE",
    "SBILIFE.NS": "INSURANCE",
}


class MarketDataIngestionPipeline:
    """Ingests market data from yfinance and nsepy for Indian markets.

    Provides methods for fetching single and multiple stock OHLCV data,
    cleaning/normalising it, and bulk-inserting into TimescaleDB.

    Parameters
    ----------
    db_url : str or None
        PostgreSQL / TimescaleDB connection string.  If *None*, writes
        are disabled and data is returned in-memory only.
    adjust_prices : bool
        Whether to adjust for splits and dividends (default ``True``).
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        adjust_prices: bool = True,
    ) -> None:
        self._db_url = db_url
        self._adjust_prices = adjust_prices
        self._engine = None
        if db_url:
            self._init_db_engine(db_url)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _init_db_engine(self, db_url: str) -> None:
        """Create a SQLAlchemy engine for TimescaleDB."""
        try:
            from sqlalchemy import create_engine

            self._engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                pool_pre_ping=True,
            )
            logger.info("TimescaleDB engine initialised.")
        except Exception as exc:
            logger.error("Failed to initialise DB engine: %s", exc)
            self._engine = None

    def _ensure_market_data_table(self) -> None:
        """Create the ``market_data`` hypertable if it does not exist."""
        if self._engine is None:
            return
        create_sql = """
        CREATE TABLE IF NOT EXISTS market_data (
            symbol    TEXT        NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open      DOUBLE PRECISION,
            high      DOUBLE PRECISION,
            low       DOUBLE PRECISION,
            close     DOUBLE PRECISION,
            volume    BIGINT,
            adj_close DOUBLE PRECISION,
            sector    TEXT,
            source    TEXT        DEFAULT 'yfinance',
            CONSTRAINT pk_market_data PRIMARY KEY (symbol, timestamp)
        );

        SELECT create_hypertable('market_data', 'timestamp',
            if_not_exists => TRUE, migrate_data => TRUE);
        """
        try:
            with self._engine.begin() as conn:
                # Create basic table first (TimescaleDB extension may need the table)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_data (
                        symbol    TEXT        NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        open      DOUBLE PRECISION,
                        high      DOUBLE PRECISION,
                        low       DOUBLE PRECISION,
                        close     DOUBLE PRECISION,
                        volume    BIGINT,
                        adj_close DOUBLE PRECISION,
                        sector    TEXT,
                        source    TEXT        DEFAULT 'yfinance',
                        CONSTRAINT pk_market_data PRIMARY KEY (symbol, timestamp)
                    );
                    """
                )
                # Try to convert to hypertable (ignores if already done)
                try:
                    conn.execute(
                        "SELECT create_hypertable('market_data', 'timestamp', "
                        "if_not_exists => TRUE, migrate_data => TRUE);"
                    )
                except Exception:
                    pass  # Already a hypertable or extension not loaded
            logger.info("market_data table ready.")
        except Exception as exc:
            logger.error("Failed to create market_data table: %s", exc)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single stock symbol via yfinance.

        Parameters
        ----------
        symbol : str
            Yahoo Finance ticker symbol (e.g. ``"RELIANCE.NS"``).
        period : str
            Data period: ``"1d"``, ``"5d"``, ``"1mo"``, ``"3mo"``,
            ``"6mo"``, ``"1y"``, ``"2y"``, ``"5y"``, ``"10y"``,
            ``"ytd"``, ``"max"``.  Ignored when *start* is provided.
        interval : str
            Data interval: ``"1m"``, ``"2m"``, ``"5m"``, ``"15m"``,
            ``"30m"``, ``"60m"``, ``"90m"``, ``"1h"``, ``"1d"``,
            ``"5d"``, ``"1wk"``, ``"1mo"``, ``"3mo"``.
        start : str or None
            Start date (``YYYY-MM-DD``).  Overrides *period*.
        end : str or None
            End date (``YYYY-MM-DD``).  Defaults to today.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``[Open, High, Low, Close, Volume]``
            and a timezone-aware ``DatetimeIndex``.  May include
            ``Adj Close`` if prices are adjusted.

        Raises
        ------
        RuntimeError
            If yfinance fails to return data after retries.
        """
        import yfinance as yf

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                if start:
                    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=self._adjust_prices)
                else:
                    df = ticker.history(period=period, interval=interval, auto_adjust=self._adjust_prices)

                if df.empty:
                    logger.warning(
                        "No data returned for %s (attempt %d/3).", symbol, attempt + 1
                    )
                    last_error = ValueError(f"No data for symbol {symbol}")
                    time.sleep(2 ** attempt)
                    continue

                df = self._clean_stock_data(df, symbol)
                logger.info(
                    "Fetched %d rows for %s [%s] from yfinance.",
                    len(df),
                    symbol,
                    interval,
                )
                return df

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "yfinance error for %s (attempt %d/3): %s",
                    symbol,
                    attempt + 1,
                    exc,
                )
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Failed to fetch data for {symbol} after 3 attempts"
        ) from last_error

    def fetch_from_nsepy(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        series: str = "EQ",
    ) -> pd.DataFrame:
        """Fetch daily OHLCV from nsepy for an NSE equity.

        Parameters
        ----------
        symbol : str
            NSE stock symbol without suffix (e.g. ``"RELIANCE"``).
        start_date : str
            ``YYYY-MM-DD`` format.
        end_date : str or None
            ``YYYY-MM-DD`` format.  Defaults to today.
        series : str
            NSE series (``"EQ"``, ``"BE"``, etc.).

        Returns
        -------
        pd.DataFrame
            Cleaned OHLCV DataFrame with ``DatetimeIndex``.
        """
        try:
            from nsepy import get_history

            end_dt = end_date or datetime.now().strftime("%Y-%m-%d")
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt_parsed = datetime.strptime(end_dt, "%Y-%m-%d")

            df = get_history(
                symbol=symbol,
                start=start_dt,
                end=end_dt_parsed,
                series=series,
            )

            if df.empty:
                logger.warning("nsepy returned no data for %s.", symbol)
                return pd.DataFrame()

            # Standardise columns
            df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
            rename_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "vwap": "VWAP",
                "turnover": "Turnover",
                "trades": "Trades",
                "deliverable_volume": "DeliverableVolume",
                "%deliverable": "PctDeliverable",
            }
            for old, new in rename_map.items():
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)

            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "Date"
            df["Symbol"] = f"{symbol}.NS"
            df["Sector"] = _SYMBOL_SECTOR_MAP.get(f"{symbol}.NS", "UNKNOWN")
            df["Source"] = "nsepy"

            logger.info("Fetched %d rows for %s from nsepy.", len(df), symbol)
            return df

        except ImportError:
            logger.error("nsepy is not installed. Install with: pip install nsepy")
            return pd.DataFrame()
        except Exception as exc:
            logger.error("nsepy fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def fetch_multiple_stocks(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        parallel: bool = True,
        max_workers: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols, optionally in parallel.

        Parameters
        ----------
        symbols : list[str]
            Yahoo Finance ticker symbols.
        period, interval, start, end :
            Passed to :meth:`fetch_stock_data`.
        parallel : bool
            Use thread pool for concurrent requests.
        max_workers : int
            Max threads when *parallel* is ``True``.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of symbol -> cleaned OHLCV DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}

        if not parallel or len(symbols) <= 1:
            for sym in symbols:
                try:
                    results[sym] = self.fetch_stock_data(
                        sym, period=period, interval=interval, start=start, end=end
                    )
                except Exception as exc:
                    logger.error("Failed to fetch %s: %s", sym, exc)
            return results

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_one(sym: str) -> tuple[str, Optional[pd.DataFrame]]:
            try:
                return sym, self.fetch_stock_data(
                    sym, period=period, interval=interval, start=start, end=end
                )
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", sym, exc)
                return sym, None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
            for future in as_completed(futures):
                sym, df = future.result()
                if df is not None:
                    results[sym] = df

        logger.info(
            "Fetched data for %d/%d symbols.", len(results), len(symbols)
        )
        return results

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------

    def _clean_stock_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and normalise a raw OHLCV DataFrame.

        Steps:
        1. Ensure timezone-aware DatetimeIndex.
        2. Strip whitespace from column names.
        3. Forward-fill small gaps, drop large gaps (> 5 consecutive NaNs).
        4. Remove rows where all OHLCV values are NaN.
        5. Clamp negative prices/volumes to 0.
        6. Add symbol and sector metadata columns.
        7. Detect and flag potential corporate actions (split/dividend)
           by checking for > 20 % single-day price moves.
        """
        if df.empty:
            return df

        # 1. Timezone-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        # 2. Column names
        df.columns = [c.strip() for c in df.columns]

        # 3. Handle missing values
        ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if ohlcv_cols:
            # Forward-fill up to 3 consecutive NaN rows (weekends/holidays)
            df[ohlcv_cols] = df[ohlcv_cols].ffill(limit=3)
            # Back-fill up to 1 row
            df[ohlcv_cols] = df[ohlcv_cols].bfill(limit=1)
            # Drop rows where Close is still NaN
            df = df.dropna(subset=["Close"] if "Close" in df.columns else [])

        # 4. Remove entirely empty rows
        df = df.dropna(how="all")

        # 5. Clamp negatives
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # Ensure High >= Low (correct data errors)
        if "High" in df.columns and "Low" in df.columns:
            mask = df["High"] < df["Low"]
            if mask.any():
                logger.warning(
                    "Correcting %d rows where High < Low for %s.", mask.sum(), symbol
                )
                df.loc[mask, ["High", "Low"]] = df.loc[mask, ["Low", "High"]].values

        # 6. Metadata columns
        df["Symbol"] = symbol
        df["Sector"] = _SYMBOL_SECTOR_MAP.get(symbol, "UNKNOWN")
        df["Source"] = "yfinance"

        # 7. Corporate action detection flag
        if "Close" in df.columns:
            pct_change = df["Close"].pct_change().abs()
            df["CorporateActionFlag"] = (pct_change > 0.20).astype(int)

        # Reset index to column for DB insertion
        df = df.reset_index()
        df.rename(columns={"index": "timestamp", "Date": "timestamp"}, inplace=True)
        if "timestamp" not in df.columns and df.columns[0] not in ohlcv_cols:
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

        return df

    # ------------------------------------------------------------------
    # Bulk ingestion to TimescaleDB
    # ------------------------------------------------------------------

    def bulk_ingest_to_timescaledb(self, data: dict[str, pd.DataFrame]) -> int:
        """Write multiple symbol DataFrames into TimescaleDB.

        Uses upsert (``INSERT … ON CONFLICT``) so that re-ingesting
 overlapping date ranges is safe.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Mapping of symbol -> OHLCV DataFrame.

        Returns
        -------
        int
            Total number of rows successfully inserted/upserted.
        """
        if self._engine is None:
            logger.error("No database connection. Set db_url when initialising.")
            return 0

        self._ensure_market_data_table()

        total_rows = 0
        for symbol, df in data.items():
            rows = self._upsert_symbol(symbol, df)
            total_rows += rows

        logger.info("Bulk ingest complete: %d total rows for %d symbols.", total_rows, len(data))
        return total_rows

    def _upsert_symbol(self, symbol: str, df: pd.DataFrame) -> int:
        """Upsert a single symbol's DataFrame into ``market_data``."""
        if df.empty:
            return 0

        try:
            # Normalise column names for DB
            col_map = {
                "timestamp": "timestamp",
                "Timestamp": "timestamp",
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adj_close",
                "Sector": "sector",
                "Source": "source",
            }
            db_df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # Ensure required columns
            if "symbol" not in db_df.columns:
                db_df["symbol"] = symbol
            if "sector" not in db_df.columns:
                db_df["sector"] = _SYMBOL_SECTOR_MAP.get(symbol, "UNKNOWN")
            if "source" not in db_df.columns:
                db_df["source"] = "yfinance"

            # Convert timestamp to datetime
            if "timestamp" in db_df.columns:
                db_df["timestamp"] = pd.to_datetime(db_df["timestamp"], utc=True)

            # Select only the columns we need
            target_cols = [
                "symbol", "timestamp", "open", "high", "low", "close",
                "volume", "adj_close", "sector", "source",
            ]
            available_cols = [c for c in target_cols if c in db_df.columns]
            db_df = db_df[available_cols]

            # Drop fully duplicate rows
            db_df = db_df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")

            # Upsert using pandas to_sql with a temp table approach
            with self._engine.begin() as conn:
                # Write to temp table
                temp_table = f"temp_market_data_{symbol.replace('.', '_').lower()}"
                db_df.to_sql(temp_table, conn, if_exists="replace", index=False)

                # Upsert from temp table
                merge_sql = f"""
                INSERT INTO market_data (symbol, timestamp, open, high, low, close,
                                         volume, adj_close, sector, source)
                SELECT symbol, timestamp, open, high, low, close,
                       volume, adj_close, sector, source
                FROM {temp_table}
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open      = EXCLUDED.open,
                    high      = EXCLUDED.high,
                    low       = EXCLUDED.low,
                    close     = EXCLUDED.close,
                    volume    = EXCLUDED.volume,
                    adj_close = EXCLUDED.adj_close,
                    sector    = EXCLUDED.sector,
                    source    = EXCLUDED.source;
                """
                conn.execute(merge_sql)

                # Drop temp table
                conn.execute(f"DROP TABLE IF EXISTS {temp_table};")

            n_rows = len(db_df)
            logger.info("Upserted %d rows for %s.", n_rows, symbol)
            return n_rows

        except Exception as exc:
            logger.error("Failed to upsert %s: %s", symbol, exc)
            return 0

    # ------------------------------------------------------------------
    # Convenience: fetch latest data for all tracked Indian symbols
    # ------------------------------------------------------------------

    def fetch_all_indian_symbols(
        self,
        period: str = "1y",
        interval: str = "1d",
        parallel: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for all symbols in :data:`INDIAN_MARKET_SYMBOLS`.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of symbol -> OHLCV DataFrame.
        """
        return self.fetch_multiple_stocks(
            symbols=INDIAN_MARKET_SYMBOLS,
            period=period,
            interval=interval,
            parallel=parallel,
        )

    # ------------------------------------------------------------------
    # Scheduled ingestion helpers (used by PipelineScheduler)
    # ------------------------------------------------------------------

    def scheduled_ingest_trading_hours(self, period: str = "1d") -> int:
        """Ingest today's data for all Indian symbols during trading hours.

        Designed to be called by Prefect every 15 minutes during
        NSE trading hours (09:15 – 15:30 IST).

        Returns
        -------
        int
            Total rows ingested.
        """
        now = datetime.now()
        ist_now = now.astimezone(
            __import__("zoneinfo").ZoneInfo("Asia/Kolkata")
        )
        hour = ist_now.hour
        minute = ist_now.minute
        ist_minutes = hour * 60 + minute

        # NSE trading window: 9:15 – 15:30 IST
        if ist_minutes < 9 * 60 + 15 or ist_minutes > 15 * 60 + 30:
            logger.info("Outside NSE trading hours (%02d:%02d IST). Skipping.", hour, minute)
            return 0

        data = self.fetch_all_indian_symbols(period=period, interval="15m", parallel=True)
        if self._engine:
            return self.bulk_ingest_to_timescaledb(data)
        return sum(len(df) for df in data.values())

    def historical_backfill(
        self,
        symbols: Optional[list[str]] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
    ) -> int:
        """Backfill historical data for specified symbols.

        Parameters
        ----------
        symbols : list[str] or None
            Symbols to backfill.  Defaults to :data:`INDIAN_MARKET_SYMBOLS`.
        start_date : str
            Start date in ``YYYY-MM-DD`` format.
        end_date : str or None
            End date.  Defaults to today.

        Returns
        -------
        int
            Total rows inserted.
        """
        target_symbols = symbols or INDIAN_MARKET_SYMBOLS
        data = self.fetch_multiple_stocks(
            symbols=target_symbols,
            start=start_date,
            end=end_date,
            parallel=True,
        )
        if self._engine:
            return self.bulk_ingest_to_timescaledb(data)
        return sum(len(df) for df in data.values())
