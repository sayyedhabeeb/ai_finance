"""
Async PostgreSQL Connection Manager.

Provides a connection pool via asyncpg with convenient query helpers
and transaction support. Designed for use as a singleton attached to
`app.state.db` during the FastAPI lifespan.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

_DEFAULT_DSN = get_settings().database_url


class DatabaseManager:
    """
    Async PostgreSQL connection manager using asyncpg.

    Usage::

        db = DatabaseManager()
        await db.initialize_pool()
        rows = await db.fetch_all("SELECT * FROM users WHERE id = $1", user_id)
        await db.close_pool()
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: float = 60.0,
    ):
        self._dsn = dsn or _DEFAULT_DSN
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout
        self._pool: Optional[asyncpg.pool.Pool] = None

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------

    async def initialize_pool(self) -> None:
        """Create the connection pool and run schema initialization."""
        if self._pool is not None:
            logger.warning("Connection pool already initialised")
            return

        logger.info("Creating PostgreSQL connection pool: min=%d max=%d", self._min_size, self._max_size)
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            command_timeout=self._command_timeout,
        )

        # Verify connectivity
        async with self._pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info("Connected to PostgreSQL: %s", version[:80])

        # Run schema init on first connection
        await self._init_schema()

    async def close_pool(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    async def _init_schema(self) -> None:
        """Run the schema initialization script."""
        async with self._pool.acquire() as conn:
            from backend.database.schemas.init_postgres import init_database
            await init_database(conn)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def pool(self) -> asyncpg.pool.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool not initialised. Call initialize_pool() first.")
        return self._pool

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def execute(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Execute a SQL statement and return the status string.

        Example::

            status = await db.execute(
                "INSERT INTO users (id, email) VALUES ($1, $2)",
                "usr_1", "a@b.com",
            )
        """
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetch_one(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> Optional[asyncpg.Record]:
        """
        Fetch exactly one row. Returns None if no rows match.

        Example::

            row = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
            if row:
                print(row["email"])
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def fetch_all(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> List[asyncpg.Record]:
        """
        Fetch all matching rows.

        Example::

            rows = await db.fetch_all("SELECT * FROM users WHERE risk_profile = $1", "moderate")
        """
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetch_val(
        self,
        query: str,
        *args: Any,
        column: int = 0,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Fetch a single value from the first column of the first row.

        Example::

            count = await db.fetch_val("SELECT COUNT(*) FROM users")
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def fetch_many(
        self,
        query: str,
        *args: Any,
        chunk_size: int = 1000,
        timeout: Optional[float] = None,
    ):
        """
        Iterate over results in chunks (cursor-based). Yields lists of records.

        Useful for large result sets that shouldn't be loaded into memory all at once.

        Example::

            async for chunk in db.fetch_many("SELECT * FROM market_data WHERE symbol = $1", "AAPL", chunk_size=5000):
                process(chunk)
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                async for chunk in conn.cursor(query, *args, prefetch=chunk_size):
                    yield chunk

    # ------------------------------------------------------------------
    # Transaction support
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions.

        Example::

            async with db.transaction():
                await db.execute("UPDATE portfolios SET total_value = $1 WHERE id = $2", val, pid)
                await db.execute("INSERT INTO transactions (...) VALUES (...)", ...)
        """
        if self._pool is None:
            raise RuntimeError("Database pool not initialised")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute_in_transaction(
        self,
        statements: List[Tuple[str, tuple]],
    ) -> List[str]:
        """
        Execute multiple statements in a single transaction.

        Args:
            statements: List of (query, args) tuples.

        Returns:
            List of status strings from each statement.

        Example::

            results = await db.execute_in_transaction([
                ("INSERT INTO users (id, email) VALUES ($1, $2)", ("usr_1", "a@b.com")),
                ("INSERT INTO portfolios (user_id, name) VALUES ($1, $2)", ("usr_1", "Main")),
            ])
        """
        statuses: List[str] = []
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for query, args in statements:
                    status = await conn.execute(query, *args)
                    statuses.append(status)
        return statuses

    # ------------------------------------------------------------------
    # Schema / utility
    # ------------------------------------------------------------------

    async def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the public schema."""
        result = await self.fetch_val(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = $1
            )
            """,
            table_name,
        )
        return bool(result)

    async def get_pool_status(self) -> Dict[str, Any]:
        """Return connection pool statistics."""
        if self._pool is None:
            return {"status": "not initialised"}

        return {
            "status": "healthy",
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size(),
            "idle_size": self._pool.get_idle_size(),
            "size": self._pool.get_size(),
            "freed": self._pool.get_freed(),
        }
