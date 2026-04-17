"""
Episodic memory store for specific interaction events and outcomes.

Backed by PostgreSQL (via asyncpg).  Stores discrete events (queries,
decisions, portfolio changes, alerts, etc.) with rich metadata enabling
temporal, sentiment-based, and similarity-based recall.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helper (re-uses the same approach as semantic_memory)
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSIONS = 1536
_LOCAL_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_local_embedder: Any | None = None


def _get_local_embedder() -> Any:
    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer

        _local_embedder = SentenceTransformer(_LOCAL_EMBEDDING_MODEL_ID)
    return _local_embedder


def _fit_dimensions(vector: list[float], target_dim: int) -> list[float]:
    if len(vector) == target_dim:
        return vector
    if len(vector) > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))


async def _generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    try:
        embedder = _get_local_embedder()
        vector = embedder.encode(text, normalize_embeddings=True).tolist()
        return _fit_dimensions(vector, EMBEDDING_DIMENSIONS)
    except Exception as exc:
        logger.warning("Local embedding generation failed, using deterministic fallback: %s", exc)
        import hashlib

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = __import__("random").Random(seed)
        return [rng.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSIONS)]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single episodic-memory record."""

    id: str
    user_id: str
    event_type: str
    summary: str
    outcome: str
    sentiment: str
    entities: dict[str, Any]
    context: dict[str, Any]
    timestamp: datetime
    importance_score: float
    similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "summary": self.summary,
            "outcome": self.outcome,
            "sentiment": self.sentiment,
            "entities": self.entities,
            "context": self.context,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "importance_score": self.importance_score,
            "similarity": self.similarity,
        }

    @classmethod
    def from_row(cls, row: Record) -> "MemoryEntry":  # type: ignore[type-arg]
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            event_type=row["event_type"],
            summary=row["summary"],
            outcome=row["outcome"],
            sentiment=row["sentiment"],
            entities=row["entities"] or {},
            context=row["context"] or {},
            timestamp=row["timestamp"],
            importance_score=float(row["importance_score"]),
            similarity=float(row.get("similarity", 0.0)),
        )


# ---------------------------------------------------------------------------
# EpisodicMemoryStore
# ---------------------------------------------------------------------------


class EpisodicMemoryStore:
    """Episodic memory for storing specific interaction events and outcomes.

    Each record captures a discrete event (e.g. a stock query, a portfolio
    rebalance, a risk alert) together with its outcome, detected sentiment,
    extracted entities, and surrounding context.  Retrieval supports:
    * semantic similarity to an arbitrary query
    * time-range filtering
    * sentiment filtering
    * full chronological user-journey reconstruction
    """

    def __init__(
        self,
        db_dsn: str = "postgresql://postgres:postgres@localhost:5432/ai_financial_brain",
        min_connection_pool_size: int = 2,
        max_connection_pool_size: int = 10,
    ) -> None:
        """Initialise the store.

        Args:
            db_dsn: PostgreSQL connection string.
            min_connection_pool_size: Minimum pool size.
            max_connection_pool_size: Maximum pool size.
        """
        self._db_dsn = db_dsn
        self._pool: asyncpg.Pool | None = None
        self._min_pool = min_connection_pool_size
        self._max_pool = max_connection_pool_size

        logger.info("EpisodicMemoryStore initialised  dsn=%s...", db_dsn[:50])

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("EpisodicMemoryStore is not initialised. Call `ensure_tables_exist()` first.")
        return self._pool

    async def ensure_tables_exist(self) -> None:
        """Create the ``episodic_memories`` table and indexes if they do not
        already exist."""
        self._pool = await asyncpg.create_pool(
            self._db_dsn,
            min_size=self._min_pool,
            max_size=self._max_pool,
        )
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id          TEXT NOT NULL,
                    event_type       TEXT NOT NULL,
                    summary          TEXT NOT NULL,
                    outcome          TEXT NOT NULL DEFAULT '',
                    sentiment        TEXT NOT NULL DEFAULT 'neutral',
                    entities         JSONB NOT NULL DEFAULT '{}'::jsonb,
                    context          JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding        vector(1536),
                    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    importance_score FLOAT NOT NULL DEFAULT 0.5
                );

                -- Index for user-scoped queries
                CREATE INDEX IF NOT EXISTS idx_episodic_user
                    ON episodic_memories (user_id);

                -- Index for event-type queries
                CREATE INDEX IF NOT EXISTS idx_episodic_event_type
                    ON episodic_memories (user_id, event_type);

                -- Index for time-range queries
                CREATE INDEX IF NOT EXISTS idx_episodic_timestamp
                    ON episodic_memories (user_id, timestamp);

                -- Index for sentiment queries
                CREATE INDEX IF NOT EXISTS idx_episodic_sentiment
                    ON episodic_memories (user_id, sentiment);

                -- HNSW index for vector similarity search
                CREATE INDEX IF NOT EXISTS idx_episodic_embedding
                    ON episodic_memories USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 200);
                """
            )
        logger.info("episodic_memories table ready")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("EpisodicMemoryStore connection closed")

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    async def record_event(
        self,
        user_id: str,
        event_type: str,
        summary: str,
        outcome: str = "",
        sentiment: str = "neutral",
        entities: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """Persist a new episodic event.

        Args:
            user_id: Owner of the event.
            event_type: Category (e.g. ``query``, ``trade``, ``alert``,
                        ``insight``, ``portfolio_rebalance``).
            summary: Human-readable summary of the event.
            outcome: Result or outcome of the event.
            sentiment: Detected sentiment (``positive``, ``negative``,
                       ``neutral``).
            entities: Extracted named entities (tickers, amounts, dates, etc.).
            context: Additional context (full query, agent response, etc.).
            importance: Importance score in ``[0.0, 1.0]``.

        Returns:
            The newly-created event ID (UUID string).
        """
        pool = await self._get_pool()
        event_id = str(uuid.uuid4())
        entities = entities or {}
        context = context or {}
        now = datetime.now(timezone.utc)

        # Build a text representation for embedding
        embed_text = f"{event_type}: {summary}. Outcome: {outcome}"
        embedding = await _generate_embedding(embed_text)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episodic_memories
                    (id, user_id, event_type, summary, outcome, sentiment,
                     entities, context, embedding, timestamp, importance_score)
                VALUES
                    ($1, $2, $3, $4, $5, $6,
                     $7::jsonb, $8::jsonb, $9::vector, $10, $11)
                """,
                event_id,
                user_id,
                event_type,
                summary,
                outcome,
                sentiment,
                json.dumps(entities),
                json.dumps(context),
                json.dumps(embedding),
                now,
                importance,
            )

        logger.info(
            "Episodic event recorded  id=%s  user=%s  type=%s  sentiment=%s",
            event_id,
            user_id,
            event_type,
            sentiment,
        )
        return event_id

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall_events(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Recall events most semantically similar to *query*.

        Args:
            user_id: Scope the recall to this user.
            query: Natural-language query.
            limit: Maximum number of events to return.

        Returns:
            List of :class:`MemoryEntry` sorted by descending similarity.
        """
        pool = await self._get_pool()
        query_embedding = await _generate_embedding(query)
        embedding_json = json.dumps(query_embedding)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id, user_id, event_type, summary, outcome, sentiment,
                    entities, context, timestamp, importance_score,
                    1 - (embedding <=> $2::vector) AS similarity
                FROM episodic_memories
                WHERE user_id = $1
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                user_id,
                embedding_json,
                limit,
            )

        return [MemoryEntry.from_row(r) for r in rows]

    async def recall_by_time_range(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None,
    ) -> list[MemoryEntry]:
        """Recall events that occurred within a time range.

        Args:
            user_id: Scope to this user.
            start: Start of the range (inclusive).
            end: End of the range (inclusive).
            event_type: Optional event-type filter.

        Returns:
            Chronologically ordered list of events.
        """
        pool = await self._get_pool()

        if event_type:
            query = """
                SELECT
                    id, user_id, event_type, summary, outcome, sentiment,
                    entities, context, timestamp, importance_score,
                    0.0 AS similarity
                FROM episodic_memories
                WHERE user_id = $1
                  AND timestamp >= $2
                  AND timestamp <= $3
                  AND event_type = $4
                ORDER BY timestamp DESC
                """
            params: tuple[Any, ...] = (user_id, start, end, event_type)
        else:
            query = """
                SELECT
                    id, user_id, event_type, summary, outcome, sentiment,
                    entities, context, timestamp, importance_score,
                    0.0 AS similarity
                FROM episodic_memories
                WHERE user_id = $1
                  AND timestamp >= $2
                  AND timestamp <= $3
                ORDER BY timestamp DESC
                """
            params = (user_id, start, end)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [MemoryEntry.from_row(r) for r in rows]

    async def recall_by_sentiment(
        self,
        user_id: str,
        sentiment: str,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Recall events matching a specific sentiment.

        Args:
            user_id: Scope to this user.
            sentiment: Sentiment to filter by (``positive``, ``negative``,
                       ``neutral``).
            limit: Maximum results.

        Returns:
            List of matching events ordered by most recent first.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id, user_id, event_type, summary, outcome, sentiment,
                    entities, context, timestamp, importance_score,
                    0.0 AS similarity
                FROM episodic_memories
                WHERE user_id = $1
                  AND sentiment = $2
                ORDER BY timestamp DESC
                LIMIT $3
                """,
                user_id,
                sentiment,
                limit,
            )

        return [MemoryEntry.from_row(r) for r in rows]

    async def get_user_journey(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Reconstruct the chronological user-journey timeline.

        Returns a flat, chronologically-ordered list of all episodic events
        for a user, suitable for feeding into an LLM as context.

        Args:
            user_id: Unique user identifier.
            limit: Maximum number of events to return.

        Returns:
            List of dicts in chronological order (oldest → newest).
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id, user_id, event_type, summary, outcome, sentiment,
                    entities, context, timestamp, importance_score
                FROM episodic_memories
                WHERE user_id = $1
                ORDER BY timestamp ASC
                LIMIT $2
                """,
                user_id,
                limit,
            )

        return [
            {
                "id": str(r["id"]),
                "event_type": r["event_type"],
                "summary": r["summary"],
                "outcome": r["outcome"],
                "sentiment": r["sentiment"],
                "entities": r["entities"] or {},
                "context": r["context"] or {},
                "timestamp": r["timestamp"].isoformat() if r["timestamp"] else None,
                "importance_score": float(r["importance_score"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_event(self, event_id: str) -> bool:
        """Delete a single episodic event.

        Args:
            event_id: UUID of the event.

        Returns:
            ``True`` if the event was deleted.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM episodic_memories WHERE id = $1::uuid",
                event_id,
            )

        deleted = result == "DELETE 1"
        if deleted:
            logger.info("Episodic event deleted  id=%s", event_id)
        return deleted

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_event_count(self, user_id: str) -> int:
        """Return the total number of episodic events for a user."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM episodic_memories WHERE user_id = $1",
                user_id,
            )
        return int(count) if count is not None else 0

    async def get_sentiment_summary(self, user_id: str) -> dict[str, int]:
        """Return a count of events grouped by sentiment for a user."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT sentiment, COUNT(*) AS cnt
                FROM episodic_memories
                WHERE user_id = $1
                GROUP BY sentiment
                ORDER BY cnt DESC
                """,
                user_id,
            )

        return {r["sentiment"]: int(r["cnt"]) for r in rows}
