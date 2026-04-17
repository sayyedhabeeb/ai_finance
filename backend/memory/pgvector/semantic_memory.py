"""
Long-term semantic memory backed by PostgreSQL + pgvector.

Provides vector-similarity search over stored memories, tag-based retrieval,
importance scoring, and memory consolidation (merging semantically-duplicate
memories).
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
# Default configuration
# ---------------------------------------------------------------------------
EMBEDDING_DIMENSIONS = 1536
SIMILARITY_THRESHOLD = 0.85  # cosine similarity above which memories are considered duplicates
CONSOLIDATION_BATCH_SIZE = 50
_LOCAL_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_local_embedder: Any | None = None

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single semantic-memory record."""

    id: str
    user_id: str
    content: str
    memory_type: str
    importance_score: float
    access_count: int
    tags: list[str]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    similarity: float = 0.0  # populated after a similarity search

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "similarity": self.similarity,
        }

    @classmethod
    def from_row(cls, row: Record) -> "MemoryEntry":  # type: ignore[type-arg]
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance_score=float(row["importance_score"]),
            access_count=int(row["access_count"]),
            tags=row["tags"] or [],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            similarity=float(row.get("similarity", 0.0)),
        )


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


async def _generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Uses a local sentence-transformers model. Falls back to a
    deterministic dummy vector in development / test environments.
    """
    try:
        global _local_embedder
        if _local_embedder is None:
            from sentence_transformers import SentenceTransformer

            _local_embedder = SentenceTransformer(_LOCAL_EMBEDDING_MODEL_ID)

        vector = _local_embedder.encode(text, normalize_embeddings=True).tolist()
        if len(vector) > EMBEDDING_DIMENSIONS:
            return vector[:EMBEDDING_DIMENSIONS]
        if len(vector) < EMBEDDING_DIMENSIONS:
            return vector + [0.0] * (EMBEDDING_DIMENSIONS - len(vector))
        return vector
    except Exception as exc:
        logger.warning("Local embedding generation failed, using fallback: %s", exc)
        # Deterministic fallback based on text hash (for testing / dev only)
        import hashlib

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = __import__("random").Random(seed)
        return [rng.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSIONS)]


# ---------------------------------------------------------------------------
# SemanticMemoryStore
# ---------------------------------------------------------------------------


class SemanticMemoryStore:
    """Long-term semantic memory using pgvector for similarity-based retrieval.

    Stores memories as rows in ``semantic_memories`` where each row carries a
    1536-dimensional embedding vector.  Retrieval is performed via cosine
    distance (``<=>``) which pgvector accelerates with an IVFFlat or HNSW
    index.
    """

    def __init__(
        self,
        db_dsn: str = "postgresql://postgres:postgres@localhost:5432/ai_financial_brain",
        min_connection_pool_size: int = 2,
        max_connection_pool_size: int = 10,
    ) -> None:
        """Initialise the store.

        Args:
            db_dsn: PostgreSQL connection string (asyncpg-compatible).
            min_connection_pool_size: Minimum pool size.
            max_connection_pool_size: Maximum pool size.
        """
        self._db_dsn = db_dsn
        self._pool: asyncpg.Pool | None = None
        self._min_pool = min_connection_pool_size
        self._max_pool = max_connection_pool_size

        logger.info("SemanticMemoryStore initialised  dsn=%s...", db_dsn[:50])

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("SemanticMemoryStore is not initialised. Call `ensure_tables_exist()` first.")
        return self._pool

    async def ensure_tables_exist(self) -> None:
        """Create the ``semantic_memories`` table and supporting index if they
        do not already exist."""
        self._pool = await asyncpg.create_pool(
            self._db_dsn,
            min_size=self._min_pool,
            max_size=self._max_pool,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id         TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    embedding       vector(1536),
                    memory_type     TEXT NOT NULL DEFAULT 'general',
                    importance_score FLOAT NOT NULL DEFAULT 0.5,
                    access_count    INT NOT NULL DEFAULT 0,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    tags            TEXT[] NOT NULL DEFAULT '{}',
                    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                -- Index for user-scoped queries
                CREATE INDEX IF NOT EXISTS idx_sem_memories_user
                    ON semantic_memories (user_id);

                -- Index for tag-based queries (GIN on text array)
                CREATE INDEX IF NOT EXISTS idx_sem_memories_tags
                    ON semantic_memories USING GIN (tags);

                -- HNSW index for fast cosine similarity search
                CREATE INDEX IF NOT EXISTS idx_sem_memories_embedding
                    ON semantic_memories USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 200);
                """
            )
        logger.info("semantic_memories table ready")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("SemanticMemoryStore connection closed")

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store(
        self,
        user_id: str,
        content: str,
        memory_type: str = "general",
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Persist a new semantic memory.

        Args:
            user_id: Owner of the memory.
            content: Free-text content.
            memory_type: Category / type of the memory (e.g. ``preference``,
                         ``fact``, ``insight``).
            tags: Optional list of string tags for filtering.
            importance: Score in ``[0.0, 1.0]`` indicating importance.
            metadata: Arbitrary JSON-serialisable metadata.

        Returns:
            The newly-created memory ID (UUID string).
        """
        pool = await self._get_pool()
        memory_id = str(uuid.uuid4())
        tags = tags or []
        metadata = metadata or {}
        now = datetime.now(timezone.utc)

        embedding = await _generate_embedding(content)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO semantic_memories
                    (id, user_id, content, embedding, memory_type,
                     importance_score, tags, metadata, created_at, updated_at)
                VALUES
                    ($1, $2, $3, $4::vector, $5, $6, $7, $8::jsonb, $9, $10)
                """,
                memory_id,
                user_id,
                content,
                json.dumps(embedding),
                memory_type,
                importance,
                tags,
                json.dumps(metadata),
                now,
                now,
            )

        logger.info(
            "Memory stored  id=%s  user=%s  type=%s",
            memory_id,
            user_id,
            memory_type,
        )
        return memory_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve memories most semantically similar to *query*.

        Args:
            user_id: Scope the search to this user.
            query: Natural-language query.
            top_k: Number of results to return.

        Returns:
            List of :class:`MemoryEntry` instances sorted by descending
            similarity (highest first).
        """
        pool = await self._get_pool()
        query_embedding = await _generate_embedding(query)
        embedding_json = json.dumps(query_embedding)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id, user_id, content, memory_type, importance_score,
                    access_count, tags, metadata, created_at, updated_at,
                    1 - (embedding <=> $2::vector) AS similarity
                FROM semantic_memories
                WHERE user_id = $1
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                user_id,
                embedding_json,
                top_k,
            )

            # Bump access_count for returned memories
            ids = [r["id"] for r in rows]
            if ids:
                await conn.execute(
                    """
                    UPDATE semantic_memories
                    SET access_count = access_count + 1,
                        updated_at   = NOW()
                    WHERE id = ANY($1::uuid[])
                    """,
                    ids,
                )

        return [MemoryEntry.from_row(r) for r in rows]

    async def search_by_tags(
        self,
        user_id: str,
        tags: list[str],
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Retrieve memories matching any of the given tags.

        Args:
            user_id: Scope to this user.
            tags: Tags to match (OR logic).
            limit: Maximum results.

        Returns:
            List of matching :class:`MemoryEntry` instances.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id, user_id, content, memory_type, importance_score,
                    access_count, tags, metadata, created_at, updated_at,
                    0.0 AS similarity
                FROM semantic_memories
                WHERE user_id = $1
                  AND tags && $2::text[]
                ORDER BY importance_score DESC, updated_at DESC
                LIMIT $3
                """,
                user_id,
                tags,
                limit,
            )

        return [MemoryEntry.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Importance
    # ------------------------------------------------------------------

    async def update_importance(self, memory_id: str, delta: float) -> None:
        """Adjust a memory's importance score.

        Args:
            memory_id: UUID of the memory to update.
            delta: Amount to add to the current score (clamped to ``[0.0, 1.0]``).
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE semantic_memories
                SET importance_score = LEAST(1.0, GREATEST(0.0, importance_score + $2)),
                    updated_at       = NOW()
                WHERE id = $1::uuid
                """,
                memory_id,
                delta,
            )

        logger.debug("Importance updated  id=%s  delta=%.3f", memory_id, delta)

    # ------------------------------------------------------------------
    # Retrieve / delete
    # ------------------------------------------------------------------

    async def get_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """List memories for a user, optionally filtered by type.

        Args:
            user_id: Scope to this user.
            memory_type: Optional type filter.
            limit: Maximum results.

        Returns:
            List of :class:`MemoryEntry` instances ordered by importance.
        """
        pool = await self._get_pool()

        if memory_type:
            query = """
                SELECT
                    id, user_id, content, memory_type, importance_score,
                    access_count, tags, metadata, created_at, updated_at,
                    0.0 AS similarity
                FROM semantic_memories
                WHERE user_id = $1 AND memory_type = $2
                ORDER BY importance_score DESC, updated_at DESC
                LIMIT $3
            """
            params: tuple[Any, ...] = (user_id, memory_type, limit)
        else:
            query = """
                SELECT
                    id, user_id, content, memory_type, importance_score,
                    access_count, tags, metadata, created_at, updated_at,
                    0.0 AS similarity
                FROM semantic_memories
                WHERE user_id = $1
                ORDER BY importance_score DESC, updated_at DESC
                LIMIT $2
            """
            params = (user_id, limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [MemoryEntry.from_row(r) for r in rows]

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by ID.

        Args:
            memory_id: UUID of the memory.

        Returns:
            ``True`` if a row was actually deleted.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM semantic_memories WHERE id = $1::uuid",
                memory_id,
            )

        deleted = result == "DELETE 1"
        if deleted:
            logger.info("Memory deleted  id=%s", memory_id)
        return deleted

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    async def consolidate_memories(self, user_id: str) -> int:
        """Merge semantically-duplicate memories for a user.

        For each pair of memories whose cosine similarity exceeds
        ``SIMILARITY_THRESHOLD``, the less-important one is merged into the
        more-important one and then deleted.

        Returns:
            The number of memories that were removed during consolidation.
        """
        pool = await self._get_pool()
        removed = 0

        async with pool.acquire() as conn:
            # Fetch all memories for the user
            rows = await conn.fetch(
                """
                SELECT id, content, embedding, memory_type, importance_score,
                       access_count, tags, metadata
                FROM semantic_memories
                WHERE user_id = $1 AND embedding IS NOT NULL
                ORDER BY importance_score DESC
                """,
                user_id,
            )

            to_delete: list[str] = []

            for i in range(len(rows)):
                if rows[i]["id"] in to_delete:
                    continue

                emb_i = rows[i]["embedding"]
                # asyncpg returns vector as a list
                if isinstance(emb_i, str):
                    emb_i = json.loads(emb_i)

                for j in range(i + 1, len(rows)):
                    if rows[j]["id"] in to_delete:
                        continue

                    emb_j = rows[j]["embedding"]
                    if isinstance(emb_j, str):
                        emb_j = json.loads(emb_j)

                    # Compute cosine similarity
                    sim = self._cosine_similarity(emb_i, emb_j)

                    if sim >= SIMILARITY_THRESHOLD:
                        # Merge j into i (the higher-importance one)
                        merged_tags = list(set(rows[i]["tags"] or []) | set(rows[j]["tags"] or []))
                        merged_meta = {**(rows[i]["metadata"] or {}), **(rows[j]["metadata"] or {})}

                        await conn.execute(
                            """
                            UPDATE semantic_memories
                            SET tags     = $2,
                                metadata = $3::jsonb,
                                access_count = access_count + $4,
                                updated_at   = NOW()
                            WHERE id = $1::uuid
                            """,
                            rows[i]["id"],
                            merged_tags,
                            json.dumps(merged_meta),
                            rows[j]["access_count"],
                        )

                        to_delete.append(rows[j]["id"])
                        removed += 1
                        logger.info(
                            "Consolidated  kept=%s  removed=%s  sim=%.3f",
                            rows[i]["id"],
                            rows[j]["id"],
                            sim,
                        )

            # Delete merged memories
            if to_delete:
                await conn.execute(
                    "DELETE FROM semantic_memories WHERE id = ANY($1::uuid[])",
                    to_delete,
                )

        logger.info(
            "Consolidation complete  user=%s  removed=%d",
            user_id,
            removed,
        )
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute the cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
