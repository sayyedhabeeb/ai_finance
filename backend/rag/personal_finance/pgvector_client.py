"""
pgvector-based vector store for personal finance documents.

Stores personal finance, tax, insurance, and investment rule documents
with their embeddings in PostgreSQL + pgvector.

Tables:
  - ``personal_finance_documents``: document metadata and full text
  - ``document_chunks``: chunked text with embedding vectors
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

# Embedding dimension for local HF model vectors padded/truncated for storage
_DEFAULT_VECTOR_DIM = 1536


# ──────────────────────────────────────────────────────────────
# SQL DDL
# ──────────────────────────────────────────────────────────────

_CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS personal_finance_documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT NOT NULL,
    doc_type        VARCHAR(100) NOT NULL DEFAULT 'text_document',
    category        VARCHAR(100) NOT NULL DEFAULT '',
    jurisdiction    VARCHAR(10)  NOT NULL DEFAULT 'IN',
    source          VARCHAR(500) NOT NULL DEFAULT '',
    title           VARCHAR(500) NOT NULL DEFAULT '',
    checksum        VARCHAR(64)  NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Index for common filter queries
CREATE INDEX IF NOT EXISTS idx_pfd_doc_type    ON personal_finance_documents (doc_type);
CREATE INDEX IF NOT EXISTS idx_pfd_category    ON personal_finance_documents (category);
CREATE INDEX IF NOT EXISTS idx_pfd_jurisdiction ON personal_finance_documents (jurisdiction);
CREATE INDEX IF NOT EXISTS idx_pfd_checksum    ON personal_finance_documents (checksum);
"""

_CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES personal_finance_documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    content         TEXT NOT NULL,
    embedding       vector({dimensions}) NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index for approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_dc_embedding
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_dc_document_id ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_dc_chunk_index ON document_chunks (document_id, chunk_index);
"""


class PersonalFinanceVectorStore:
    """pgvector-based vector store for personal finance documents.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string (asyncpg format), e.g.
        ``postgresql://user:pass@host:5432/db``.
    pool_min_size:
        Minimum number of connections in the pool.
    pool_max_size:
        Maximum number of connections in the pool.
    vector_dimensions:
        Embedding vector dimensionality.
    """

    def __init__(
        self,
        dsn: str | None = None,
        pool_min_size: int = 5,
        pool_max_size: int = 20,
        vector_dimensions: int = _DEFAULT_VECTOR_DIM,
    ) -> None:
        self._dsn = dsn
        self._pool_min = pool_min_size
        self._pool_max = pool_max_size
        self._dimensions = vector_dimensions
        self._pool: asyncpg.Pool | None = None

    # ── Connection lifecycle ────────────────────────────────

    async def connect(self) -> None:
        """Create the asyncpg connection pool and register pgvector."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        # Register the pgvector extension with asyncpg
        async with self._pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        logger.info("Connected to pgvector at %s (pool: %d–%d).", self._dsn or "default DSN", self._pool_min, self._pool_max)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("pgvector connection pool closed.")

    @property
    def pool(self) -> asyncpg.Pool:
        """Return the active pool, raising if not connected."""
        if self._pool is None:
            raise RuntimeError("Vector store is not connected. Call await connect() first.")
        return self._pool

    async def __aenter__(self) -> "PersonalFinanceVectorStore":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ── Table management ────────────────────────────────────

    async def ensure_tables_exist(self) -> None:
        """Create the documents and chunks tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(
                _CREATE_DOCUMENTS_TABLE.format(dimensions=self._dimensions)
            )
            await conn.execute(
                _CREATE_CHUNKS_TABLE.format(dimensions=self._dimensions)
            )
            logger.info("Ensured personal_finance_documents and document_chunks tables exist.")

    async def drop_tables(self) -> None:
        """Drop both tables (for testing / reset)."""
        async with self.pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
            await conn.execute("DROP TABLE IF EXISTS personal_finance_documents CASCADE;")
            logger.info("Dropped personal_finance_documents and document_chunks tables.")

    # ── Ingestion ───────────────────────────────────────────

    async def batch_ingest(
        self,
        documents: list[dict[str, Any]],
    ) -> int:
        """Ingest a batch of documents with their chunk embeddings.

        Each document dict must have:
          - ``content`` (str): full text
          - ``chunks`` (list[dict]): each chunk dict must have
            ``content`` (str) and ``embedding`` (list[float])
          - Optionally: ``doc_type``, ``category``, ``jurisdiction``,
            ``source``, ``title``, ``metadata``

        Returns
        -------
        int
            Total number of chunks stored.
        """
        if not documents:
            return 0

        total_chunks = 0

        async with self.pool.acquire() as conn:
            await register_vector(conn)
            async with conn.transaction():
                for doc in documents:
                    try:
                        n_chunks = await self._ingest_single_document(conn, doc)
                        total_chunks += n_chunks
                    except Exception as exc:
                        logger.error("Failed to ingest document: %s", exc)

        logger.info("Batch ingest complete: %d chunks from %d documents.", total_chunks, len(documents))
        return total_chunks

    async def _ingest_single_document(self, conn: asyncpg.Connection, doc: dict[str, Any]) -> int:
        """Insert one document and its chunks within a transaction."""
        content = doc.get("content", "")
        if not content.strip():
            return 0

        doc_type = doc.get("doc_type", "text_document")
        category = doc.get("category", "")
        jurisdiction = doc.get("jurisdiction", "IN")
        source = doc.get("source", "")
        title = doc.get("title", "")
        checksum = doc.get("checksum", self._compute_checksum(content, source))

        # Check for duplicate
        existing = await conn.fetchval(
            "SELECT id FROM personal_finance_documents WHERE checksum = $1 LIMIT 1;",
            checksum,
        )
        if existing:
            logger.debug("Skipping duplicate document (checksum: %s).", checksum[:12])
            return 0

        # Insert document
        doc_id = await conn.fetchval(
            """
            INSERT INTO personal_finance_documents (content, doc_type, category, jurisdiction, source, title, checksum)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id;
            """,
            content,
            doc_type,
            category,
            jurisdiction,
            source,
            title,
            checksum,
        )

        # Insert chunks
        chunks = doc.get("chunks", [])
        if not chunks:
            # Auto-chunk: treat entire document as one chunk
            chunks = [{"content": content, "embedding": doc.get("embedding", [])}]

        stored = 0
        for idx, chunk in enumerate(chunks):
            chunk_content = chunk.get("content", "")
            embedding = chunk.get("embedding", [])
            if not chunk_content or not embedding:
                continue

            metadata = chunk.get("metadata", {})
            if isinstance(metadata, dict):
                meta_json = json.dumps(metadata)
            else:
                meta_json = json.dumps({})

            await conn.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, content, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb);
                """,
                doc_id,
                idx,
                chunk_content,
                json.dumps(embedding),
                meta_json,
            )
            stored += 1

        return stored

    @staticmethod
    def _compute_checksum(content: str, source: str) -> str:
        """Compute SHA-256 checksum for deduplication."""
        import hashlib
        return hashlib.sha256(f"{content}:{source}".encode("utf-8")).hexdigest()

    # ── Vector search ───────────────────────────────────────

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Pure vector similarity search.

        Parameters
        ----------
        query_embedding:
            Query embedding vector.
        top_k:
            Number of results to return.
        filters:
            Optional filters: ``{"doc_type": "tax_rule", "category": "deductions"}``.

        Returns
        -------
        list[dict]
            Each dict has ``content``, ``doc_id``, ``chunk_id``, ``score``,
            ``doc_type``, ``category``, ``title``, ``source``, ``metadata``.
        """
        where_clauses, params = self._build_where_clause(filters, param_offset=1)
        query_embedding_json = json.dumps(query_embedding)

        sql = f"""
            SELECT
                dc.content,
                dc.id AS chunk_id,
                d.id AS doc_id,
                d.title,
                d.source,
                d.doc_type,
                d.category,
                d.jurisdiction,
                dc.metadata,
                1 - (dc.embedding <=> $1::vector) AS score
            FROM document_chunks dc
            JOIN personal_finance_documents d ON d.id = dc.document_id
            {f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""}
            ORDER BY dc.embedding <=> $1::vector
            LIMIT {int(top_k)};
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, query_embedding_json, *params)

        return [
            {
                "content": row["content"],
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "title": row["title"],
                "source": row["source"],
                "doc_type": row["doc_type"],
                "category": row["category"],
                "jurisdiction": row["jurisdiction"],
                "score": float(row["score"]),
                "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
            }
            for row in rows
        ]

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        alpha: float = 0.7,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining full-text (tsvector) and vector search.

        Parameters
        ----------
        query:
            Text query for full-text search.
        query_embedding:
            Query embedding for vector search.
        top_k:
            Number of results.
        alpha:
            Weight of vector component (0 = full-text only, 1 = vector only).
        filters:
            Optional metadata filters.

        Returns
        -------
        list[dict]
            Ranked results with blended scores.
        """
        where_clauses, params = self._build_where_clause(filters, param_offset=1)

        # Normalize alpha
        alpha = max(0.0, min(1.0, alpha))

        # Build the hybrid query using RRF (Reciprocal Rank Fusion)
        # Vector search rank
        query_embedding_json = json.dumps(query_embedding)

        sql = f"""
            WITH vector_results AS (
                SELECT
                    dc.content,
                    dc.id AS chunk_id,
                    d.id AS doc_id,
                    d.title,
                    d.source,
                    d.doc_type,
                    d.category,
                    d.jurisdiction,
                    dc.metadata,
                    1 - (dc.embedding <=> $1::vector) AS vector_score,
                    ROW_NUMBER() OVER (ORDER BY dc.embedding <=> $1::vector) AS vector_rank
                FROM document_chunks dc
                JOIN personal_finance_documents d ON d.id = dc.document_id
                {f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""}
            ),
            fts_results AS (
                SELECT
                    dc.content,
                    dc.id AS chunk_id,
                    d.id AS doc_id,
                    d.title,
                    d.source,
                    d.doc_type,
                    d.category,
                    d.jurisdiction,
                    dc.metadata,
                    ts_rank_cd(d.fts_vector, plainto_tsquery('english', $2)) AS fts_score,
                    ROW_NUMBER() OVER (ORDER BY ts_rank_cd(d.fts_vector, plainto_tsquery('english', $2)) DESC) AS fts_rank
                FROM document_chunks dc
                JOIN personal_finance_documents d ON d.id = dc.document_id
                WHERE d.fts_vector @@ plainto_tsquery('english', $2)
                {(' AND ' + ' AND '.join(where_clauses)) if where_clauses else ""}
            )
            SELECT
                COALESCE(v.content, f.content) AS content,
                COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
                COALESCE(v.doc_id, f.doc_id) AS doc_id,
                COALESCE(v.title, f.title) AS title,
                COALESCE(v.source, f.source) AS source,
                COALESCE(v.doc_type, f.doc_type) AS doc_type,
                COALESCE(v.category, f.category) AS category,
                COALESCE(v.jurisdiction, f.jurisdiction) AS jurisdiction,
                COALESCE(v.metadata, f.metadata) AS metadata,
                (
                    {float(alpha)} * COALESCE(v.vector_score, 0) +
                    {float(1 - alpha)} * COALESCE(f.fts_score, 0)
                ) AS score
            FROM vector_results v
            FULL OUTER JOIN fts_results f ON v.chunk_id = f.chunk_id
            ORDER BY score DESC
            LIMIT {int(top_k)};
        """

        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(sql, query_embedding_json, query, *params)
            except Exception as exc:
                logger.warning("Hybrid search failed (%s). Falling back to vector search.", exc)
                return await self.search(query_embedding, top_k, filters)

        return [
            {
                "content": row["content"],
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "title": row["title"],
                "source": row["source"],
                "doc_type": row["doc_type"],
                "category": row["category"],
                "jurisdiction": row["jurisdiction"],
                "score": float(row["score"]),
                "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
            }
            for row in rows
        ]

    async def fulltext_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search using PostgreSQL tsvector.

        This method ensures the ``fts_vector`` column exists by adding it
        and a GIN index if they don't exist yet.
        """
        # Ensure FTS column exists
        async with self.pool.acquire() as conn:
            # Add tsvector column if missing
            col_exists = await conn.fetchval(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'personal_finance_documents' AND column_name = 'fts_vector';
                """
            )
            if not col_exists:
                await conn.execute(
                    "ALTER TABLE personal_finance_documents ADD COLUMN fts_vector tsvector;"
                )
                await conn.execute(
                    "UPDATE personal_finance_documents SET fts_vector = to_tsvector('english', content);"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_pfd_fts ON personal_finance_documents USING GIN (fts_vector);"
                )
                logger.info("Created fts_vector column and GIN index.")

        # Now search
        where_clauses, params = self._build_where_clause(filters, param_offset=1)
        fts_where = "fts_vector @@ plainto_tsquery('english', $1)"
        if where_clauses:
            fts_where += " AND " + " AND ".join(where_clauses)

        sql = f"""
            SELECT
                dc.content,
                dc.id AS chunk_id,
                d.id AS doc_id,
                d.title,
                d.source,
                d.doc_type,
                d.category,
                d.jurisdiction,
                dc.metadata,
                ts_rank_cd(d.fts_vector, plainto_tsquery('english', $1)) AS score
            FROM document_chunks dc
            JOIN personal_finance_documents d ON d.id = dc.document_id
            WHERE {fts_where}
            ORDER BY score DESC
            LIMIT {int(top_k)};
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, query, *params)

        return [
            {
                "content": row["content"],
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "title": row["title"],
                "source": row["source"],
                "doc_type": row["doc_type"],
                "category": row["category"],
                "jurisdiction": row["jurisdiction"],
                "score": float(row["score"]),
                "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
            }
            for row in rows
        ]

    # ── Document management ─────────────────────────────────

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks by document UUID.

        Returns ``True`` if the document was found and deleted.
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM personal_finance_documents WHERE id = $1::uuid;",
                    doc_id,
                )
                deleted = result.split(" ")[1] if result else "0"
                return int(deleted) > 0
        except Exception as exc:
            logger.error("Failed to delete document %s: %s", doc_id, exc)
            return False

    async def get_document_count(self) -> int:
        """Return the total number of documents."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM personal_finance_documents;") or 0

    async def get_chunk_count(self) -> int:
        """Return the total number of chunks."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM document_chunks;") or 0

    # ── Filter builder ──────────────────────────────────────

    @staticmethod
    def _build_where_clause(
        filters: dict[str, Any] | None,
        param_offset: int = 1,
    ) -> tuple[list[str], list[Any]]:
        """Build SQL WHERE clauses and params from a filter dict.

        Returns (clause_strings, param_values).
        """
        if not filters:
            return [], []

        clauses: list[str] = []
        params: list[Any] = []
        idx = param_offset

        for key, value in filters.items():
            if isinstance(value, dict):
                # Operator dict: {"$gte": X, "$lt": Y}
                for op, operand in value.items():
                    sql_op = {
                        "$eq": "=", "$ne": "!=",
                        "$gt": ">", "$gte": ">=",
                        "$lt": "<", "$lte": "<=",
                    }.get(op)
                    if sql_op:
                        clauses.append(f"d.{key} ${idx} {sql_op}")
                        params.append(operand)
                        idx += 1
            elif isinstance(value, list):
                placeholders = ", ".join([f"${idx + j}" for j in range(len(value))])
                clauses.append(f"d.{key} IN ({placeholders})")
                params.extend(value)
                idx += len(value)
            else:
                clauses.append(f"d.{key} = ${idx}")
                params.append(value)
                idx += 1

        return clauses, params
