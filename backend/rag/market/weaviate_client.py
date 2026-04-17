"""
Weaviate vector store client for market-related documents.

Manages two Weaviate collections:
  - **MarketDocuments**: research reports, filings, earnings transcripts, etc.
  - **NewsDocuments**: news articles and market commentary.

Uses the Weaviate v4 Python client (``weaviate-client>=4.9``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, HybridFusion

from backend.rag.models import RAGDocument, RetrievalResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Collection schemas
# ──────────────────────────────────────────────────────────────

MARKET_DOCUMENTS_PROPERTIES = [
    Property(name="content", data_type=DataType.TEXT, description="Full text content for vector search"),
    Property(name="title", data_type=DataType.TEXT, description="Document title"),
    Property(name="source", data_type=DataType.TEXT, description="Source URL or file path"),
    Property(name="docType", data_type=DataType.TEXT, description="Type of document"),
    Property(name="date", data_type=DataType.TEXT, description="Document date YYYY-MM-DD"),
    Property(name="tickerSymbols", data_type=DataType.TEXT_ARRAY, description="Associated ticker symbols"),
    Property(name="sector", data_type=DataType.TEXT, description="Industry sector"),
    Property(name="sentimentScore", data_type=DataType.NUMBER, description="Sentiment score [-1, 1]"),
    Property(name="text", data_type=DataType.TEXT, description="Alias for content (searchable)"),
    Property(name="checksum", data_type=DataType.TEXT, description="SHA-256 for deduplication"),
    Property(name="docId", data_type=DataType.TEXT, description="Unique document ID"),
    Property(name="chunkIndex", data_type=DataType.NUMBER, description="Chunk index within document"),
]

NEWS_DOCUMENTS_PROPERTIES = [
    Property(name="content", data_type=DataType.TEXT, description="Article content for vector search"),
    Property(name="title", data_type=DataType.TEXT, description="Article title"),
    Property(name="source", data_type=DataType.TEXT, description="Source URL"),
    Property(name="publishDate", data_type=DataType.TEXT, description="Publication date YYYY-MM-DD"),
    Property(name="tickers", data_type=DataType.TEXT_ARRAY, description="Associated ticker symbols"),
    Property(name="sentiment", data_type=DataType.NUMBER, description="Sentiment score [-1, 1]"),
    Property(name="summary", data_type=DataType.TEXT, description="Article summary"),
    Property(name="text", data_type=DataType.TEXT, description="Alias for content (searchable)"),
    Property(name="checksum", data_type=DataType.TEXT, description="SHA-256 for deduplication"),
    Property(name="docId", data_type=DataType.TEXT, description="Unique document ID"),
    Property(name="chunkIndex", data_type=DataType.NUMBER, description="Chunk index within document"),
]


class MarketWeaviateClient:
    """Manages Weaviate vector store for market-related documents.

    Parameters
    ----------
    weaviate_url:
        Weaviate REST/gRPC URL.
    api_key:
        Weaviate API key (``None`` for local unauthenticated).
    groq_api_key:
        Unused legacy placeholder for backward compatibility.
        Groq is used for LLM calls, while embeddings should be pre-computed.
    """

    def __init__(
        self,
        weaviate_url: str | None = None,
        api_key: str | None = None,
        groq_api_key: str | None = None,
    ) -> None:
        self._url = weaviate_url
        self._api_key = api_key
        self._groq_key = groq_api_key
        self._client: weaviate.WeaviateClient | None = None

    # ── Connection lifecycle ────────────────────────────────

    def connect(self) -> None:
        """Establish a connection to Weaviate."""
        try:
            headers: dict[str, str] = {}

            if self._api_key:
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self._url,
                    auth_credentials=weaviate.auth.AuthApiKey(self._api_key),
                    headers=headers,
                )
            else:
                self._client = weaviate.connect_to_local(
                    host="localhost",
                    port=8080,
                    grpc_port=50051,
                    headers=headers,
                )

            self._client.connect()
            logger.info("Connected to Weaviate at %s", self._url or "localhost:8080")
        except Exception as exc:
            logger.error("Failed to connect to Weaviate: %s", exc)
            raise

    def close(self) -> None:
        """Close the Weaviate connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Weaviate connection closed.")

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Return the active Weaviate client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Weaviate client is not connected. Call connect() first.")
        return self._client

    def __enter__(self) -> "MarketWeaviateClient":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Collection management ───────────────────────────────

    def ensure_collections_exist(self) -> None:
        """Create MarketDocuments and NewsDocuments collections if they do not exist."""
        self.create_collection("MarketDocuments", MARKET_DOCUMENTS_PROPERTIES)
        self.create_collection("NewsDocuments", NEWS_DOCUMENTS_PROPERTIES)
        logger.info("Ensured MarketDocuments and NewsDocuments collections exist.")

    def create_collection(
        self,
        name: str,
        properties: list[Property],
        vectorizer: str = "none",
        distance_metric: VectorDistances = VectorDistances.COSINE,
    ) -> None:
        """Create a Weaviate collection (class) with the given schema.

        Parameters
        ----------
        name:
            Collection name.
        properties:
            List of ``Property`` objects defining the schema.
        vectorizer:
            Vectorizer module (``"none"`` for pre-computed embeddings).
        distance_metric:
            Distance metric for vector search.
        """
        try:
            if self.client.collections.exists(name):
                logger.debug("Collection %s already exists – skipping.", name)
                return

            self.client.collections.create(
                name=name,
                properties=properties,
                vectorizer_config=(
                    Configure.Vectorizer.none()
                    if vectorizer == "none"
                    # TODO: Weaviate has no native Groq text2vec module.
                    # Use pre-computed embeddings and vectorizer="none".
                    else Configure.Vectorizer.none()
                ),
                vector_index_config=Configure.VectorIndex.flat(
                    distance_metric=distance_metric,
                ),
            )
            logger.info("Created collection: %s", name)
        except Exception as exc:
            logger.error("Failed to create collection %s: %s", name, exc)
            raise

    def delete_collection(self, name: str) -> None:
        """Delete a Weaviate collection."""
        try:
            if self.client.collections.exists(name):
                self.client.collections.delete(name)
                logger.info("Deleted collection: %s", name)
        except Exception as exc:
            logger.error("Failed to delete collection %s: %s", name, exc)
            raise

    # ── Batch ingestion ─────────────────────────────────────

    def batch_ingest(
        self,
        documents: list[RAGDocument],
        collection_name: str = "MarketDocuments",
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """Ingest a batch of documents into a Weaviate collection.

        Parameters
        ----------
        documents:
            List of :class:`RAGDocument` to ingest.
        collection_name:
            Target Weaviate collection.
        embeddings:
            Pre-computed embedding vectors.  If provided, must have the same
            length as *documents*.  If ``None``, the Weaviate server-side
            vectorizer will be used.

        Returns
        -------
        int
            Number of documents successfully ingested.
        """
        if not documents:
            return 0

        collection = self.client.collections.get(collection_name)
        ingested = 0
        seen_checksums: set[str] = set()

        with collection.batch.dynamic() as batch:
            for i, doc in enumerate(documents):
                # Dedup: skip if we already ingested this checksum in the batch
                if doc.checksum in seen_checksums:
                    logger.debug("Skipping duplicate document (checksum): %s", doc.doc_id)
                    continue
                seen_checksums.add(doc.checksum)

                # Determine if this is a news doc or market doc
                is_news = collection_name == "NewsDocuments"
                properties = self._doc_to_properties(doc, is_news)

                vector = embeddings[i] if embeddings and i < len(embeddings) else None

                try:
                    batch.add_object(
                        properties=properties,
                        uuid=None,  # auto-generated
                        vector=vector,
                    )
                    ingested += 1
                except Exception as exc:
                    logger.error("Failed to ingest doc %s: %s", doc.doc_id, exc)

        logger.info("Ingested %d/%d documents into %s.", ingested, len(documents), collection_name)
        return ingested

    def batch_ingest_chunks(
        self,
        chunks: list[dict[str, Any]],
        collection_name: str = "MarketDocuments",
    ) -> int:
        """Ingest pre-chunked documents with their embeddings.

        Each item in *chunks* must contain at least ``content`` and
        optionally ``vector``.

        Returns
        -------
        int
            Number of chunks successfully ingested.
        """
        if not chunks:
            return 0

        collection = self.client.collections.get(collection_name)
        ingested = 0

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                vector = chunk.pop("vector", None)
                try:
                    batch.add_object(
                        properties=chunk,
                        vector=vector,
                    )
                    ingested += 1
                except Exception as exc:
                    logger.error("Failed to ingest chunk: %s", exc)

        logger.info("Ingested %d/%d chunks into %s.", ingested, len(chunks), collection_name)
        return ingested

    @staticmethod
    def _doc_to_properties(doc: RAGDocument, is_news: bool = False) -> dict[str, Any]:
        """Convert a :class:`RAGDocument` to a Weaviate property dict."""
        if is_news:
            return {
                "content": doc.content,
                "title": doc.title,
                "source": doc.source,
                "publishDate": doc.date,
                "tickers": doc.ticker_symbols,
                "sentiment": doc.sentiment_score,
                "summary": doc.metadata.get("summary", ""),
                "text": doc.content,
                "checksum": doc.checksum,
                "docId": doc.doc_id,
                "chunkIndex": doc.metadata.get("chunk_index", 0),
            }
        return {
            "content": doc.content,
            "title": doc.title,
            "source": doc.source,
            "docType": doc.doc_type.value,
            "date": doc.date,
            "tickerSymbols": doc.ticker_symbols,
            "sector": doc.sector,
            "sentimentScore": doc.sentiment_score,
            "text": doc.content,
            "checksum": doc.checksum,
            "docId": doc.doc_id,
            "chunkIndex": doc.metadata.get("chunk_index", 0),
        }

    # ── Search ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        collection_name: str = "MarketDocuments",
        query_vector: list[float] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """Pure vector (or BM25) search.

        Parameters
        ----------
        query:
            Query text (used for BM25 if no vector provided).
        query_vector:
            Pre-computed query embedding.  If ``None``, a vector-only
            search is not possible; BM25 text search will be used.
        filters:
            Optional metadata filters, e.g. ``{"sector": "IT", "date": {"$gte": "2024-01-01"}}``.
        limit:
            Maximum number of results.

        Returns
        -------
        list[RetrievalResult]
        """
        collection = self.client.collections.get(collection_name)

        # Build Weaviate filter from dict
        w_filter = self._build_filter(filters) if filters else None

        results: list[RetrievalResult] = []

        if query_vector is not None:
            # Vector search
            response = collection.query.near_vector(
                near_vector=query_vector,
                filters=w_filter,
                limit=limit,
                include_metadata=True,
            )
            for obj in response.objects:
                results.append(self._object_to_result(obj))
        else:
            # BM25 text search
            response = collection.query.bm25(
                query=query,
                filters=w_filter,
                limit=limit,
            )
            for obj in response.objects:
                results.append(self._object_to_result(obj, score_field="score"))

        return results

    def hybrid_search(
        self,
        query: str,
        collection_name: str = "MarketDocuments",
        query_vector: list[float] | None = None,
        alpha: float = 0.7,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        fusion_type: HybridFusion = HybridFusion.RELATIVE_SCORE,
    ) -> list[RetrievalResult]:
        """Hybrid search combining BM25 (keyword) and vector (semantic) search.

        Parameters
        ----------
        query:
            Query text for BM25 component.
        query_vector:
            Pre-computed query embedding.  If ``None``, only BM25 is used.
        alpha:
            Weight of the vector component (0 = pure BM25, 1 = pure vector).
        filters:
            Optional metadata filters.
        limit:
            Maximum number of results.
        fusion_type:
            How to combine BM25 and vector scores.

        Returns
        -------
        list[RetrievalResult]
        """
        collection = self.client.collections.get(collection_name)
        w_filter = self._build_filter(filters) if filters else None

        kwargs: dict[str, Any] = {
            "query": query,
            "alpha": alpha,
            "filters": w_filter,
            "limit": limit,
            "fusion_type": fusion_type,
        }

        if query_vector is not None:
            kwargs["vector"] = query_vector

        response = collection.query.hybrid(**kwargs)

        results: list[RetrievalResult] = []
        for obj in response.objects:
            results.append(self._object_to_result(obj, score_field="score"))

        return results

    # ── Document management ─────────────────────────────────

    def delete_documents(
        self,
        doc_ids: list[str],
        collection_name: str = "MarketDocuments",
    ) -> int:
        """Delete documents by their ``docId`` property.

        Returns
        -------
        int
            Number of documents deleted.
        """
        if not doc_ids:
            return 0

        collection = self.client.collections.get(collection_name)
        deleted = 0

        for doc_id in doc_ids:
            try:
                # First find the UUID by docId
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("docId").equal(doc_id),
                    limit=1,
                )
                for obj in response.objects:
                    collection.data.delete_by_id(obj.uuid)
                    deleted += 1
                    logger.debug("Deleted document %s (Weaviate UUID: %s)", doc_id, obj.uuid)
            except Exception as exc:
                logger.error("Failed to delete document %s: %s", doc_id, exc)

        logger.info("Deleted %d/%d documents from %s.", deleted, len(doc_ids), collection_name)
        return deleted

    def get_document_count(self, collection_name: str = "MarketDocuments") -> int:
        """Return the total number of objects in a collection."""
        try:
            collection = self.client.collections.get(collection_name)
            return collection.aggregate.overall(total_count=True).total_count or 0
        except Exception as exc:
            logger.error("Failed to get document count for %s: %s", collection_name, exc)
            return 0

    # ── Internal helpers ────────────────────────────────────

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """Recursively build a Weaviate :class:`Filter` from a plain dict.

        Supports:
          - ``{"field": "value"}``  →  equality
          - ``{"field": {"$eq": "value"}}``  →  equality
          - ``{"field": {"$ne": "value"}}``  →  not-equal
          - ``{"field": {"$gt": N, "$lt": M}}``  →  numeric range
          - ``{"field": {"$gte": "2024-01-01"}}``  →  text prefix/gte
          - ``{"field": {"$in": ["a", "b"]}}``  →  contains-any
          - ``{"$and": [...]}``, ``{"$or": [...]}``  →  logical operators
        """
        conditions: list[Filter] = []

        for key, value in filters.items():
            if key == "$and":
                sub = [MarketWeaviateClient._build_filter(f) for f in value]
                conditions.append(Filter.all_of(sub))
            elif key == "$or":
                sub = [MarketWeaviateClient._build_filter(f) for f in value]
                conditions.append(Filter.any_of(sub))
            elif isinstance(value, dict):
                # Operator dict like {"$gt": 100, "$lt": 200}
                for op, operand in value.items():
                    conditions.append(MarketWeaviateClient._operator_filter(key, op, operand))
            else:
                # Direct equality
                conditions.append(Filter.by_property(key).equal(value))

        if len(conditions) == 1:
            return conditions[0]
        return Filter.all_of(conditions)

    @staticmethod
    def _operator_filter(prop: str, op: str, value: Any) -> Filter:
        """Translate a single operator to a Weaviate Filter."""
        f = Filter.by_property(prop)
        ops = {
            "$eq": lambda: f.equal(value),
            "$ne": lambda: f.not_equal(value),
            "$gt": lambda: f.greater_than(value),
            "$gte": lambda: f.greater_or_equal(value),
            "$lt": lambda: f.less_than(value),
            "$lte": lambda: f.less_or_equal(value),
            "$in": lambda: f.contains_any(value) if isinstance(value, list) else f.contains_any([value]),
            "$contains": lambda: f.contains_any([value]),
        }
        handler = ops.get(op)
        if handler is None:
            raise ValueError(f"Unknown filter operator: {op}")
        return handler()

    @staticmethod
    def _object_to_result(obj: Any, score_field: str = "distance") -> RetrievalResult:
        """Convert a Weaviate query object to a :class:`RetrievalResult`."""
        props = obj.properties if hasattr(obj, "properties") else {}

        # Normalise distance to a [0, 1] score (1 = most relevant)
        raw_score = 0.0
        if hasattr(obj, "metadata") and obj.metadata:
            score_dict = getattr(obj.metadata, "distance", None)
            if score_dict is not None:
                raw_score = float(score_dict)
            else:
                score_val = getattr(obj.metadata, "score", None)
                if score_val is not None:
                    raw_score = float(score_val)

        # If score_field is "score", it's already a relevance score
        if score_field == "score":
            relevance = max(0.0, min(1.0, raw_score))
        else:
            # distance → convert to similarity (cosine)
            relevance = max(0.0, 1.0 - raw_score) if raw_score <= 1.0 else 1.0 / (1.0 + raw_score)

        return RetrievalResult(
            content=props.get("content", props.get("text", "")),
            doc_id=props.get("docId", ""),
            chunk_id=str(obj.uuid) if hasattr(obj, "uuid") else "",
            title=props.get("title", ""),
            source=props.get("source", ""),
            score=relevance,
            doc_type=props.get("docType", props.get("docType", "")),
            metadata={
                "tickerSymbols": props.get("tickerSymbols", props.get("tickers", [])),
                "sector": props.get("sector", ""),
                "sentimentScore": props.get("sentimentScore", props.get("sentiment", 0.0)),
                "date": props.get("date", props.get("publishDate", "")),
                "chunkIndex": props.get("chunkIndex", 0),
                "summary": props.get("summary", ""),
            },
        )
