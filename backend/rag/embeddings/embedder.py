"""
Unified embedding service using local HuggingFace sentence-transformers.

Groq currently does not provide embedding endpoints in this codebase, so
embeddings are generated locally with sentence-transformers by default.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_MODELS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIMENSIONS = 384


class EmbeddingService:
    """Unified embedding service supporting local HuggingFace models.

    Parameters
    ----------
    model_name:
        HuggingFace sentence-transformers model name.
    api_key:
        Unused for local embeddings. Kept for backward compatibility.
    redis_client:
        Optional ``redis.Redis`` instance for caching.
    cache_ttl:
        Time-to-live for cache entries in seconds (default 7 days).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        redis_client: Any | None = None,
        cache_ttl: int = 604800,
    ) -> None:
        self.model_name = model_name
        self.dimensions = _EMBEDDING_MODELS.get(model_name, _DEFAULT_DIMENSIONS)
        self._api_key = api_key
        self._redis = redis_client
        self._cache_ttl = cache_ttl
        self._local_model: Any | None = None

    def _get_local_model(self) -> Any:
        """Lazily load a local sentence-transformers model."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer(self.model_name)
                self.dimensions = self._local_model.get_sentence_embedding_dimension()
                logger.info("Loaded local sentence-transformers model: %s", self.model_name)
            except Exception as exc:
                logger.error("Failed to load local model: %s", exc)
                raise
        return self._local_model

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"emb:{self.model_name}:{digest}"

    def _cache_get(self, text: str) -> list[float] | None:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._cache_key(text))
            if raw is not None:
                return json.loads(raw)
        except Exception:
            logger.debug("Cache miss/error for text (first 50 chars): %s", text[:50])
        return None

    def _cache_set(self, text: str, embedding: list[float]) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(self._cache_key(text), self._cache_ttl, json.dumps(embedding))
        except Exception:
            logger.debug("Failed to cache embedding for text (first 50 chars): %s", text[:50])

    def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        if use_cache:
            cached = self._cache_get(text)
            if cached is not None:
                return cached

        embedding = self._embed_local(text)
        if use_cache:
            self._cache_set(text, embedding)
        return embedding

    async def aembed_text(self, text: str, use_cache: bool = True) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        if use_cache:
            cached = self._cache_get(text)
            if cached is not None:
                return cached

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed_local, text)
        if use_cache:
            self._cache_set(text, embedding)
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
        use_cache: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        _ = batch_size  # kept for API compatibility
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if use_cache:
                cached = self._cache_get(text)
                if cached is not None:
                    results[i] = cached
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        for idx, text in zip(uncached_indices, uncached_texts):
            emb = self._embed_local(text)
            results[idx] = emb
            if use_cache:
                self._cache_set(text, emb)

        return [r for r in results if r is not None]

    async def aembed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
        use_cache: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        _ = batch_size  # kept for API compatibility
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if use_cache:
                cached = self._cache_get(text)
                if cached is not None:
                    results[i] = cached
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        loop = asyncio.get_event_loop()
        generated = await asyncio.gather(
            *[loop.run_in_executor(None, self._embed_local, t) for t in uncached_texts]
        )

        for idx, text, emb in zip(uncached_indices, uncached_texts, generated):
            results[idx] = emb
            if use_cache:
                self._cache_set(text, emb)

        return [r for r in results if r is not None]

    def _embed_local(self, text: str) -> list[float]:
        model = self._get_local_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        vec_a = np.array(a, dtype=np.float32)
        vec_b = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def normalize(embedding: list[float]) -> list[float]:
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return embedding
        return (vec / norm).tolist()
