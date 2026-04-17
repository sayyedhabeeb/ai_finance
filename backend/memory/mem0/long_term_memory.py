"""
Long-term memory management using Mem0.

Mem0 provides an intelligent memory layer that automatically extracts,
deduplicates, and retrieves relevant memories from conversations.  This
module wraps the Mem0 client for use in the AI Financial Brain.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Mem0LongTermMemory:
    """Long-term memory management using Mem0.

    Mem0 automatically manages the extraction and deduplication of memories
    from conversational content.  This wrapper exposes a synchronous and
    asynchronous-friendly interface.

    Requires the ``mem0ai`` package::

        pip install mem0ai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialise the Mem0 client.

        Args:
            api_key: Mem0 API key.  If ``None`` the value is read from the
                     ``MEM0_API_KEY`` environment variable.
            config: Optional Mem0 configuration dict that overrides the
                    default.  If not provided a sensible default config is
                    built automatically.
        """
        import os

        api_key = api_key or os.environ.get("MEM0_API_KEY", "")

        if config is None:
            config = {
                "llm": {
                    "provider": "groq",
                    "config": {
                        "model": "llama3-70b-8192",
                        "api_key": os.environ.get("GROQ_API_KEY", ""),
                        "temperature": 0.0,
                    },
                },
                "embedder": {
                    # TODO: Validate provider compatibility with the installed mem0 version.
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "ai_financial_brain",
                        "host": os.environ.get("QDRANT_HOST", "localhost"),
                        "port": int(os.environ.get("QDRANT_PORT", "6333")),
                    },
                },
                "version": "v1.1",
            }

        # Import here so the module can still be loaded even if mem0ai is
        # not installed (e.g. during testing of other components).
        try:
            from mem0 import Memory

            self._client = Memory.from_config(config_dict=config)
        except ImportError:
            logger.warning(
                "mem0ai package not installed.  "
                "Long-term memory features will be unavailable.  "
                "Install via: pip install mem0ai"
            )
            self._client = None

        self._api_key = api_key
        self._config = config
        logger.info("Mem0LongTermMemory initialised  client=%s", "ok" if self._client else "unavailable")

    def _require_client(self) -> Any:
        """Raise if the Mem0 client is not available."""
        if self._client is None:
            raise RuntimeError(
                "Mem0 client is not initialised.  "
                "Ensure the 'mem0ai' package is installed and a valid "
                "configuration / API key is provided."
            )
        return self._client

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add a new memory for a user.

        Mem0 automatically deduplicates: if the content is semantically
        equivalent to an existing memory it will be merged rather than
        duplicated.

        Args:
            user_id: Unique user identifier.
            content: Text content to remember (e.g. a user preference,
                     a financial fact, or a conversational insight).
            metadata: Optional metadata attached to the memory.

        Returns:
            Result dict from the Mem0 API, typically containing a
            ``results`` key with the added/updated memory IDs.
        """
        client = self._require_client()
        metadata = metadata or {}
        metadata["user_id"] = user_id  # ensure user_id is in metadata for filtering

        result = client.add(
            content,
            user_id=user_id,
            metadata=metadata,
        )

        logger.info(
            "Memory added via Mem0  user=%s  result=%s",
            user_id,
            result,
        )
        return result

    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for relevant memories.

        Args:
            user_id: Scope the search to this user.
            query: Natural-language query.
            limit: Maximum number of results.

        Returns:
            List of memory dicts, each containing at least ``id``,
            ``memory``, ``score``, and ``metadata``.
        """
        client = self._require_client()

        results = client.search(
            query,
            user_id=user_id,
            limit=limit,
        )

        logger.debug(
            "Mem0 search  user=%s  query=%.50s  hits=%d",
            user_id,
            query,
            len(results.get("results", [])) if isinstance(results, dict) else len(results),
        )

        # Mem0 v1.1 returns {"results": [...]}
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        if isinstance(results, list):
            return results

        return []

    def get_all_memories(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve all stored memories for a user.

        Args:
            user_id: Unique user identifier.

        Returns:
            List of memory dicts.
        """
        client = self._require_client()

        all_memories = client.get_all(user_id=user_id)

        logger.debug(
            "Mem0 get_all  user=%s  count=%d",
            user_id,
            len(all_memories.get("results", [])) if isinstance(all_memories, dict) else len(all_memories),
        )

        if isinstance(all_memories, dict) and "results" in all_memories:
            return all_memories["results"]
        if isinstance(all_memories, list):
            return all_memories

        return []

    def update_memory(self, memory_id: str, content: str) -> dict[str, Any]:
        """Update the content of an existing memory.

        Args:
            memory_id: The Mem0 memory ID.
            content: New text content.

        Returns:
            Result dict from the Mem0 API.
        """
        client = self._require_client()

        result = client.update(memory_id, content)

        logger.info("Memory updated via Mem0  id=%s", memory_id)
        return result

    def delete_memory(self, memory_id: str) -> None:
        """Delete a single memory.

        Args:
            memory_id: The Mem0 memory ID.
        """
        client = self._require_client()
        client.delete(memory_id)

        logger.info("Memory deleted via Mem0  id=%s", memory_id)

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """Retrieve the aggregated user profile built by Mem0.

        Mem0 can maintain a running profile of a user that captures key
        facts and preferences inferred across conversations.

        Args:
            user_id: Unique user identifier.

        Returns:
            User-profile dict.
        """
        client = self._require_client()

        try:
            profile = client.get_user(user_id)
            logger.debug("Mem0 user profile  user=%s  keys=%s", user_id, list(profile.keys()) if isinstance(profile, dict) else "N/A")
            return profile
        except Exception as exc:
            logger.warning("Failed to retrieve user profile from Mem0: %s", exc)
            return {}
