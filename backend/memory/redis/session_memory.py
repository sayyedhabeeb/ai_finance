"""
Redis-backed short-term session memory.

Stores conversation history, context windows, and ephemeral user data
with automatic TTL expiration and message windowing.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_SESSION_TTL_SECONDS = 24 * 60 * 60  # 24 hours
DEFAULT_MAX_MESSAGES = 50  # messages kept per session for the context window
KEY_SEPARATOR = ":"
PREFIX = "afb"  # ai-financial-brain


def _session_key(user_id: str, session_id: str) -> str:
    """Redis key for a session's message list."""
    return f"{PREFIX}{KEY_SEPARATOR}session{KEY_SEPARATOR}{user_id}{KEY_SEPARATOR}{session_id}"


def _context_key(user_id: str, session_id: str) -> str:
    """Redis key for a session's conversation-context hash."""
    return f"{PREFIX}{KEY_SEPARATOR}context{KEY_SEPARATOR}{user_id}{KEY_SEPARATOR}{session_id}"


def _user_key(user_id: str, key: str) -> str:
    """Redis key for arbitrary user-scoped key-value storage."""
    return f"{PREFIX}{KEY_SEPARATOR}kv{KEY_SEPARATOR}{user_id}{KEY_SEPARATOR}{key}"


def _session_index_key(user_id: str) -> str:
    """Redis sorted-set key listing all sessions for a user (score = last-active timestamp)."""
    return f"{PREFIX}{KEY_SEPARATOR}sessions{KEY_SEPARATOR}{user_id}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single conversation message."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ConversationContext:
    """Aggregated context for a conversation session."""

    user_id: str
    session_id: str
    summary: str = ""
    detected_intents: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    financial_context: dict[str, Any] = field(default_factory=dict)
    sentiment: str = "neutral"
    turn_count: int = 0
    last_active: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationContext":
        return cls(
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            summary=data.get("summary", ""),
            detected_intents=data.get("detected_intents", []),
            entities=data.get("entities", {}),
            financial_context=data.get("financial_context", {}),
            sentiment=data.get("sentiment", "neutral"),
            turn_count=data.get("turn_count", 0),
            last_active=data.get("last_active", time.time()),
        )


# ---------------------------------------------------------------------------
# RedisSessionMemory
# ---------------------------------------------------------------------------


class RedisSessionMemory:
    """Short-term session memory stored in Redis.

    Features:
    * Per-session message lists with automatic 24 h TTL (refreshed on activity).
    * Message windowing – only the last *max_messages* are kept so the list
      never grows unbounded.
    * Conversation-context hash with arbitrary key/value updates.
    * User-scoped generic key/value store with optional per-key TTL.
    * Session index sorted-set for listing a user's active sessions.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        session_ttl: int = DEFAULT_SESSION_TTL_SECONDS,
        max_messages: int = DEFAULT_MAX_MESSAGES,
    ) -> None:
        """Initialise the Redis connection pool.

        Args:
            redis_url: Redis connection URL.
            session_ttl: Default TTL applied to new sessions (seconds).
            max_messages: Maximum number of messages retained per session.
        """
        self._redis_url = redis_url
        self._session_ttl = session_ttl
        self._max_messages = max_messages

        self._pool = redis.ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=20,
            retry_on_timeout=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        self._client: redis.Redis = redis.Redis(connection_pool=self._pool)  # type: ignore[arg-type]

        logger.info(
            "RedisSessionMemory initialised  url=%s  ttl=%ds  max_msgs=%d",
            redis_url,
            session_ttl,
            max_messages,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        """Create (or reset) a new session.

        Sets up the message list, context hash, and registers the session in
        the user's session index.

        Args:
            user_id: Unique user identifier.
            session_id: Unique session identifier.

        Returns:
            Dictionary with session metadata.
        """
        now = time.time()
        s_key = _session_key(user_id, session_id)
        c_key = _context_key(user_id, session_id)
        idx_key = _session_index_key(user_id)

        pipe = self._client.pipeline(transaction=True)

        # Reset / create the message list
        pipe.delete(s_key)
        pipe.expire(s_key, self._session_ttl)

        # Initialise the context hash
        pipe.delete(c_key)
        ctx = ConversationContext(user_id=user_id, session_id=session_id, last_active=now)
        for field_name, value in ctx.to_dict().items():
            pipe.hset(c_key, field_name, json.dumps(value))
        pipe.expire(c_key, self._session_ttl)

        # Register in the user's session index (score = last-active timestamp)
        pipe.zadd(idx_key, {session_id: now})
        pipe.expire(idx_key, self._session_ttl + 3600)  # keep index a bit longer

        pipe.execute()

        logger.info("Session created  user=%s  session=%s", user_id, session_id)
        return {
            "user_id": user_id,
            "session_id": session_id,
            "created_at": now,
            "ttl": self._session_ttl,
        }

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append a message to a session and trim the window.

        Args:
            user_id: Unique user identifier.
            session_id: Unique session identifier.
            role: Message role (``user``, ``assistant``, ``system``, ``tool``).
            content: Text content of the message.
            metadata: Optional arbitrary metadata attached to the message.
        """
        if metadata is None:
            metadata = {}

        msg = Message(role=role, content=content, metadata=metadata)
        s_key = _session_key(user_id, session_id)
        c_key = _context_key(user_id, session_id)
        idx_key = _session_index_key(user_id)
        now = time.time()

        pipe = self._client.pipeline(transaction=True)

        # Store message as a JSON string on the right side of the list
        pipe.rpush(s_key, json.dumps(msg.to_dict()))

        # Trim to keep only the last N messages
        pipe.ltrim(s_key, -self._max_messages, -1)

        # Refresh TTL
        pipe.expire(s_key, self._session_ttl)
        pipe.expire(c_key, self._session_ttl)

        # Update session index (last-active score)
        pipe.zadd(idx_key, {session_id: now})

        # Increment turn counter in context
        pipe.hincrby(c_key, "turn_count", 1)
        pipe.hset(c_key, "last_active", json.dumps(now))

        pipe.execute()

        logger.debug(
            "Message added  user=%s  session=%s  role=%s  len=%d",
            user_id,
            session_id,
            role,
            len(content),
        )

    def get_session_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recent messages from a session.

        Args:
            user_id: Unique user identifier.
            session_id: Unique session identifier.
            limit: Maximum number of messages to return (most recent first).

        Returns:
            List of message dictionaries ordered oldest → newest.
        """
        s_key = _session_key(user_id, session_id)

        # Get the last *limit* messages from the tail
        raw: list[str] = self._client.lrange(s_key, -limit, -1)

        messages: list[dict[str, Any]] = []
        for item in raw:
            try:
                messages.append(json.loads(item))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to deserialize message: %s", exc)
                continue

        # lrange with negative indices returns in list-order (oldest→newest)
        return messages

    # ------------------------------------------------------------------
    # Conversation context
    # ------------------------------------------------------------------

    def get_context(self, user_id: str, session_id: str) -> ConversationContext:
        """Read the full conversation context for a session.

        Args:
            user_id: Unique user identifier.
            session_id: Unique session identifier.

        Returns:
            A :class:`ConversationContext` instance.
        """
        c_key = _context_key(user_id, session_id)
        raw: dict[str, str] = self._client.hgetall(c_key)

        if not raw:
            return ConversationContext(user_id=user_id, session_id=session_id)

        parsed: dict[str, Any] = {}
        for field_name, value in raw.items():
            try:
                parsed[field_name] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed[field_name] = value

        return ConversationContext.from_dict(parsed)

    def update_context(
        self,
        user_id: str,
        session_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Merge updates into the conversation context.

        Only the keys present in *updates* are modified; all other keys remain
        unchanged.

        Args:
            user_id: Unique user identifier.
            session_id: Unique session identifier.
            updates: Mapping of field names to new values.
        """
        c_key = _context_key(user_id, session_id)
        now = time.time()

        pipe = self._client.pipeline(transaction=False)

        for field_name, value in updates.items():
            if field_name in ("user_id", "session_id"):
                continue  # immutable fields
            pipe.hset(c_key, field_name, json.dumps(value))

        pipe.hset(c_key, "last_active", json.dumps(now))
        pipe.expire(c_key, self._session_ttl)
        pipe.execute()

        logger.debug("Context updated  user=%s  session=%s  keys=%s", user_id, session_id, list(updates.keys()))

    # ------------------------------------------------------------------
    # Generic key / value store (user-scoped)
    # ------------------------------------------------------------------

    def set_key(
        self,
        user_id: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Store an arbitrary value under a user-scoped key.

        Args:
            user_id: Unique user identifier.
            key: Key name.
            value: Any JSON-serialisable value.
            ttl: Optional TTL in seconds.  If ``None`` the key persists until
                 the session expires.
        """
        k = _user_key(user_id, key)
        serialised = json.dumps(value)
        if ttl is not None:
            self._client.setex(k, ttl, serialised)
        else:
            self._client.set(k, serialised)

    def get_key(self, user_id: str, key: str) -> Any | None:
        """Retrieve a value from the user-scoped key / value store.

        Args:
            user_id: Unique user identifier.
            key: Key name.

        Returns:
            The stored value, or ``None`` if the key does not exist.
        """
        k = _user_key(user_id, key)
        raw: str | None = self._client.get(k)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_session(self, user_id: str, session_id: str) -> None:
        """Delete all data associated with a session.

        Args:
            user_id: Unique user identifier.
            session_id: Unique session identifier.
        """
        s_key = _session_key(user_id, session_id)
        c_key = _context_key(user_id, session_id)
        idx_key = _session_index_key(user_id)

        pipe = self._client.pipeline(transaction=True)
        pipe.delete(s_key)
        pipe.delete(c_key)
        pipe.zrem(idx_key, session_id)
        pipe.execute()

        logger.info("Session cleared  user=%s  session=%s", user_id, session_id)

    def cleanup_expired_sessions(self, user_id: str) -> list[str]:
        """Remove sessions whose TTL has lapsed.

        The session index is a sorted-set keyed by last-active timestamp.
        We remove entries older than the configured TTL.

        Args:
            user_id: Unique user identifier.

        Returns:
            List of session IDs that were removed.
        """
        idx_key = _session_index_key(user_id)
        cutoff = time.time() - self._session_ttl

        # Find expired session IDs (score < cutoff)
        expired: list[str] = self._client.zrangebyscore(idx_key, "-inf", cutoff)

        if not expired:
            return []

        pipe = self._client.pipeline(transaction=True)
        for sid in expired:
            pipe.delete(_session_key(user_id, sid))
            pipe.delete(_context_key(user_id, sid))
            pipe.zrem(idx_key, sid)
        pipe.execute()

        logger.info(
            "Expired sessions removed  user=%s  count=%d",
            user_id,
            len(expired),
        )
        return expired

    def list_sessions(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """List the user's most-recently-active sessions.

        Args:
            user_id: Unique user identifier.
            limit: Maximum number of sessions to return.

        Returns:
            List of dicts with ``session_id`` and ``last_active``.
        """
        idx_key = _session_index_key(user_id)
        results: list[tuple[str, float]] = self._client.zrevrange(
            idx_key, 0, limit - 1, withscores=True
        )

        return [
            {"session_id": sid, "last_active": score}
            for sid, score in results
        ]

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Redis connection pool."""
        self._client.close()
        logger.info("RedisSessionMemory connection closed")

    def ping(self) -> bool:
        """Return ``True`` if the Redis server is reachable."""
        try:
            return self._client.ping()
        except redis.ConnectionError as exc:
            logger.error("Redis ping failed: %s", exc)
            return False
