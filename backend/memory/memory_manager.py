"""
Unified memory manager that coordinates all memory subsystems.

Provides a single entry-point for the rest of the AI Financial Brain to
interact with session memory (Redis), semantic memory (pgvector), long-term
memory (Mem0), and episodic memory (PostgreSQL).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .redis.session_memory import RedisSessionMemory, ConversationContext
from .pgvector.semantic_memory import SemanticMemoryStore, MemoryEntry as SemanticMemoryEntry
from .mem0.long_term_memory import Mem0LongTermMemory
from .episodic.episodic_memory import EpisodicMemoryStore, MemoryEntry as EpisodicMemoryEntry

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified memory manager that coordinates all memory subsystems.

    Usage::

        manager = MemoryManager()
        await manager.initialize()

        # Full context for an LLM call
        ctx = await manager.get_full_context(user_id, session_id, "Show my portfolio")

        # Store insights after an interaction
        await manager.create_memory_from_interaction(
            user_id, query, response, agent_results
        )

        await manager.close()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        db_dsn: str = "postgresql://postgres:postgres@localhost:5432/ai_financial_brain",
        mem0_api_key: Optional[str] = None,
        mem0_config: Optional[dict[str, Any]] = None,
        session_ttl: int = 86400,
        max_session_messages: int = 50,
    ) -> None:
        """Initialise all memory subsystems.

        Args:
            redis_url: Redis connection URL for session memory.
            db_dsn: PostgreSQL DSN for semantic & episodic stores.
            mem0_api_key: Optional Mem0 API key.
            mem0_config: Optional Mem0 configuration override.
            session_ttl: Default session TTL in seconds.
            max_session_messages: Max messages kept per session.
        """
        self._redis_url = redis_url
        self._db_dsn = db_dsn
        self._mem0_api_key = mem0_api_key
        self._mem0_config = mem0_config
        self._session_ttl = session_ttl
        self._max_session_messages = max_session_messages

        # Subsystems – instantiated in `initialize()`
        self.session: RedisSessionMemory = RedisSessionMemory(
            redis_url=redis_url,
            session_ttl=session_ttl,
            max_messages=max_session_messages,
        )
        self.semantic: SemanticMemoryStore = SemanticMemoryStore(db_dsn=db_dsn)
        self.long_term: Mem0LongTermMemory = Mem0LongTermMemory(
            api_key=mem0_api_key,
            config=mem0_config,
        )
        self.episodic: EpisodicMemoryStore = EpisodicMemoryStore(db_dsn=db_dsn)

        self._initialised = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Set up all connections and create required database tables.

        This must be called once before any other method.
        """
        if self._initialised:
            logger.debug("MemoryManager already initialised")
            return

        logger.info("Initialising MemoryManager …")

        # Verify Redis connectivity
        if not self.session.ping():
            logger.warning("Redis is not reachable – session memory will not work")

        # Create tables for both PostgreSQL-backed stores
        # They share the same pool connection so we only need to call one.
        await self.semantic.ensure_tables_exist()
        await self.episodic.ensure_tables_exist()

        self._initialised = True
        logger.info("MemoryManager initialised successfully")

    async def close(self) -> None:
        """Close all connections and clean up resources."""
        logger.info("Shutting down MemoryManager …")
        self.session.close()
        await self.semantic.close()
        await self.episodic.close()
        self._initialised = False
        logger.info("MemoryManager shut down")

    # ------------------------------------------------------------------
    # Full context retrieval
    # ------------------------------------------------------------------

    async def get_full_context(
        self,
        user_id: str,
        session_id: str,
        query: str,
        session_limit: int = 50,
        semantic_top_k: int = 10,
        episodic_top_k: int = 10,
    ) -> dict[str, Any]:
        """Gather the complete memory context for a user interaction.

        Concurrently fetches:
        * Recent session messages from Redis
        * Semantically-similar long-term memories from pgvector
        * Relevant episodic events from pgvector
        * Long-term Mem0 memories

        Args:
            user_id: Unique user identifier.
            session_id: Active session identifier.
            query: The current user query (used for similarity searches).
            session_limit: Number of recent session messages to include.
            semantic_top_k: Number of semantic memories to retrieve.
            episodic_top_k: Number of episodic events to retrieve.

        Returns:
            Dictionary with keys ``session_messages``, ``session_context``,
            ``semantic_memories``, ``episodic_events``, ``mem0_memories``.
        """
        # Run independent fetches concurrently
        (
            session_messages,
            session_context,
            semantic_results,
            episodic_results,
            mem0_results,
        ) = await asyncio.gather(
            # Session messages (sync Redis call run in executor)
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.session.get_session_history(user_id, session_id, session_limit),
            ),
            # Session context
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.session.get_context(user_id, session_id),
            ),
            # Semantic memory search
            self.semantic.search(user_id, query, top_k=semantic_top_k),
            # Episodic memory recall
            self.episodic.recall_events(user_id, query, limit=episodic_top_k),
            # Mem0 long-term memories (sync, run in executor)
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.long_term.search_memories(user_id, query),
            ),
        )

        result: dict[str, Any] = {
            "session_messages": session_messages,
            "session_context": session_context.to_dict() if isinstance(session_context, ConversationContext) else session_context,
            "semantic_memories": [m.to_dict() if hasattr(m, "to_dict") else m for m in semantic_results],
            "episodic_events": [e.to_dict() if hasattr(e, "to_dict") else e for e in episodic_results],
            "mem0_memories": mem0_results,
            "user_preferences": await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.session.get_key(user_id, "preferences"),
            ),
        }

        logger.debug(
            "Full context assembled  user=%s  session=%s  msgs=%d  sem=%d  epi=%d  mem0=%d",
            user_id,
            session_id,
            len(session_messages),
            len(semantic_results),
            len(episodic_results),
            len(mem0_results),
        )

        return result

    # ------------------------------------------------------------------
    # User preferences
    # ------------------------------------------------------------------

    def store_user_preference(self, user_id: str, key: str, value: Any) -> None:
        """Store a user preference (persisted in Redis).

        Preferences are stored under a single ``preferences`` key in the
        user-scoped KV store.  Multiple calls merge into the same dict.

        Args:
            user_id: Unique user identifier.
            key: Preference name.
            value: Preference value (must be JSON-serialisable).
        """
        current: dict[str, Any] = self.session.get_key(user_id, "preferences") or {}
        current[key] = value
        self.session.set_key(user_id, "preferences", current)

    def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """Read a single user preference.

        Args:
            user_id: Unique user identifier.
            key: Preference name.
            default: Default if the key is not set.

        Returns:
            The preference value or *default*.
        """
        prefs: dict[str, Any] | None = self.session.get_key(user_id, "preferences")
        if prefs is None:
            return default
        return prefs.get(key, default)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall_relevant(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant memories across all subsystems.

        Combines semantic and episodic results, deduplicates, and returns
        the top *top_k* entries.

        Args:
            user_id: Unique user identifier.
            query: Natural-language query.
            top_k: Maximum total results.

        Returns:
            List of memory dicts sorted by a composite relevance score.
        """
        semantic_results, episodic_results = await asyncio.gather(
            self.semantic.search(user_id, query, top_k=top_k),
            self.episodic.recall_events(user_id, query, limit=top_k),
        )

        # Build a unified list with a composite score
        unified: list[dict[str, Any]] = []

        for m in semantic_results:
            entry = m.to_dict() if hasattr(m, "to_dict") else m
            entry["source"] = "semantic"
            entry["composite_score"] = float(entry.get("similarity", 0.0)) * 0.6 + float(entry.get("importance_score", 0.5)) * 0.4
            unified.append(entry)

        for e in episodic_results:
            entry = e.to_dict() if hasattr(e, "to_dict") else e
            entry["source"] = "episodic"
            entry["composite_score"] = float(entry.get("similarity", 0.0)) * 0.6 + float(entry.get("importance_score", 0.5)) * 0.4
            unified.append(entry)

        # Sort by composite score descending and take top_k
        unified.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)
        return unified[:top_k]

    # ------------------------------------------------------------------
    # Create memories from an interaction
    # ------------------------------------------------------------------

    async def create_memory_from_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        agent_results: dict[str, Any],
    ) -> None:
        """Persist memories derived from a completed user interaction.

        This method:
        1. Records an episodic event for the interaction.
        2. Extracts key insights and stores them as semantic memories.
        3. Feeds the interaction to Mem0 for automatic long-term extraction.
        4. Updates session context with detected intents.

        Args:
            user_id: Unique user identifier.
            query: The user's original query.
            response: The agent's response.
            agent_results: Structured results from downstream agents (e.g.
                          portfolio data, analysis results, tool outputs).
        """
        # --- 1. Episodic event ---
        event_type = self._infer_event_type(agent_results)
        sentiment = self._infer_sentiment(agent_results)
        entities = self._extract_entities(agent_results)
        summary = f"User asked: {query[:200]}"
        outcome = response[:500] if response else ""

        try:
            await self.episodic.record_event(
                user_id=user_id,
                event_type=event_type,
                summary=summary,
                outcome=outcome,
                sentiment=sentiment,
                entities=entities,
                context={
                    "query": query,
                    "response_preview": response[:500],
                    "agent_types": list(agent_results.keys()),
                },
                importance=self._compute_event_importance(agent_results),
            )
        except Exception as exc:
            logger.error("Failed to record episodic event: %s", exc)

        # --- 2. Semantic memory ---
        insights = self._extract_insights(query, response, agent_results)
        for insight in insights:
            try:
                await self.semantic.store(
                    user_id=user_id,
                    content=insight["content"],
                    memory_type=insight.get("type", "insight"),
                    tags=insight.get("tags", []),
                    importance=insight.get("importance", 0.6),
                    metadata={
                        "source": "interaction",
                        "event_type": event_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception as exc:
                logger.error("Failed to store semantic memory: %s", exc)

        # --- 3. Mem0 long-term memory ---
        mem0_content = self._build_mem0_content(query, response, agent_results)
        if mem0_content:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.long_term.add_memory(
                        user_id=user_id,
                        content=mem0_content,
                        metadata={"source": "interaction", "event_type": event_type},
                    ),
                )
            except Exception as exc:
                logger.warning("Failed to add Mem0 memory: %s", exc)

        # --- 4. Update session context ---
        try:
            intents = self._detect_intents(query, agent_results)
            ctx_updates: dict[str, Any] = {}
            if intents:
                existing_ctx = self.session.get_context(user_id, "")  # type: ignore[arg-type]
                existing_intents = existing_ctx.detected_intents if hasattr(existing_ctx, "detected_intents") else []
                ctx_updates["detected_intents"] = list(set(existing_intents + intents))

            if ctx_updates:
                # We need session_id here but this method signature doesn't have it;
                # store intents at user level instead.
                current_intents: list[str] = self.session.get_key(user_id, "recent_intents") or []
                new_intents = list(set(current_intents + intents)) if intents else current_intents
                # Keep only last 20 intents
                self.session.set_key(user_id, "recent_intents", new_intents[-20:], ttl=self._session_ttl)
        except Exception as exc:
            logger.warning("Failed to update session context: %s", exc)

        logger.info(
            "Memories created from interaction  user=%s  event_type=%s  insights=%d",
            user_id,
            event_type,
            len(insights),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup_expired(self, user_id: str) -> dict[str, Any]:
        """Remove expired sessions and optionally consolidate memories.

        Args:
            user_id: Unique user identifier.

        Returns:
            Dictionary with cleanup statistics.
        """
        # Expire Redis sessions (sync call in executor)
        expired_sessions = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.session.cleanup_expired_sessions(user_id),
        )

        # Optionally consolidate semantic memories
        consolidated = 0
        try:
            consolidated = await self.semantic.consolidate_memories(user_id)
        except Exception as exc:
            logger.warning("Semantic consolidation failed for user %s: %s", user_id, exc)

        result = {
            "expired_sessions": len(expired_sessions),
            "session_ids": expired_sessions,
            "consolidated_semantic_memories": consolidated,
        }

        logger.info("Cleanup completed  user=%s  %s", user_id, result)
        return result

    # ------------------------------------------------------------------
    # Session shortcuts
    # ------------------------------------------------------------------

    def create_session(self, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Create a new session (delegates to RedisSessionMemory).

        Args:
            user_id: Unique user identifier.
            session_id: Optional session ID.  If ``None`` a UUID is generated.

        Returns:
            Session metadata dict.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        return self.session.create_session(user_id, session_id)

    def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a message to the current session (delegates to RedisSessionMemory)."""
        self.session.add_message(user_id, session_id, role, content, metadata)

    def get_session_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get session history (delegates to RedisSessionMemory)."""
        return self.session.get_session_history(user_id, session_id, limit)

    def clear_session(self, user_id: str, session_id: str) -> None:
        """Clear a session (delegates to RedisSessionMemory)."""
        self.session.clear_session(user_id, session_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_event_type(agent_results: dict[str, Any]) -> str:
        """Infer the event type from agent result keys."""
        if not agent_results:
            return "general_query"

        type_mapping: dict[str, str] = {
            "portfolio": "portfolio_query",
            "stock": "stock_analysis",
            "market": "market_analysis",
            "risk": "risk_assessment",
            "news": "news_retrieval",
            "screener": "stock_screening",
            "watchlist": "watchlist_update",
            "alert": "alert_management",
            "backtest": "backtest_run",
            "sentiment": "sentiment_analysis",
            "financials": "financial_analysis",
        }

        for key in agent_results:
            key_lower = key.lower()
            for pattern, event_type in type_mapping.items():
                if pattern in key_lower:
                    return event_type

        return "general_query"

    @staticmethod
    def _infer_sentiment(agent_results: dict[str, Any]) -> str:
        """Infer overall sentiment from agent results."""
        if not agent_results:
            return "neutral"

        # Check for explicit sentiment
        for _key, value in agent_results.items():
            if isinstance(value, dict):
                sent = value.get("sentiment") or value.get("overall_sentiment")
                if isinstance(sent, str) and sent.lower() in ("positive", "negative", "neutral"):
                    return sent.lower()

        return "neutral"

    @staticmethod
    def _extract_entities(agent_results: dict[str, Any]) -> dict[str, Any]:
        """Extract structured entities from agent results."""
        entities: dict[str, Any] = {
            "tickers": set(),
            "amounts": [],
            "dates": [],
        }

        for _key, value in agent_results.items():
            if isinstance(value, dict):
                # Tickers
                for ticker_key in ("ticker", "symbol", "tickers", "symbols"):
                    if ticker_key in value:
                        tickers = value[ticker_key]
                        if isinstance(tickers, str):
                            entities["tickers"].add(tickers.upper())
                        elif isinstance(tickers, (list, tuple)):
                            for t in tickers:
                                if isinstance(t, str):
                                    entities["tickers"].add(t.upper())

                # Amounts
                for amount_key in ("amount", "value", "price", "total", "gain", "loss"):
                    if amount_key in value:
                        amt = value[amount_key]
                        if isinstance(amt, (int, float)):
                            entities["amounts"].append(float(amt))

        entities["tickers"] = sorted(entities["tickers"])
        return entities

    @staticmethod
    def _compute_event_importance(agent_results: dict[str, Any]) -> float:
        """Compute an importance score for the interaction."""
        if not agent_results:
            return 0.3

        score = 0.3  # baseline

        # Higher importance for portfolio-affecting actions
        high_importance_keys = {"portfolio", "trade", "backtest", "risk"}
        for key in agent_results:
            if key.lower() in high_importance_keys:
                score += 0.2

        # Larger result sets might indicate more impactful queries
        for _key, value in agent_results.items():
            if isinstance(value, (list, dict)):
                if len(value) > 5:
                    score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _extract_insights(
        query: str,
        response: str,
        agent_results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract storable insights from an interaction.

        Returns a list of dicts with keys: content, type, tags, importance.
        """
        insights: list[dict[str, Any]] = []

        # Extract user-stated preferences from the query
        preference_indicators = [
            "i prefer", "i like", "i want", "i need", "my risk",
            "always use", "never use", "default to", "my goal",
        ]
        query_lower = query.lower()
        for indicator in preference_indicators:
            if indicator in query_lower:
                insights.append({
                    "content": f"User preference: {query.strip()}",
                    "type": "preference",
                    "tags": ["preference", "user-stated"],
                    "importance": 0.8,
                })
                break  # one preference per interaction is enough

        # Extract key financial facts from agent results
        for agent_name, result in agent_results.items():
            if not isinstance(result, dict):
                continue

            # Stock/fundamental data
            if "pe_ratio" in result or "market_cap" in result:
                ticker = result.get("ticker", result.get("symbol", "unknown"))
                fact_parts = [f"{ticker}"]
                if "pe_ratio" in result:
                    fact_parts.append(f"P/E ratio: {result['pe_ratio']}")
                if "market_cap" in result:
                    fact_parts.append(f"Market cap: {result['market_cap']}")
                if "sector" in result:
                    fact_parts.append(f"Sector: {result['sector']}")
                insights.append({
                    "content": " | ".join(fact_parts),
                    "type": "financial_fact",
                    "tags": [ticker, "fundamentals"] if ticker != "unknown" else ["fundamentals"],
                    "importance": 0.5,
                })

            # Portfolio position changes
            if "positions" in result or "holdings" in result:
                positions = result.get("positions") or result.get("holdings", [])
                if isinstance(positions, list) and positions:
                    summary = f"Portfolio has {len(positions)} positions"
                    insights.append({
                        "content": summary,
                        "type": "portfolio_state",
                        "tags": ["portfolio", "holdings"],
                        "importance": 0.4,
                    })

        return insights

    @staticmethod
    def _build_mem0_content(
        query: str,
        response: str,
        agent_results: dict[str, Any],
    ) -> str:
        """Build a text summary suitable for Mem0 ingestion."""
        parts = [f"User asked: {query}"]

        if response:
            parts.append(f"Agent responded: {response[:300]}")

        if agent_results:
            agent_keys = list(agent_results.keys())
            parts.append(f"Data sources consulted: {', '.join(agent_keys)}")

        return "\n".join(parts)

    @staticmethod
    def _detect_intents(query: str, agent_results: dict[str, Any]) -> list[str]:
        """Detect likely intents from the query and agent results."""
        intents: list[str] = []

        query_lower = query.lower()

        intent_keywords: dict[str, list[str]] = {
            "portfolio_review": ["portfolio", "holdings", "positions", "my stocks"],
            "stock_analysis": ["analyze", "analysis", "valuation", "fundamental", "technical"],
            "market_overview": ["market", "indices", "s&p", "nasdaq", "dow"],
            "risk_assessment": ["risk", "volatility", "drawdown", "beta", "sharpe"],
            "news_research": ["news", "earnings", "sec filing", "press release"],
            "screening": ["screen", "filter", "find stocks", "screener", "scan"],
            "comparison": ["compare", "vs", "versus", "difference between"],
            "education": ["what is", "explain", "how does", "definition", "meaning"],
        }

        for intent, keywords in intent_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    intents.append(intent)
                    break

        # Also infer from agent types that returned results
        for agent_key in agent_results:
            key_lower = agent_key.lower()
            if "portfolio" in key_lower and "portfolio_review" not in intents:
                intents.append("portfolio_review")
            elif "stock" in key_lower and "stock_analysis" not in intents:
                intents.append("stock_analysis")
            elif "risk" in key_lower and "risk_assessment" not in intents:
                intents.append("risk_assessment")

        return intents
