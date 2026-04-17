"""
AI Financial Brain – Memory System.

This package provides a unified, multi-layered memory architecture for
the AI Financial Brain:

* **Redis Session Memory** – short-term conversation history with automatic
  TTL and message windowing.

* **Pgvector Semantic Memory** – long-term vector-similarity-based retrieval
  of facts, preferences, and insights.

* **Mem0 Long-Term Memory** – automatically-managed persistent memory via
  the Mem0 platform.

* **Episodic Memory** – timestamped event records supporting temporal,
  sentiment, and similarity-based recall.

* **MemoryManager** – unified coordinator that ties all subsystems together
  into a single, easy-to-use interface.
"""

from .memory_manager import MemoryManager
from .redis.session_memory import RedisSessionMemory, ConversationContext, Message
from .pgvector.semantic_memory import SemanticMemoryStore, MemoryEntry as SemanticMemoryEntry
from .mem0.long_term_memory import Mem0LongTermMemory
from .episodic.episodic_memory import EpisodicMemoryStore, MemoryEntry as EpisodicMemoryEntry

__all__ = [
    # Unified
    "MemoryManager",
    # Redis session
    "RedisSessionMemory",
    "ConversationContext",
    "Message",
    # Pgvector semantic
    "SemanticMemoryStore",
    "SemanticMemoryEntry",
    # Mem0 long-term
    "Mem0LongTermMemory",
    # Episodic
    "EpisodicMemoryStore",
    "EpisodicMemoryEntry",
]
