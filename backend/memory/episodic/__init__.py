"""
Episodic memory store.

Provides storage and retrieval of discrete interaction events with
temporal, sentiment-based, and semantic similarity recall.
"""

from backend.memory.episodic.episodic_memory import (
    EpisodicMemoryStore,
    MemoryEntry,
)

__all__ = [
    "EpisodicMemoryStore",
    "MemoryEntry",
]
