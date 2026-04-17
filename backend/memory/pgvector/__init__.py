"""
Pgvector-backed semantic memory.

Provides long-term vector-similarity search over stored memories with
importance scoring, tag-based retrieval, and automatic consolidation
of semantically-duplicate entries.
"""

from .semantic_memory import (
    SemanticMemoryStore,
    MemoryEntry,
    _generate_embedding,
)

__all__ = [
    "SemanticMemoryStore",
    "MemoryEntry",
]
