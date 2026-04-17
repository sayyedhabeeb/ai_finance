"""
Mem0-backed long-term memory.

Provides intelligent, automatically-managed long-term memory through the
Mem0 platform, including deduplication and user-profile maintenance.
"""

from backend.memory.mem0.long_term_memory import Mem0LongTermMemory

__all__ = [
    "Mem0LongTermMemory",
]
