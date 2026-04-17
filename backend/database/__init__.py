"""
Database package — async PostgreSQL connection management and schema initialization.
"""

from backend.database.connection import DatabaseManager

__all__ = ["DatabaseManager"]
