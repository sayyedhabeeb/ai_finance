import asyncio
import os
from backend.database.connection import DatabaseManager
from backend.config.settings import get_settings

async def test_db_conn():
    settings = get_settings()
    print(f"Testing connection to: {settings.database_url}")
    db = DatabaseManager(dsn=settings.database_url, command_timeout=5)
    try:
        await db.initialize_pool()
        print("Successfully connected to database!")
        await db.close_pool()
    except Exception as e:
        print(f"Database connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_db_conn())
