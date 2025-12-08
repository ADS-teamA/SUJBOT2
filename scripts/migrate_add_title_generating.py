"""
Migration: Add is_title_generating column to auth.conversations table.

This migration adds support for race condition protection during LLM title generation.
The flag prevents multiple workers from generating titles simultaneously.

Usage:
    uv run python scripts/migrate_add_title_generating.py
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate():
    """Add is_title_generating column to auth.conversations table."""
    # Load environment variables
    load_dotenv()
    connection_string = os.getenv("DATABASE_URL")

    if not connection_string:
        logger.error("DATABASE_URL not found in environment")
        return False

    try:
        # Connect to database
        conn = await asyncpg.connect(connection_string)
        logger.info("Connected to database")

        try:
            # Add is_title_generating column
            await conn.execute("""
                ALTER TABLE auth.conversations
                ADD COLUMN IF NOT EXISTS is_title_generating BOOLEAN NOT NULL DEFAULT false
            """)
            logger.info("Added is_title_generating column")

            # Verify column was added
            result = await conn.fetchval("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'auth'
                  AND table_name = 'conversations'
                  AND column_name = 'is_title_generating'
            """)

            if result:
                logger.info("Migration completed - is_title_generating column added")
                return True
            else:
                logger.error("Verification failed - column not found")
                return False

        finally:
            await conn.close()
            logger.info("Database connection closed")

    except asyncpg.PostgresError as e:
        logger.error(f"Database error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(migrate())
    exit(0 if success else 1)
