"""
Migration: Add agent_variant column to auth.users table.

This migration adds support for per-user agent variant preferences.
Users can choose between:
- 'premium': Claude Haiku 4.5 (default)
- 'local': Qwen 2.5-72B Instruct via DeepInfra

Usage:
    uv run python scripts/migrate_add_agent_variant.py
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate():
    """Add agent_variant column to auth.users table."""
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
            # Add agent_variant column
            await conn.execute("""
                ALTER TABLE auth.users
                ADD COLUMN IF NOT EXISTS agent_variant VARCHAR(20) DEFAULT 'premium'
                CHECK (agent_variant IN ('premium', 'local'))
            """)
            logger.info("✓ Migration completed - agent_variant column added")

            # Verify column was added
            result = await conn.fetchval("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'auth'
                  AND table_name = 'users'
                  AND column_name = 'agent_variant'
            """)

            if result:
                logger.info("✓ Verification passed - column exists")
                return True
            else:
                logger.error("✗ Verification failed - column not found")
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
