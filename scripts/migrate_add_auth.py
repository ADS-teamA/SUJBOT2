#!/usr/bin/env python3
"""
Migration script to add authentication schema to existing SUJBOT2 database.

This script:
1. Creates auth schema and tables if they don't exist
2. Creates default admin user with Argon2 hashed password
3. Handles existing data gracefully (idempotent)

Usage:
    python scripts/migrate_add_auth.py

Or via Docker:
    docker-compose exec backend python scripts/migrate_add_auth.py
"""

import asyncio
import asyncpg
import os
import sys
from argon2 import PasswordHasher
from argon2.exceptions import HashingError

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def migrate_database():
    """Run migration to add auth schema."""

    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        print("   Set it in .env file or pass directly:")
        print("   DATABASE_URL=postgresql://... python scripts/migrate_add_auth.py")
        return False

    print("🔄 Starting authentication schema migration...")
    print(f"   Database: {db_url.split('@')[-1]}")  # Hide credentials

    try:
        # Connect to database
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to database")

        # Check if auth schema already exists
        schema_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'auth')"
        )

        if schema_exists:
            print("⚠️  Auth schema already exists - checking tables...")
        else:
            print("📝 Creating auth schema...")

        # Create auth schema
        await conn.execute("CREATE SCHEMA IF NOT EXISTS auth")

        # Create users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS auth.users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(100),
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_login_at TIMESTAMP WITH TIME ZONE
            )
        """)
        print("✅ Users table ready")

        # Create indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_email ON auth.users(email)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_active ON auth.users(is_active) WHERE is_active = true"
        )

        # Create conversations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS auth.conversations (
                id VARCHAR(36) PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
                title VARCHAR(500),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        print("✅ Conversations table ready")

        # Create conversation indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON auth.conversations(user_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON auth.conversations(updated_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON auth.conversations(user_id, updated_at DESC)"
        )

        # Create messages table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS auth.messages (
                id SERIAL PRIMARY KEY,
                conversation_id VARCHAR(36) NOT NULL REFERENCES auth.conversations(id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        print("✅ Messages table ready")

        # Create message indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON auth.messages(conversation_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON auth.messages(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON auth.messages(conversation_id, created_at ASC)"
        )

        # Create trigger function for updated_at
        await conn.execute("""
            CREATE OR REPLACE FUNCTION auth.update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)

        # Create triggers
        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname = 'update_users_updated_at'
                ) THEN
                    CREATE TRIGGER update_users_updated_at
                        BEFORE UPDATE ON auth.users
                        FOR EACH ROW
                        EXECUTE FUNCTION auth.update_updated_at_column();
                END IF;
            END $$
        """)

        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname = 'update_conversations_updated_at'
                ) THEN
                    CREATE TRIGGER update_conversations_updated_at
                        BEFORE UPDATE ON auth.conversations
                        FOR EACH ROW
                        EXECUTE FUNCTION auth.update_updated_at_column();
                END IF;
            END $$
        """)
        print("✅ Triggers configured")

        # Create default admin user if doesn't exist
        admin_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM auth.users WHERE email = 'admin@sujbot.local')"
        )

        if not admin_exists:
            print("👤 Creating default admin user...")

            # Hash default password with Argon2
            ph = PasswordHasher(
                time_cost=3,        # iterations
                memory_cost=65536,  # 64 MB
                parallelism=4,      # threads
                hash_len=32,        # output length
                salt_len=16         # salt length
            )

            default_password = "adssujbot"
            password_hash = ph.hash(default_password)

            await conn.execute(
                """
                INSERT INTO auth.users (email, password_hash, full_name, is_active)
                VALUES ($1, $2, $3, $4)
                """,
                'admin@sujbot.local',
                password_hash,
                'System Administrator',
                True
            )

            print("✅ Default admin user created")
            print("   📧 Email: admin@sujbot.local")
            print("   🔑 Password: adssujbot")
        else:
            print("ℹ️  Admin user already exists - skipping creation")

        # Get statistics
        user_count = await conn.fetchval("SELECT COUNT(*) FROM auth.users")
        conv_count = await conn.fetchval("SELECT COUNT(*) FROM auth.conversations")
        msg_count = await conn.fetchval("SELECT COUNT(*) FROM auth.messages")

        await conn.close()

        print("\n✅ Migration completed successfully!")
        print(f"   Users: {user_count}")
        print(f"   Conversations: {conv_count}")
        print(f"   Messages: {msg_count}")

        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(migrate_database())
    sys.exit(0 if success else 1)
