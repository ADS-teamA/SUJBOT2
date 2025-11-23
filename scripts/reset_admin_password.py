#!/usr/bin/env python3
"""
Reset admin user password to a known value.

Usage:
    python scripts/reset_admin_password.py

Or via Docker:
    docker-compose exec backend python scripts/reset_admin_password.py
"""

import asyncio
import asyncpg
import os
import sys
from argon2 import PasswordHasher

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def reset_password():
    """Reset admin password."""

    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        return False

    print("🔄 Resetting admin password...")

    try:
        # Connect to database
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to database")

        # Hash new password
        ph = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16
        )

        new_password = "ChangeThisPassword123!"
        password_hash = ph.hash(new_password)

        # Update admin user
        result = await conn.execute(
            """
            UPDATE auth.users
            SET password_hash = $1, updated_at = NOW()
            WHERE email = 'admin@example.com'
            """,
            password_hash
        )

        await conn.close()

        print("\n✅ Password reset successfully!")
        print("   📧 Email: admin@example.com")
        print("   🔑 Password: ChangeThisPassword123!")

        return True

    except Exception as e:
        print(f"\n❌ Password reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(reset_password())
    sys.exit(0 if success else 1)
