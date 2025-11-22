#!/usr/bin/env python3
"""
Create new user in the system.

Usage:
    python scripts/create_user.py <email> <password> [full_name] [--admin]

Examples:
    python scripts/create_user.py prusemic@cvut.cz myPassword123!
    python scripts/create_user.py admin@example.com SecurePass123! "Admin User" --admin

Or via Docker:
    docker-compose exec backend python scripts/create_user.py <email> <password>
"""

import asyncio
import asyncpg
import os
import sys
from argon2 import PasswordHasher

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def create_user(email: str, password: str, full_name: str = None, is_admin: bool = False):
    """Create new user with specified credentials."""

    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        print("\nSet it with:")
        print("  export DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'")
        return False

    print(f"🔄 Creating user: {email}...")

    try:
        # Connect to database
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to database")

        # Check if user already exists
        existing = await conn.fetchrow(
            "SELECT id, email FROM auth.users WHERE email = $1",
            email
        )
        if existing:
            print(f"\n⚠️  User already exists!")
            print(f"   📧 Email: {existing['email']}")
            print(f"   🆔 ID: {existing['id']}")
            await conn.close()
            return False

        # Hash password with Argon2
        ph = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16
        )
        password_hash = ph.hash(password)

        # Insert new user
        user_id = await conn.fetchval(
            """
            INSERT INTO auth.users (email, password_hash, full_name, is_active, is_admin, created_at, updated_at)
            VALUES ($1, $2, $3, true, $4, NOW(), NOW())
            RETURNING id
            """,
            email,
            password_hash,
            full_name,
            is_admin
        )

        await conn.close()

        print("\n✅ User created successfully!")
        print(f"   🆔 ID: {user_id}")
        print(f"   📧 Email: {email}")
        if full_name:
            print(f"   👤 Name: {full_name}")
        print(f"   🔑 Password: {password}")
        print(f"   👑 Admin: {'Yes' if is_admin else 'No'}")

        # Warn about weak passwords
        if len(password) < 12 or not any(c.isupper() for c in password) or not any(c in '@$!%*?&' for c in password):
            print("\n⚠️  WARNING: Weak password detected!")
            print("   Recommended: 12+ chars, uppercase, lowercase, digits, special chars (@$!%*?&)")
            print("   Please change password after first login.")

        return True

    except Exception as e:
        print(f"\n❌ User creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_usage():
    """Print usage instructions."""
    print("Usage: python scripts/create_user.py <email> <password> [full_name] [--admin]")
    print("\nExamples:")
    print("  python scripts/create_user.py user@example.com MyPass123!")
    print("  python scripts/create_user.py admin@example.com SecurePass123! 'Admin User' --admin")
    print("\nOptions:")
    print("  --admin     Create user with admin privileges")


if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    email = sys.argv[1]
    password = sys.argv[2]
    full_name = None
    is_admin = False

    # Parse optional arguments
    for arg in sys.argv[3:]:
        if arg == "--admin":
            is_admin = True
        elif not full_name:
            full_name = arg

    # Validate email format (basic check)
    if "@" not in email or "." not in email:
        print(f"❌ ERROR: Invalid email format: {email}")
        sys.exit(1)

    # Create user
    success = asyncio.run(create_user(email, password, full_name, is_admin))
    sys.exit(0 if success else 1)
