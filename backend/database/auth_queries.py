"""
User Database Operations - Single Source of Truth for User CRUD

Handles:
- User creation with email uniqueness validation
- User lookup by email/ID
- Login timestamp tracking
- Active user filtering

Usage:
    queries = AuthQueries(postgres_adapter)

    # Create user
    user_id = await queries.create_user("user@example.com", password_hash, "John Doe")

    # Get user by email (for login)
    user = await queries.get_user_by_email("user@example.com")

    # Update last login
    await queries.update_last_login(user_id)
"""

from typing import Optional, Dict, List
from datetime import datetime, timezone


class AuthQueries:
    """
    User database operations using asyncpg connection pool.

    Integrates with PostgreSQLStorageAdapter for connection pooling.
    All queries use parameterized SQL to prevent injection attacks.
    """

    def __init__(self, postgres_adapter):
        """
        Initialize auth queries with existing connection pool.

        Args:
            postgres_adapter: PostgreSQLStorageAdapter instance (from src/storage/postgres_adapter.py)
        """
        self.pool = postgres_adapter.pool

    # =========================================================================
    # User Creation
    # =========================================================================

    async def create_user(
        self,
        email: str,
        password_hash: str,
        full_name: Optional[str] = None,
        is_active: bool = True
    ) -> int:
        """
        Create new user with unique email validation.

        Args:
            email: User email (must be unique)
            password_hash: Argon2 password hash from AuthManager
            full_name: Optional display name
            is_active: Whether user can log in (default: True)

        Returns:
            User ID (SERIAL primary key)

        Raises:
            asyncpg.UniqueViolationError: If email already exists

        Example:
            >>> user_id = await queries.create_user(
            ...     "admin@sujbot.local",
            ...     "$argon2id$v=19$m=65536,t=3,p=4$...",
            ...     "System Administrator"
            ... )
            >>> print(user_id)
            1
        """
        async with self.pool.acquire() as conn:
            user_id = await conn.fetchval(
                """
                INSERT INTO auth.users (email, password_hash, full_name, is_active, created_at, updated_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                RETURNING id
                """,
                email,
                password_hash,
                full_name,
                is_active
            )
            return user_id

    # =========================================================================
    # User Lookup
    # =========================================================================

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        Get user by email address (for login).

        Args:
            email: User email to look up

        Returns:
            User dict with keys: id, email, password_hash, full_name, is_active,
            created_at, updated_at, last_login_at
            None if user not found

        Example:
            >>> user = await queries.get_user_by_email("admin@sujbot.local")
            >>> if user and user['is_active']:
            ...     print(f"User {user['id']} found")
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, password_hash, full_name, is_active,
                       created_at, updated_at, last_login_at
                FROM auth.users
                WHERE email = $1
                """,
                email
            )
            return dict(row) if row else None

    async def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        Get user by ID (for token validation).

        Args:
            user_id: User ID from JWT token

        Returns:
            User dict (same structure as get_user_by_email)
            None if user not found

        Example:
            >>> user = await queries.get_user_by_id(1)
            >>> if user:
            ...     print(f"User: {user['email']}")
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, password_hash, full_name, is_active,
                       created_at, updated_at, last_login_at
                FROM auth.users
                WHERE id = $1
                """,
                user_id
            )
            return dict(row) if row else None

    async def get_active_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        Get active user by ID (for middleware authentication).

        Similar to get_user_by_id but filters out inactive users.

        Args:
            user_id: User ID from JWT token

        Returns:
            User dict if active, None if not found or inactive

        Example:
            >>> user = await queries.get_active_user_by_id(1)
            >>> if user:
            ...     # User is active and can access protected routes
            ...     pass
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, password_hash, full_name, is_active,
                       created_at, updated_at, last_login_at
                FROM auth.users
                WHERE id = $1 AND is_active = true
                """,
                user_id
            )
            return dict(row) if row else None

    # =========================================================================
    # User Updates
    # =========================================================================

    async def update_last_login(self, user_id: int) -> None:
        """
        Update user's last login timestamp.

        Called after successful login/token validation.

        Args:
            user_id: User ID to update

        Example:
            >>> await queries.update_last_login(1)
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE auth.users
                SET last_login_at = NOW()
                WHERE id = $1
                """,
                user_id
            )

    async def deactivate_user(self, user_id: int) -> None:
        """
        Deactivate user (soft delete - prevents login).

        Args:
            user_id: User ID to deactivate

        Example:
            >>> await queries.deactivate_user(1)
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE auth.users
                SET is_active = false, updated_at = NOW()
                WHERE id = $1
                """,
                user_id
            )

    async def activate_user(self, user_id: int) -> None:
        """
        Activate user (enable login).

        Args:
            user_id: User ID to activate

        Example:
            >>> await queries.activate_user(1)
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE auth.users
                SET is_active = true, updated_at = NOW()
                WHERE id = $1
                """,
                user_id
            )

    # =========================================================================
    # User Listing (for admin)
    # =========================================================================

    async def list_users(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        List all users (for admin interface).

        Args:
            limit: Maximum users to return (default: 100)
            offset: Pagination offset (default: 0)

        Returns:
            List of user dicts (excludes password_hash for security)

        Example:
            >>> users = await queries.list_users(limit=10)
            >>> for user in users:
            ...     print(f"{user['email']} - Active: {user['is_active']}")
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, email, full_name, is_active,
                       created_at, updated_at, last_login_at
                FROM auth.users
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset
            )
            return [dict(row) for row in rows]

    async def count_users(self) -> int:
        """
        Get total user count (for pagination).

        Returns:
            Total number of users

        Example:
            >>> total = await queries.count_users()
            >>> print(f"Total users: {total}")
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM auth.users")
