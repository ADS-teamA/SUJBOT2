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
import logging
import asyncpg

logger = logging.getLogger(__name__)


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

    def _handle_db_error(self, operation: str, context: Dict, error: Exception) -> None:
        """Helper to log and re-raise database errors with context."""
        if isinstance(error, asyncpg.PostgresConnectionError):
            logger.error(
                f"Database connection error during {operation}",
                exc_info=True,
                extra={**context, "error_type": "ConnectionError"}
            )
            raise RuntimeError(f"Database connection failed: {error}") from error
        else:
            logger.error(
                f"Unexpected error during {operation}: {error}",
                exc_info=True,
                extra={**context, "error_type": error.__class__.__name__}
            )
            raise

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
            RuntimeError: If database connection fails

        Example:
            >>> user_id = await queries.create_user(
            ...     "admin@sujbot.local",
            ...     "$argon2id$v=19$m=65536,t=3,p=4$...",
            ...     "System Administrator"
            ... )
            >>> print(user_id)
            1
        """
        try:
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
                logger.info(f"Created user {user_id} with email {email}")
                return user_id
        except asyncpg.UniqueViolationError:
            logger.warning(f"Failed to create user: email {email} already exists")
            raise
        except asyncpg.PostgresConnectionError as e:
            logger.error(
                f"Database connection error while creating user {email}",
                exc_info=True,
                extra={"email": email, "error_type": "ConnectionError"}
            )
            raise RuntimeError(f"Database connection failed: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while creating user {email}: {e}",
                exc_info=True,
                extra={"email": email, "error_type": e.__class__.__name__}
            )
            raise

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

        Raises:
            RuntimeError: If database connection fails

        Example:
            >>> user = await queries.get_user_by_email("admin@sujbot.local")
            >>> if user and user['is_active']:
            ...     print(f"User {user['id']} found")
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, email, password_hash, full_name, is_active, is_admin,
                           created_at, updated_at, last_login_at
                    FROM auth.users
                    WHERE email = $1
                    """,
                    email
                )
                return dict(row) if row else None
        except asyncpg.PostgresConnectionError as e:
            logger.error(
                f"Database connection error while fetching user by email",
                exc_info=True,
                extra={"email": email, "error_type": "ConnectionError"}
            )
            raise RuntimeError(f"Database connection failed: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while fetching user by email {email}: {e}",
                exc_info=True,
                extra={"email": email, "error_type": e.__class__.__name__}
            )
            raise

    async def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        Get user by ID (for token validation).

        Args:
            user_id: User ID from JWT token

        Returns:
            User dict (same structure as get_user_by_email)
            None if user not found

        Raises:
            RuntimeError: If database connection fails

        Example:
            >>> user = await queries.get_user_by_id(1)
            >>> if user:
            ...     print(f"User: {user['email']}")
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, email, password_hash, full_name, is_active, is_admin,
                           created_at, updated_at, last_login_at
                    FROM auth.users
                    WHERE id = $1
                    """,
                    user_id
                )
                return dict(row) if row else None
        except Exception as e:
            self._handle_db_error("get_user_by_id", {"user_id": user_id}, e)

    async def get_active_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        Get active user by ID (for middleware authentication).

        Similar to get_user_by_id but filters out inactive users.

        Args:
            user_id: User ID from JWT token

        Returns:
            User dict if active, None if not found or inactive

        Raises:
            RuntimeError: If database connection fails

        Example:
            >>> user = await queries.get_active_user_by_id(1)
            >>> if user:
            ...     # User is active and can access protected routes
            ...     pass
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, email, password_hash, full_name, is_active, is_admin,
                           created_at, updated_at, last_login_at
                    FROM auth.users
                    WHERE id = $1 AND is_active = true
                    """,
                    user_id
                )
                return dict(row) if row else None
        except Exception as e:
            self._handle_db_error("get_active_user_by_id", {"user_id": user_id}, e)

    # =========================================================================
    # User Updates
    # =========================================================================

    async def update_last_login(self, user_id: int) -> None:
        """
        Update user's last login timestamp.

        Called after successful login/token validation.

        Args:
            user_id: User ID to update

        Raises:
            RuntimeError: If database connection fails

        Example:
            >>> await queries.update_last_login(1)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE auth.users
                    SET last_login_at = NOW()
                    WHERE id = $1
                    """,
                    user_id
                )
                logger.debug(f"Updated last_login for user {user_id}")
        except Exception as e:
            # Log but don't fail the request - this is non-critical
            logger.warning(
                f"Failed to update last_login for user {user_id}: {e}",
                extra={"user_id": user_id, "error_type": e.__class__.__name__}
            )

    async def update_password(self, user_id: int, password_hash: str) -> None:
        """
        Update user's password hash.

        Args:
            user_id: User ID to update
            password_hash: New Argon2 password hash

        Raises:
            RuntimeError: If database connection fails
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE auth.users
                    SET password_hash = $2, updated_at = NOW()
                    WHERE id = $1
                    """,
                    user_id,
                    password_hash
                )
                logger.info(f"Password updated for user {user_id}")
        except Exception as e:
            self._handle_db_error("update_password", {"user_id": user_id}, e)

    async def deactivate_user(self, user_id: int) -> None:
        """
        Deactivate user (soft delete - prevents login).

        Args:
            user_id: User ID to deactivate

        Raises:
            RuntimeError: If database connection fails

        Example:
            >>> await queries.deactivate_user(1)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE auth.users
                    SET is_active = false, updated_at = NOW()
                    WHERE id = $1
                    """,
                    user_id
                )
                logger.info(f"Deactivated user {user_id}")
        except Exception as e:
            self._handle_db_error("deactivate_user", {"user_id": user_id}, e)

    async def activate_user(self, user_id: int) -> None:
        """
        Activate user (enable login).

        Args:
            user_id: User ID to activate

        Raises:
            RuntimeError: If database connection fails

        Example:
            >>> await queries.activate_user(1)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE auth.users
                    SET is_active = true, updated_at = NOW()
                    WHERE id = $1
                    """,
                    user_id
                )
                logger.info(f"Activated user {user_id}")
        except Exception as e:
            self._handle_db_error("activate_user", {"user_id": user_id}, e)

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

        Raises:
            RuntimeError: If database connection fails

        Example:
            >>> users = await queries.list_users(limit=10)
            >>> for user in users:
            ...     print(f"{user['email']} - Active: {user['is_active']}")
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, email, full_name, is_active, is_admin,
                           agent_variant, spending_limit_czk, total_spent_czk,
                           spending_reset_at, created_at, updated_at, last_login_at
                    FROM auth.users
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset
                )
                return [dict(row) for row in rows]
        except Exception as e:
            self._handle_db_error("list_users", {"limit": limit, "offset": offset}, e)

    async def count_users(self) -> int:
        """
        Get total user count (for pagination).

        Returns:
            Total number of users

        Raises:
            RuntimeError: If database connection fails

        Example:
            >>> total = await queries.count_users()
            >>> print(f"Total users: {total}")
        """
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval("SELECT COUNT(*) FROM auth.users")
        except Exception as e:
            self._handle_db_error("count_users", {}, e)

    async def get_agent_variant(self, user_id: int) -> str:
        """
        Get user's agent variant preference.

        Args:
            user_id: User ID

        Returns:
            Agent variant ('premium', 'cheap', or 'local'), defaults to 'cheap'
        """
        try:
            async with self.pool.acquire() as conn:
                variant = await conn.fetchval(
                    "SELECT agent_variant FROM auth.users WHERE id = $1",
                    user_id
                )
                return variant or "cheap"
        except Exception as e:
            # _handle_db_error always raises, so this is the error path
            self._handle_db_error("get_agent_variant", {"user_id": user_id}, e)
            # Note: _handle_db_error raises, so code below is unreachable
            raise  # Make unreachable explicit for type checker

    async def update_agent_variant(self, user_id: int, variant: str) -> None:
        """
        Update user's agent variant preference.

        Args:
            user_id: User ID
            variant: Agent variant ('premium', 'cheap', or 'local')

        Raises:
            ValueError: If variant is not valid
        """
        if variant not in ["premium", "cheap", "local"]:
            raise ValueError(f"Invalid variant: {variant}. Must be 'premium', 'cheap', or 'local'")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE auth.users
                    SET agent_variant = $1, updated_at = NOW()
                    WHERE id = $2
                    """,
                    variant,
                    user_id
                )
                logger.info(f"User {user_id} switched to variant: {variant}")
        except Exception as e:
            self._handle_db_error("update_agent_variant", {"user_id": user_id, "variant": variant}, e)

    # =========================================================================
    # Spending Tracking
    # =========================================================================

    async def get_user_spending(self, user_id: int) -> Dict:
        """
        Get user's current spending status.

        Args:
            user_id: User ID

        Returns:
            Dict with:
                - total_spent_czk: Current total spending in CZK
                - spending_limit_czk: Admin-set limit in CZK
                - remaining_czk: Remaining budget (limit - spent)
                - reset_at: ISO timestamp of last reset

        Example:
            >>> spending = await queries.get_user_spending(1)
            >>> print(f"Spent: {spending['total_spent_czk']} / {spending['spending_limit_czk']} CZK")
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT spending_limit_czk, total_spent_czk, spending_reset_at
                    FROM auth.users WHERE id = $1
                    """,
                    user_id
                )
                if not row:
                    # Return defaults for non-existent user
                    return {
                        "total_spent_czk": 0.0,
                        "spending_limit_czk": 500.0,
                        "remaining_czk": 500.0,
                        "reset_at": None
                    }

                limit = float(row["spending_limit_czk"] or 500.0)
                spent = float(row["total_spent_czk"] or 0.0)

                return {
                    "total_spent_czk": spent,
                    "spending_limit_czk": limit,
                    "remaining_czk": max(0, limit - spent),
                    "reset_at": row["spending_reset_at"].isoformat() if row["spending_reset_at"] else None
                }
        except Exception as e:
            self._handle_db_error("get_user_spending", {"user_id": user_id}, e)

    async def add_spending(self, user_id: int, cost_czk: float) -> bool:
        """
        Add spending to user's total.

        Uses atomic UPDATE to prevent race conditions.
        Will NOT update if doing so would exceed the limit.

        Args:
            user_id: User ID
            cost_czk: Cost to add in CZK

        Returns:
            True if spending was added successfully
            False if adding would exceed limit (spending NOT added)

        Example:
            >>> success = await queries.add_spending(1, 0.15)
            >>> if not success:
            ...     print("User has exceeded spending limit")
        """
        try:
            async with self.pool.acquire() as conn:
                # Atomic update that checks limit
                result = await conn.fetchval(
                    """
                    UPDATE auth.users
                    SET total_spent_czk = total_spent_czk + $2,
                        updated_at = NOW()
                    WHERE id = $1
                      AND (total_spent_czk + $2) <= spending_limit_czk
                    RETURNING id
                    """,
                    user_id,
                    cost_czk
                )
                if result is not None:
                    logger.debug(f"Added {cost_czk:.2f} CZK spending for user {user_id}")
                    return True
                else:
                    logger.warning(f"User {user_id} would exceed spending limit with +{cost_czk:.2f} CZK")
                    return False
        except Exception as e:
            self._handle_db_error("add_spending", {"user_id": user_id, "cost_czk": cost_czk}, e)

    async def check_spending_limit(self, user_id: int, estimated_cost_czk: float = 0.0) -> bool:
        """
        Check if user can make a request (hasn't exceeded limit).

        Args:
            user_id: User ID
            estimated_cost_czk: Estimated cost of upcoming request (optional)

        Returns:
            True if user can proceed (under limit)
            False if user has exceeded or would exceed limit

        Example:
            >>> if not await queries.check_spending_limit(user_id):
            ...     raise HTTPException(status_code=402, detail="Spending limit exceeded")
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT spending_limit_czk, total_spent_czk
                    FROM auth.users WHERE id = $1
                    """,
                    user_id
                )
                if not row:
                    # User not found - allow (will fail on auth anyway)
                    return True

                limit = float(row["spending_limit_czk"] or 500.0)
                spent = float(row["total_spent_czk"] or 0.0)

                # Check if current spending + estimated would exceed limit
                can_proceed = (spent + estimated_cost_czk) <= limit

                if not can_proceed:
                    logger.info(
                        f"User {user_id} blocked: {spent:.2f}/{limit:.2f} CZK "
                        f"(+{estimated_cost_czk:.2f} estimated)"
                    )

                return can_proceed
        except Exception as e:
            self._handle_db_error("check_spending_limit", {"user_id": user_id}, e)

    async def reset_user_spending(self, user_id: int) -> None:
        """
        Reset user's total spending to zero.

        Called by admin to reset a user's spending counter.

        Args:
            user_id: User ID

        Example:
            >>> await queries.reset_user_spending(1)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE auth.users
                    SET total_spent_czk = 0,
                        spending_reset_at = NOW(),
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    user_id
                )
                logger.info(f"Reset spending for user {user_id}")
        except Exception as e:
            self._handle_db_error("reset_user_spending", {"user_id": user_id}, e)
