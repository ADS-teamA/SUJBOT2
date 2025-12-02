"""
Admin Database Operations - Single Source of Truth for Admin User Management

Handles:
- Full user CRUD with admin-level access
- System statistics aggregation
- Admin count validation (prevent last admin removal)

Usage:
    queries = AdminQueries(postgres_adapter)

    # Get full user details
    user = await queries.get_user_full(user_id)

    # Update user
    success = await queries.update_user(user_id, email="new@email.com", is_admin=True)

    # Check if last admin
    if await queries.is_last_admin(user_id):
        raise HTTPException(400, "Cannot demote the last admin")
"""

from typing import Optional, Dict, List
from datetime import datetime, timezone
import logging
import asyncpg

logger = logging.getLogger(__name__)


class AdminQueries:
    """
    Admin-level database operations.

    Extends AuthQueries functionality with admin-specific operations
    like user updates, deletions, and system statistics.
    """

    def __init__(self, postgres_adapter):
        """
        Initialize admin queries with existing connection pool.

        Args:
            postgres_adapter: PostgreSQLStorageAdapter instance
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
    # User CRUD Operations
    # =========================================================================

    async def get_user_full(self, user_id: int) -> Optional[Dict]:
        """
        Get full user details (admin view).

        Returns all fields including agent_variant.

        Args:
            user_id: User ID to fetch

        Returns:
            User dict with all fields, or None if not found

        Raises:
            RuntimeError: If database connection fails
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, email, full_name, is_active, is_admin,
                           agent_variant, created_at, updated_at, last_login_at
                    FROM auth.users
                    WHERE id = $1
                    """,
                    user_id
                )
                return dict(row) if row else None
        except Exception as e:
            self._handle_db_error("get_user_full", {"user_id": user_id}, e)

    async def update_user(
        self,
        user_id: int,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        is_admin: Optional[bool] = None,
        is_active: Optional[bool] = None,
        agent_variant: Optional[str] = None
    ) -> bool:
        """
        Update user fields (admin operation).

        Only updates provided (non-None) fields.

        Args:
            user_id: User ID to update
            email: New email address
            full_name: New display name
            is_admin: Admin privileges
            is_active: Account active status
            agent_variant: Model preference ('premium', 'cheap', 'local')

        Returns:
            True if user was updated, False if not found

        Raises:
            asyncpg.UniqueViolationError: If email already exists
            RuntimeError: If database connection fails
        """
        # Build dynamic UPDATE query
        updates = []
        params = []
        param_idx = 1

        if email is not None:
            updates.append(f"email = ${param_idx}")
            params.append(email)
            param_idx += 1
        if full_name is not None:
            updates.append(f"full_name = ${param_idx}")
            params.append(full_name)
            param_idx += 1
        if is_admin is not None:
            updates.append(f"is_admin = ${param_idx}")
            params.append(is_admin)
            param_idx += 1
        if is_active is not None:
            updates.append(f"is_active = ${param_idx}")
            params.append(is_active)
            param_idx += 1
        if agent_variant is not None:
            updates.append(f"agent_variant = ${param_idx}")
            params.append(agent_variant)
            param_idx += 1

        if not updates:
            return True  # Nothing to update

        # Always update updated_at
        updates.append("updated_at = NOW()")

        # Add user_id as last parameter
        params.append(user_id)

        query = f"""
            UPDATE auth.users
            SET {', '.join(updates)}
            WHERE id = ${param_idx}
            RETURNING id
        """

        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(query, *params)
                if result:
                    logger.info(f"Updated user {user_id}")
                return result is not None
        except asyncpg.UniqueViolationError:
            logger.warning(f"Email already exists during user update for user {user_id}")
            raise
        except Exception as e:
            self._handle_db_error("update_user", {"user_id": user_id}, e)

    async def delete_user(self, user_id: int) -> bool:
        """
        Delete user (hard delete).

        Note: This cascades to conversations/messages if foreign keys are set up.

        Args:
            user_id: User ID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            RuntimeError: If database connection fails
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM auth.users WHERE id = $1",
                    user_id
                )
                deleted = result == "DELETE 1"
                if deleted:
                    logger.info(f"Deleted user {user_id}")
                return deleted
        except Exception as e:
            self._handle_db_error("delete_user", {"user_id": user_id}, e)

    # =========================================================================
    # Admin Safety Checks
    # =========================================================================

    async def count_admins(self) -> int:
        """
        Count total active admin users.

        Used for last-admin protection.

        Returns:
            Number of active admin users

        Raises:
            RuntimeError: If database connection fails
        """
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM auth.users WHERE is_admin = true AND is_active = true"
                )
        except Exception as e:
            self._handle_db_error("count_admins", {}, e)

    async def is_last_admin(self, user_id: int) -> bool:
        """
        Check if user is the last active admin.

        Used to prevent demoting/deleting the last admin.

        Args:
            user_id: User ID to check

        Returns:
            True if this is the only active admin

        Raises:
            RuntimeError: If database connection fails
        """
        try:
            async with self.pool.acquire() as conn:
                # Check if user is admin
                is_admin = await conn.fetchval(
                    "SELECT is_admin FROM auth.users WHERE id = $1",
                    user_id
                )
                if not is_admin:
                    return False

                # Count other active admins
                other_admins = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM auth.users
                    WHERE is_admin = true AND is_active = true AND id != $1
                    """,
                    user_id
                )
                return other_admins == 0
        except Exception as e:
            self._handle_db_error("is_last_admin", {"user_id": user_id}, e)

    # =========================================================================
    # System Statistics
    # =========================================================================

    async def get_system_stats(self) -> Dict:
        """
        Get system-wide statistics for admin dashboard.

        Returns:
            Dict with user and conversation counts

        Raises:
            RuntimeError: If database connection fails
        """
        try:
            async with self.pool.acquire() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM auth.users) as total_users,
                        (SELECT COUNT(*) FROM auth.users WHERE is_active = true) as active_users,
                        (SELECT COUNT(*) FROM auth.users WHERE is_admin = true AND is_active = true) as admin_users,
                        (SELECT COUNT(*) FROM auth.conversations) as total_conversations,
                        (SELECT COUNT(*) FROM auth.messages) as total_messages,
                        (SELECT COUNT(*) FROM auth.users
                         WHERE last_login_at > NOW() - INTERVAL '24 hours') as users_last_24h
                    """
                )
                return dict(stats)
        except Exception as e:
            self._handle_db_error("get_system_stats", {}, e)

    # =========================================================================
    # Health Check Helpers
    # =========================================================================

    async def check_database_health(self) -> Dict:
        """
        Check PostgreSQL database health.

        Returns:
            Dict with status, latency_ms, and optional message

        Example:
            >>> health = await queries.check_database_health()
            >>> print(health)
            {'status': 'healthy', 'latency_ms': 2.5}
        """
        import time

        try:
            start = time.perf_counter()
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            latency_ms = (time.perf_counter() - start) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2)
            }
        except asyncpg.PostgresConnectionError as e:
            return {
                "status": "unhealthy",
                "latency_ms": None,
                "message": f"Connection failed: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "degraded",
                "latency_ms": None,
                "message": str(e)
            }
