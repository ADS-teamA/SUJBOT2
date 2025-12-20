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

from typing import Optional, Dict, List, NoReturn
from datetime import datetime, timezone
import logging
import json
import asyncpg

from src.exceptions import DatabaseConnectionError, StorageError

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

    def _handle_db_error(self, operation: str, context: Dict, error: Exception) -> NoReturn:
        """Helper to log and re-raise database errors with typed exceptions."""
        if isinstance(error, asyncpg.PostgresConnectionError):
            logger.error(
                f"Database connection error during {operation}",
                exc_info=True,
                extra={**context, "error_type": "ConnectionError"}
            )
            raise DatabaseConnectionError(
                message=f"Database connection failed during {operation}",
                details=context,
                cause=error
            )
        else:
            logger.error(
                f"Unexpected error during {operation}: {error}",
                exc_info=True,
                extra={**context, "error_type": error.__class__.__name__}
            )
            raise StorageError(
                message=f"Unexpected error during {operation}: {error}",
                details={**context, "error_type": error.__class__.__name__},
                cause=error
            )

    # =========================================================================
    # User CRUD Operations
    # =========================================================================

    async def get_user_full(self, user_id: int) -> Optional[Dict]:
        """
        Get full user details (admin view).

        Returns all fields including agent_variant and spending info.

        Args:
            user_id: User ID to fetch

        Returns:
            User dict with all fields, or None if not found

        Raises:
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, email, full_name, is_active, is_admin,
                           agent_variant, created_at, updated_at, last_login_at,
                           spending_limit_czk, total_spent_czk, spending_reset_at
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
        agent_variant: Optional[str] = None,
        spending_limit_czk: Optional[float] = None
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
            spending_limit_czk: Spending limit in CZK

        Returns:
            True if user was updated, False if not found

        Raises:
            asyncpg.UniqueViolationError: If email already exists
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
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
        if spending_limit_czk is not None:
            updates.append(f"spending_limit_czk = ${param_idx}")
            params.append(spending_limit_czk)
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
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
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
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
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
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
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
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
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
                         WHERE last_login_at > NOW() - INTERVAL '24 hours') as users_last_24h,
                        -- Spending statistics
                        COALESCE((SELECT SUM(total_spent_czk) FROM auth.users), 0) as total_spent_czk
                    """
                )
                result = dict(stats)
                # Calculate averages (avoid division by zero)
                total_messages = result.get("total_messages", 0) or 1
                total_conversations = result.get("total_conversations", 0) or 1
                total_spent = float(result.get("total_spent_czk", 0) or 0)

                result["avg_spent_per_message_czk"] = round(total_spent / total_messages, 4)
                result["avg_spent_per_conversation_czk"] = round(total_spent / total_conversations, 4)
                result["total_spent_czk"] = round(total_spent, 2)

                return result
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

    # =========================================================================
    # Conversation Access (Admin View - Read Only)
    # =========================================================================

    async def get_user_conversations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        Get all conversations for a user (admin access).

        Args:
            user_id: User ID to fetch conversations for
            limit: Maximum number of conversations to return

        Returns:
            List of conversation dicts with message counts

        Raises:
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        c.id,
                        c.title,
                        c.created_at,
                        c.updated_at,
                        COUNT(m.id) as message_count
                    FROM auth.conversations c
                    LEFT JOIN auth.messages m ON c.id = m.conversation_id
                    WHERE c.user_id = $1
                    GROUP BY c.id, c.title, c.created_at, c.updated_at
                    ORDER BY c.updated_at DESC
                    LIMIT $2
                    """,
                    user_id, limit
                )
                return [dict(row) for row in rows]
        except Exception as e:
            self._handle_db_error("get_user_conversations", {"user_id": user_id}, e)

    async def verify_conversation_ownership(self, conversation_id: str, user_id: int) -> bool:
        """
        Verify that a conversation belongs to a specific user.

        Args:
            conversation_id: Conversation UUID to check
            user_id: Expected owner user ID

        Returns:
            True if conversation belongs to user, False otherwise

        Raises:
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
        """
        try:
            async with self.pool.acquire() as conn:
                is_owner = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM auth.conversations WHERE id = $1 AND user_id = $2)",
                    conversation_id, user_id
                )
                return is_owner
        except Exception as e:
            self._handle_db_error(
                "verify_conversation_ownership",
                {"conversation_id": conversation_id, "user_id": user_id},
                e
            )

    async def get_conversation_history(self, conversation_id: str, limit: int = 200) -> List[Dict]:
        """
        Get message history for a conversation (admin access).

        Args:
            conversation_id: Conversation UUID
            limit: Maximum number of messages to return

        Returns:
            List of message dicts ordered by creation time (ascending)

        Raises:
            DatabaseConnectionError: If database connection fails
            StorageError: For other database errors
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, role, content, metadata, created_at
                    FROM auth.messages
                    WHERE conversation_id = $1
                    ORDER BY created_at ASC
                    LIMIT $2
                    """,
                    conversation_id, limit
                )
                messages = []
                for row in rows:
                    msg = dict(row)
                    # Parse metadata JSON string to dict
                    if msg.get("metadata"):
                        try:
                            msg["metadata"] = json.loads(msg["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            msg["metadata"] = None
                    messages.append(msg)
                return messages
        except Exception as e:
            self._handle_db_error(
                "get_conversation_history",
                {"conversation_id": conversation_id},
                e
            )
