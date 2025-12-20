"""
Admin API Routes

Endpoints:
- POST /admin/login - Admin-only login
- GET /admin/users - List all users (paginated)
- GET /admin/users/{id} - Get user details
- POST /admin/users - Create new user
- PUT /admin/users/{id} - Update user
- DELETE /admin/users/{id} - Delete user
- GET /admin/health - Detailed health check
- GET /admin/stats - System statistics

All endpoints (except /admin/login) require admin JWT token.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response, Query
from typing import Optional, Dict, List
from datetime import datetime, timezone
import logging
import os
import asyncpg

from backend.auth.manager import AuthManager
from backend.database.auth_queries import AuthQueries
from backend.database.admin_queries import AdminQueries
from backend.middleware.auth import get_current_admin_user
from backend.models import (
    AdminUserResponse,
    AdminUserListResponse,
    AdminUserCreateRequest,
    AdminUserUpdateRequest,
    AdminLoginRequest,
    ServiceHealthDetail,
    AdminHealthResponse,
    AdminStatsResponse,
    AdminConversationResponse,
    AdminMessageResponse,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/admin", tags=["admin"])


# =============================================================================
# Global Dependencies (injected by main.py)
# =============================================================================

_auth_manager: Optional[AuthManager] = None
_auth_queries: Optional[AuthQueries] = None
_admin_queries: Optional[AdminQueries] = None


def set_admin_dependencies(
    auth_manager: AuthManager,
    auth_queries: AuthQueries,
    postgres_adapter
):
    """
    Set global admin dependencies (called from main.py).

    Args:
        auth_manager: AuthManager instance
        auth_queries: AuthQueries instance
        postgres_adapter: PostgreSQLStorageAdapter instance
    """
    global _auth_manager, _auth_queries, _admin_queries
    _auth_manager = auth_manager
    _auth_queries = auth_queries
    _admin_queries = AdminQueries(postgres_adapter)


def get_auth_manager() -> AuthManager:
    """Dependency injection for AuthManager."""
    if _auth_manager is None:
        raise RuntimeError("AuthManager not initialized. Call set_admin_dependencies() first.")
    return _auth_manager


def get_auth_queries() -> AuthQueries:
    """Dependency injection for AuthQueries."""
    if _auth_queries is None:
        raise RuntimeError("AuthQueries not initialized. Call set_admin_dependencies() first.")
    return _auth_queries


def get_admin_queries() -> AdminQueries:
    """Dependency injection for AdminQueries."""
    if _admin_queries is None:
        raise RuntimeError("AdminQueries not initialized. Call set_admin_dependencies() first.")
    return _admin_queries


# =============================================================================
# Helper Functions
# =============================================================================

def _format_user_response(user: Dict) -> AdminUserResponse:
    """Format user dict to AdminUserResponse."""
    return AdminUserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user.get("full_name"),
        is_active=user["is_active"],
        is_admin=user.get("is_admin", False),
        agent_variant=user.get("agent_variant"),
        created_at=user["created_at"].isoformat() if user.get("created_at") else "",
        updated_at=user["updated_at"].isoformat() if user.get("updated_at") else "",
        last_login_at=user["last_login_at"].isoformat() if user.get("last_login_at") else None,
        # Spending fields
        spending_limit_czk=float(user.get("spending_limit_czk", 500.0) or 500.0),
        total_spent_czk=float(user.get("total_spent_czk", 0.0) or 0.0),
        spending_reset_at=user["spending_reset_at"].isoformat() if user.get("spending_reset_at") else None,
    )


# =============================================================================
# Admin Login
# =============================================================================

@router.post("/login")
async def admin_login(
    credentials: AdminLoginRequest,
    response: Response,
    auth_manager: AuthManager = Depends(get_auth_manager),
    auth_queries: AuthQueries = Depends(get_auth_queries)
):
    """
    Admin-only login.

    Same flow as /auth/login but validates is_admin after password verification.

    Returns:
        User profile and success message

    Raises:
        HTTPException 401: Invalid credentials
        HTTPException 403: Not an admin or inactive
    """
    # Look up user by email
    user = await auth_queries.get_user_by_email(credentials.email)

    if not user:
        logger.warning(f"Admin login attempt for non-existent user: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Verify password FIRST (prevent timing attacks on is_admin check)
    if not auth_manager.verify_password(credentials.password, user["password_hash"]):
        logger.warning(f"Failed admin login attempt for user: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Check if account is active
    if not user["is_active"]:
        logger.warning(f"Admin login attempt for inactive user: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )

    # Check admin privileges
    if not user.get("is_admin", False):
        logger.warning(f"Non-admin login attempt on /admin/login: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

    # Generate JWT token
    token = auth_manager.create_token(user_id=user["id"], email=user["email"])

    # Set httpOnly cookie
    is_production = os.getenv("BUILD_TARGET", "development") == "production"
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=is_production,
        samesite="lax",
        max_age=86400,  # 24 hours
        path="/",  # Cookie valid for all paths (needed for /auth/me check)
    )

    # Update last login timestamp
    await auth_queries.update_last_login(user["id"])

    logger.info(f"Admin {user['id']} ({credentials.email}) logged in successfully")

    return {
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "is_active": user["is_active"],
            "is_admin": user.get("is_admin", False),
            "created_at": user["created_at"].isoformat(),
            "last_login_at": user["last_login_at"].isoformat() if user["last_login_at"] else None,
        },
        "message": "Admin login successful"
    }


# =============================================================================
# User CRUD Endpoints
# =============================================================================

@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    limit: int = 50,
    offset: int = 0,
    admin: Dict = Depends(get_current_admin_user),
    auth_queries: AuthQueries = Depends(get_auth_queries)
):
    """
    List all users with pagination.

    Args:
        limit: Maximum users to return (default 50)
        offset: Pagination offset

    Returns:
        Paginated list of users
    """
    users = await auth_queries.list_users(limit=limit, offset=offset)
    total = await auth_queries.count_users()

    return AdminUserListResponse(
        users=[_format_user_response(u) for u in users],
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/users/{user_id}", response_model=AdminUserResponse)
async def get_user(
    user_id: int,
    admin: Dict = Depends(get_current_admin_user),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    Get user details by ID.

    Args:
        user_id: User ID

    Returns:
        User details

    Raises:
        HTTPException 404: User not found
    """
    user = await admin_queries.get_user_full(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return _format_user_response(user)


@router.post("/users", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: AdminUserCreateRequest,
    admin: Dict = Depends(get_current_admin_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    auth_queries: AuthQueries = Depends(get_auth_queries),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    Create new user.

    Args:
        user_data: User creation data

    Returns:
        Created user

    Raises:
        HTTPException 409: Email already exists
        HTTPException 422: Weak password
    """
    # Check if email already exists
    existing_user = await auth_queries.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Admin can set any password - no strength validation
    # Hash password
    password_hash = auth_manager.hash_password(user_data.password)

    # Create user
    try:
        user_id = await auth_queries.create_user(
            email=user_data.email,
            password_hash=password_hash,
            full_name=user_data.full_name,
            is_active=user_data.is_active
        )

        # Set admin flag if requested (requires separate update since create_user doesn't support it)
        if user_data.is_admin:
            await admin_queries.update_user(user_id, is_admin=True)

        # Retrieve created user
        user = await admin_queries.get_user_full(user_id)

        logger.info(f"Admin {admin['id']} created user {user_id} ({user_data.email})")

        return _format_user_response(user)

    except asyncpg.PostgresConnectionError as e:
        logger.error(f"Database connection failed creating user {user_data.email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable. Please try again."
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists"
        )
    except asyncpg.PostgresError as e:
        logger.error(f"Database error creating user {user_data.email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating user {user_data.email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.put("/users/{user_id}", response_model=AdminUserResponse)
async def update_user(
    user_id: int,
    user_data: AdminUserUpdateRequest,
    admin: Dict = Depends(get_current_admin_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    auth_queries: AuthQueries = Depends(get_auth_queries),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    Update user.

    Args:
        user_id: User ID to update
        user_data: Fields to update

    Returns:
        Updated user

    Raises:
        HTTPException 400: Self-protection or last-admin protection
        HTTPException 404: User not found
        HTTPException 409: Email conflict
    """
    # Check if user exists
    existing_user = await admin_queries.get_user_full(user_id)
    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Self-protection checks
    if user_id == admin["id"]:
        if user_data.is_admin is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove your own admin privileges"
            )
        if user_data.is_active is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )

    # Last-admin protection
    if user_data.is_admin is False or user_data.is_active is False:
        if await admin_queries.is_last_admin(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote or deactivate the last admin"
            )

    # Check email uniqueness if changing
    if user_data.email and user_data.email != existing_user["email"]:
        email_check = await auth_queries.get_user_by_email(user_data.email)
        if email_check:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already in use by another user"
            )

    # Update password if provided (admin can set any password, no validation)
    if user_data.password:
        password_hash = auth_manager.hash_password(user_data.password)
        await auth_queries.update_password(user_id, password_hash)
        logger.info(f"Admin {admin['id']} changed password for user {user_id}")

    # Update user
    try:
        await admin_queries.update_user(
            user_id,
            email=user_data.email,
            full_name=user_data.full_name,
            is_admin=user_data.is_admin,
            is_active=user_data.is_active,
            agent_variant=user_data.agent_variant,
            spending_limit_czk=user_data.spending_limit_czk
        )

        # Retrieve updated user
        user = await admin_queries.get_user_full(user_id)

        logger.info(f"Admin {admin['id']} updated user {user_id}")

        return _format_user_response(user)

    except asyncpg.UniqueViolationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already in use"
        )
    except asyncpg.PostgresConnectionError as e:
        logger.error(f"Database connection failed updating user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable. Please try again."
        )
    except asyncpg.PostgresError as e:
        logger.error(f"Database error updating user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error"
        )
    except Exception as e:
        logger.error(f"Unexpected error updating user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    admin: Dict = Depends(get_current_admin_user),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    Delete user.

    Args:
        user_id: User ID to delete

    Raises:
        HTTPException 400: Self-protection or last-admin protection
        HTTPException 404: User not found
    """
    # Self-protection
    if user_id == admin["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    # Check if user exists
    existing_user = await admin_queries.get_user_full(user_id)
    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Last-admin protection
    if await admin_queries.is_last_admin(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the last admin"
        )

    # Delete user
    try:
        deleted = await admin_queries.delete_user(user_id)

        if deleted:
            logger.info(f"Admin {admin['id']} deleted user {user_id}")

        return None

    except asyncpg.PostgresConnectionError as e:
        logger.error(f"Database connection failed deleting user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable. Please try again."
        )
    except asyncpg.ForeignKeyViolationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete user with associated data"
        )
    except asyncpg.PostgresError as e:
        logger.error(f"Database error deleting user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error"
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


# =============================================================================
# Health & Stats Endpoints
# =============================================================================

@router.get("/health", response_model=AdminHealthResponse)
async def admin_health_check(
    admin: Dict = Depends(get_current_admin_user),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    Detailed health check for admin dashboard.

    Checks:
    - PostgreSQL database connection
    - Backend API (implicit - if you get this response, it's healthy)

    Returns:
        Detailed health status with per-service breakdown
    """
    services = []
    overall_status = "healthy"

    # Check PostgreSQL
    db_health = await admin_queries.check_database_health()
    services.append(ServiceHealthDetail(
        name="PostgreSQL",
        status=db_health["status"],
        latency_ms=db_health.get("latency_ms"),
        message=db_health.get("message")
    ))

    if db_health["status"] == "unhealthy":
        overall_status = "unhealthy"
    elif db_health["status"] == "degraded" and overall_status == "healthy":
        overall_status = "degraded"

    # Backend is healthy if we got here
    services.append(ServiceHealthDetail(
        name="Backend API",
        status="healthy",
        latency_ms=None,
        message=None
    ))

    return AdminHealthResponse(
        status=overall_status,
        services=services,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get("/stats", response_model=AdminStatsResponse)
async def get_system_stats(
    admin: Dict = Depends(get_current_admin_user),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    Get system statistics for admin dashboard.

    Returns:
        User counts, conversation counts, activity metrics
    """
    stats = await admin_queries.get_system_stats()

    return AdminStatsResponse(
        total_users=stats["total_users"],
        active_users=stats["active_users"],
        admin_users=stats["admin_users"],
        total_conversations=stats["total_conversations"],
        total_messages=stats["total_messages"],
        users_last_24h=stats["users_last_24h"],
        total_spent_czk=stats.get("total_spent_czk", 0.0),
        avg_spent_per_message_czk=stats.get("avg_spent_per_message_czk", 0.0),
        avg_spent_per_conversation_czk=stats.get("avg_spent_per_conversation_czk", 0.0),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# Spending Management Endpoints
# =============================================================================

@router.post("/users/{user_id}/spending/reset", status_code=status.HTTP_200_OK)
async def reset_user_spending(
    user_id: int,
    admin: Dict = Depends(get_current_admin_user),
    auth_queries: AuthQueries = Depends(get_auth_queries),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    Reset user's spending counter to zero.

    Args:
        user_id: User ID whose spending to reset

    Returns:
        Success message with reset timestamp

    Raises:
        HTTPException 404: User not found
    """
    # Check if user exists
    user = await admin_queries.get_user_full(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Reset spending
    await auth_queries.reset_user_spending(user_id)

    logger.info(f"Admin {admin['id']} reset spending for user {user_id}")

    return {
        "message": "Spending reset successfully",
        "user_id": user_id,
        "reset_at": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# Conversation Viewing Endpoints (Read-Only)
# =============================================================================


@router.get("/users/{user_id}/conversations", response_model=List[AdminConversationResponse])
async def list_user_conversations(
    user_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    admin: Dict = Depends(get_current_admin_user),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    List all conversations for a specific user (admin view).

    Read-only access - returns conversations with metadata but not messages.

    Args:
        user_id: User ID to fetch conversations for
        limit: Maximum number of conversations to return (1-200, default 50)

    Returns:
        List of conversations with id, title, message_count, timestamps

    Raises:
        HTTPException 404: User not found
    """
    # Verify user exists
    user = await admin_queries.get_user_full(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Get conversations
    conversations = await admin_queries.get_user_conversations(user_id, limit=limit)

    return [
        AdminConversationResponse(
            id=conv["id"],
            title=conv["title"] or "New Conversation",
            message_count=conv.get("message_count", 0),
            created_at=conv["created_at"].isoformat(),
            updated_at=conv["updated_at"].isoformat()
        )
        for conv in conversations
    ]


@router.get(
    "/users/{user_id}/conversations/{conversation_id}/messages",
    response_model=List[AdminMessageResponse]
)
async def get_user_conversation_messages(
    user_id: int,
    conversation_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    admin: Dict = Depends(get_current_admin_user),
    admin_queries: AdminQueries = Depends(get_admin_queries)
):
    """
    Get all messages for a specific conversation (admin view).

    Read-only access - no modification allowed.

    Args:
        user_id: User ID who owns the conversation
        conversation_id: Conversation UUID to fetch messages from
        limit: Maximum number of messages to return (1-1000, default 200)

    Returns:
        List of messages with id, role, content, metadata, created_at

    Raises:
        HTTPException 404: User or conversation not found
    """
    # Verify user exists
    user = await admin_queries.get_user_full(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Verify conversation belongs to user
    if not await admin_queries.verify_conversation_ownership(conversation_id, user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Get messages
    messages = await admin_queries.get_conversation_history(conversation_id, limit=limit)

    return [
        AdminMessageResponse(
            id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            metadata=msg.get("metadata"),
            created_at=msg["created_at"].isoformat()
        )
        for msg in messages
    ]
