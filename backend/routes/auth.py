"""
Authentication API Routes

Endpoints:
- POST /auth/login - User login with email/password
- POST /auth/logout - Clear session (delete JWT cookie)
- GET /auth/me - Get current user profile
- POST /auth/change-password - Change current user's password
- POST /auth/register - Create new user (admin only in production)

All responses use httpOnly cookies for JWT storage (XSS protection).
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict
import logging
import os

from backend.auth.manager import AuthManager
from backend.database.auth_queries import AuthQueries
from backend.middleware.auth import get_current_user, get_current_admin_user

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])


# =========================================================================
# Request/Response Models
# =========================================================================

class LoginRequest(BaseModel):
    """Login credentials."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")


class RegisterRequest(BaseModel):
    """User registration data."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 chars)")
    full_name: Optional[str] = Field(None, max_length=100, description="Display name")


class ChangePasswordRequest(BaseModel):
    """Password change request."""
    current_password: str = Field(..., min_length=1, description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 chars)")


class UserResponse(BaseModel):
    """User profile response."""
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: str
    last_login_at: Optional[str]


class AuthResponse(BaseModel):
    """Authentication success response."""
    user: UserResponse
    message: str


# =========================================================================
# Global Dependencies (injected by main.py)
# =========================================================================

# These will be set by main.py during app initialization
_auth_manager: Optional[AuthManager] = None
_auth_queries: Optional[AuthQueries] = None


def set_dependencies(auth_manager: AuthManager, auth_queries: AuthQueries):
    """
    Set global auth dependencies (called from main.py).

    Args:
        auth_manager: AuthManager instance
        auth_queries: AuthQueries instance
    """
    global _auth_manager, _auth_queries
    _auth_manager = auth_manager
    _auth_queries = auth_queries


def get_auth_manager() -> AuthManager:
    """Dependency injection for AuthManager."""
    if _auth_manager is None:
        raise RuntimeError("AuthManager not initialized. Call set_dependencies() first.")
    return _auth_manager


def get_auth_queries() -> AuthQueries:
    """Dependency injection for AuthQueries."""
    if _auth_queries is None:
        raise RuntimeError("AuthQueries not initialized. Call set_dependencies() first.")
    return _auth_queries


# =========================================================================
# Authentication Endpoints
# =========================================================================

@router.post("/login", response_model=AuthResponse)
async def login(
    credentials: LoginRequest,
    response: Response,
    auth_manager: AuthManager = Depends(get_auth_manager),
    auth_queries: AuthQueries = Depends(get_auth_queries)
):
    """
    Authenticate user with email/password.

    Flow:
    1. Look up user by email
    2. Verify password with Argon2
    3. Generate JWT token
    4. Set httpOnly cookie
    5. Return user profile

    Args:
        credentials: Email and password
        response: FastAPI response (for setting cookie)

    Returns:
        User profile and success message

    Raises:
        HTTPException 401: Invalid credentials
        HTTPException 403: Account inactive

    Example:
        POST /auth/login
        {
            "email": "admin@sujbot.local",
            "password": "ChangeThisPassword123!"
        }

        Response:
        {
            "user": {
                "id": 1,
                "email": "admin@sujbot.local",
                "full_name": "System Administrator",
                "is_active": true,
                ...
            },
            "message": "Login successful"
        }

        + Set-Cookie: access_token=eyJhbGc...; HttpOnly; Secure; SameSite=Lax
    """
    # Look up user by email
    user = await auth_queries.get_user_by_email(credentials.email)

    if not user:
        logger.warning(f"Login attempt for non-existent user: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Check if account is active
    if not user["is_active"]:
        logger.warning(f"Login attempt for inactive user: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Contact administrator."
        )

    # Verify password (Argon2)
    if not auth_manager.verify_password(credentials.password, user["password_hash"]):
        logger.warning(f"Failed login attempt for user: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Generate JWT token (24h expiry)
    token = auth_manager.create_token(user_id=user["id"], email=user["email"])

    # Set httpOnly cookie (XSS protection)
    # Secure flag: True for production (HTTPS), False for local development
    is_production = os.getenv("BUILD_TARGET", "development") == "production"
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,      # JavaScript cannot access (XSS protection)
        secure=is_production,  # HTTPS only in production
        samesite="lax",     # CSRF protection
        max_age=86400,      # 24 hours (matches token expiry)
    )

    # Update last login timestamp
    await auth_queries.update_last_login(user["id"])

    logger.info(f"User {user['id']} ({credentials.email}) logged in successfully")

    return {
        "user": UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            is_active=user["is_active"],
            is_admin=user.get("is_admin", False),
            created_at=user["created_at"].isoformat(),
            last_login_at=user["last_login_at"].isoformat() if user["last_login_at"] else None
        ),
        "message": "Login successful"
    }


@router.post("/logout")
async def logout(response: Response):
    """
    Log out user by clearing JWT cookie.

    Note: JWT tokens are stateless, so we can't invalidate them server-side.
    We just remove the cookie from the client.

    For production: Consider implementing token blacklist in Redis.

    Returns:
        Success message

    Example:
        POST /auth/logout

        Response:
        {
            "message": "Logged out successfully"
        }

        + Set-Cookie: access_token=; Max-Age=0 (deletes cookie)
    """
    # Delete cookie by setting max_age=0
    response.delete_cookie(key="access_token")

    logger.info("User logged out")

    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    user: Dict = Depends(get_current_user)
):
    """
    Get current authenticated user's profile.

    Requires authentication (JWT token in cookie or header).

    Args:
        user: Current user (injected by AuthMiddleware)

    Returns:
        User profile

    Example:
        GET /auth/me
        Cookie: access_token=eyJhbGc...

        Response:
        {
            "id": 1,
            "email": "admin@sujbot.local",
            "full_name": "System Administrator",
            "is_active": true,
            "created_at": "2025-11-19T12:00:00",
            "last_login_at": "2025-11-19T14:30:00"
        }
    """
    return UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        is_active=user["is_active"],
            is_admin=user.get("is_admin", False),
        created_at=user["created_at"].isoformat(),
        last_login_at=user["last_login_at"].isoformat() if user["last_login_at"] else None
    )


@router.post("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    user: Dict = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    auth_queries: AuthQueries = Depends(get_auth_queries)
):
    """
    Change current user's password.

    Requires authentication. User must provide current password for verification.

    Args:
        password_data: Current and new password
        user: Current authenticated user

    Returns:
        Success message

    Raises:
        HTTPException 401: Current password incorrect
        HTTPException 422: New password doesn't meet requirements
    """
    # Get full user data with password hash
    full_user = await auth_queries.get_user_by_id(user["id"])

    # Verify current password
    if not auth_manager.verify_password(password_data.current_password, full_user["password_hash"]):
        logger.warning(f"Password change failed - wrong current password: user {user['id']}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )

    # Validate new password strength
    is_valid, errors = auth_manager.validate_password_strength(password_data.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "New password does not meet security requirements",
                "errors": errors
            }
        )

    # Hash and update password
    new_hash = auth_manager.hash_password(password_data.new_password)
    await auth_queries.update_password(user["id"], new_hash)

    logger.info(f"Password changed for user {user['id']} ({user['email']})")

    return {"message": "Password changed successfully"}


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: RegisterRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
    auth_queries: AuthQueries = Depends(get_auth_queries),
    admin: Dict = Depends(get_current_admin_user)  # Requires admin authentication
):
    """
    Register new user (admin-only endpoint).

    This endpoint requires admin authentication. Admins can create new user
    accounts with optional full name. Password strength is validated.

    Note: For admin portal user creation, prefer POST /admin/users which
    has additional options like setting is_admin flag directly.

    Args:
        user_data: Email, password, optional full name
        admin: Admin user (automatically validated via dependency)

    Returns:
        Created user profile (without password hash)

    Raises:
        HTTPException 401: Not authenticated
        HTTPException 403: Not an admin
        HTTPException 409: Email already exists
        HTTPException 422: Validation error (weak password, invalid email)

    Example:
        POST /auth/register (requires admin JWT cookie)
        {
            "email": "user@example.com",
            "password": "SecurePass123!",
            "full_name": "John Doe"
        }

        Response:
        {
            "id": 2,
            "email": "user@example.com",
            "full_name": "John Doe",
            "is_active": true,
            "created_at": "2025-11-19T15:00:00",
            "last_login_at": null
        }
    """
    # Check if email already exists
    existing_user = await auth_queries.get_user_by_email(user_data.email)
    if existing_user:
        logger.warning(f"Registration attempt with existing email: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Validate password strength (OWASP requirements)
    is_valid, errors = auth_manager.validate_password_strength(user_data.password)
    if not is_valid:
        logger.warning(f"Registration attempt with weak password: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Password does not meet security requirements",
                "errors": errors
            }
        )

    # Hash password with Argon2
    password_hash = auth_manager.hash_password(user_data.password)

    # Create user in database
    try:
        user_id = await auth_queries.create_user(
            email=user_data.email,
            password_hash=password_hash,
            full_name=user_data.full_name,
            is_active=True
        )

        # Retrieve created user
        user = await auth_queries.get_user_by_id(user_id)

        logger.info(f"New user registered: {user_data.email} (ID: {user_id})")

        return UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            is_active=user["is_active"],
            is_admin=user.get("is_admin", False),
            created_at=user["created_at"].isoformat(),
            last_login_at=None
        )

    except Exception as e:
        logger.error(f"Failed to create user {user_data.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )
