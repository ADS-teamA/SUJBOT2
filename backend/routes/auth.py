"""
Authentication API Routes

Endpoints:
- POST /auth/login - User login with email/password
- POST /auth/logout - Clear session (delete JWT cookie)
- GET /auth/me - Get current user profile
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
from backend.middleware.auth import get_current_user

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


class UserResponse(BaseModel):
    """User profile response."""
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool
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
        created_at=user["created_at"].isoformat(),
        last_login_at=user["last_login_at"].isoformat() if user["last_login_at"] else None
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: RegisterRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
    auth_queries: AuthQueries = Depends(get_auth_queries),
    # current_user: Dict = Depends(get_current_user)  # Uncomment for admin-only registration
):
    """
    Register new user (admin creates accounts manually).

    ⚠️ PRODUCTION SECURITY:
    - This endpoint should be protected with admin authentication
    - Uncomment the current_user dependency to require admin login
    - Add admin role check: if not current_user.get("is_admin"): raise 403

    For now: Open registration for development/testing.

    Args:
        user_data: Email, password, optional full name

    Returns:
        Created user profile (without password hash)

    Raises:
        HTTPException 409: Email already exists
        HTTPException 422: Validation error (weak password, invalid email)

    Example:
        POST /auth/register
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
            created_at=user["created_at"].isoformat(),
            last_login_at=None
        )

    except Exception as e:
        logger.error(f"Failed to create user {user_data.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )
