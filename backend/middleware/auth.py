"""
Authentication Middleware - JWT Token Validation

Protects FastAPI routes by validating JWT tokens from httpOnly cookies or Authorization headers.

Usage:
    # In main.py
    app.add_middleware(AuthMiddleware, auth_manager=auth_manager, auth_queries=auth_queries)

    # In route handlers
    @app.get("/protected")
    async def protected_route(user: Dict = Depends(get_current_user)):
        return {"user_id": user["id"], "email": user["email"]}
"""

from typing import Optional, Dict, Callable
from fastapi import Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    JWT authentication middleware for FastAPI.

    Validates tokens from:
    1. httpOnly cookie "access_token" (preferred for XSS protection)
    2. Authorization header "Bearer <token>" (fallback for API clients)

    Attaches validated user to request.state.user for downstream handlers.
    """

    # Public routes that don't require authentication
    PUBLIC_ROUTES = {
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/auth/login",
        "/auth/register",  # If admin creates accounts, this should be protected
    }

    def __init__(self, app, auth_manager, auth_queries):
        """
        Initialize auth middleware.

        Args:
            app: FastAPI application instance
            auth_manager: AuthManager instance (from backend/auth/manager.py)
            auth_queries: AuthQueries instance (from backend/database/auth_queries.py)
        """
        super().__init__(app)
        self.auth_manager = auth_manager
        self.auth_queries = auth_queries

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request and validate authentication.

        Flow:
        1. Check if route is public (skip auth)
        2. Extract JWT token from cookie or header
        3. Validate token and load user
        4. Attach user to request.state
        5. Call next middleware/handler
        """
        # Skip auth for public routes
        if request.url.path in self.PUBLIC_ROUTES:
            return await call_next(request)

        # Skip auth for static files and docs
        if request.url.path.startswith(("/static/", "/favicon.ico")):
            return await call_next(request)

        # Extract token from cookie or header
        token = self._extract_token(request)

        if not token:
            logger.warning(f"Missing authentication token for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authentication required"}
            )

        # Validate token
        payload = self.auth_manager.validate_token(token)
        if not payload:
            logger.warning(f"Invalid/expired token for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired token"}
            )

        # Load user from database (ensure user is active)
        user = await self.auth_queries.get_active_user_by_id(payload.user_id)
        if not user:
            logger.warning(f"User {payload.user_id} not found or inactive")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "User not found or inactive"}
            )

        # Attach user to request state (available in route handlers)
        request.state.user = user

        # Update last login timestamp (async background task)
        # Note: We don't await this to avoid blocking the request
        # In production, consider using a background task queue
        try:
            await self.auth_queries.update_last_login(user["id"])
        except Exception as e:
            # Log but don't fail request if timestamp update fails
            logger.error(f"Failed to update last_login for user {user['id']}: {e}")

        # Continue processing request
        return await call_next(request)

    def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from cookie or Authorization header.

        Priority:
        1. httpOnly cookie "access_token" (XSS-safe)
        2. Authorization header "Bearer <token>" (API clients)

        Args:
            request: FastAPI request object

        Returns:
            JWT token string or None if not found
        """
        # Try httpOnly cookie first (most secure)
        token = request.cookies.get("access_token")
        if token:
            logger.debug("Token extracted from httpOnly cookie")
            return token

        # Fallback to Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            logger.debug("Token extracted from Authorization header")
            return token

        return None


# =========================================================================
# Dependency Injection for Route Handlers (WITHOUT Middleware)
# =========================================================================

# Global instances (set by main.py during startup)
_auth_manager_instance: Optional['AuthManager'] = None
_auth_queries_instance: Optional['AuthQueries'] = None


def set_auth_instances(auth_manager, auth_queries):
    """
    Set global auth instances (called by main.py during startup).

    Args:
        auth_manager: AuthManager instance
        auth_queries: AuthQueries instance
    """
    global _auth_manager_instance, _auth_queries_instance
    _auth_manager_instance = auth_manager
    _auth_queries_instance = auth_queries


async def get_current_user(request: Request) -> Dict:
    """
    FastAPI dependency for getting authenticated user.

    Validates JWT token from cookie or header, loads user from database.

    Usage:
        @app.get("/profile")
        async def get_profile(user: Dict = Depends(get_current_user)):
            return {"email": user["email"], "name": user["full_name"]}

    Args:
        request: FastAPI request object

    Returns:
        User dict with keys: id, email, full_name, is_active, etc.

    Raises:
        HTTPException 401: If user not authenticated or token invalid
    """
    if not _auth_manager_instance or not _auth_queries_instance:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system not initialized"
        )

    # Extract token from cookie or Authorization header
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required (missing token)"
        )

    # Validate token
    payload = _auth_manager_instance.validate_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    # Load user from database
    user = await _auth_queries_instance.get_active_user_by_id(payload.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    return user


async def get_current_active_user(request: Request) -> Dict:
    """
    FastAPI dependency for getting active user (extra validation).

    Similar to get_current_user but enforces is_active check.

    Usage:
        @app.post("/admin/users")
        async def create_user(user: Dict = Depends(get_current_active_user)):
            # Only active users can access this route
            pass

    Args:
        request: FastAPI request object

    Returns:
        Active user dict

    Raises:
        HTTPException 401: If user not authenticated or inactive
    """
    user = await get_current_user(request)

    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    return user

async def get_current_admin_user(request: Request) -> Dict:
    """
    FastAPI dependency for getting admin user (admin-only routes).

    Validates authentication AND admin privileges.

    Usage:
        @app.post("/auth/register")
        async def register_user(admin: Dict = Depends(get_current_admin_user)):
            # Only admins can create new users
            pass

    Args:
        request: FastAPI request object

    Returns:
        Admin user dict

    Raises:
        HTTPException 401: If user not authenticated
        HTTPException 403: If user is not admin or inactive
    """
    user = await get_current_user(request)

    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    if not user.get("is_admin", False):
        logger.warning(f"Non-admin user {user['id']} attempted to access admin endpoint")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

    return user
