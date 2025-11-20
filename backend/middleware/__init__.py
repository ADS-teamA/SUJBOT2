"""
Middleware package for FastAPI.

Provides authentication, rate limiting, and security headers.
"""

from .auth import AuthMiddleware, get_current_user, set_auth_instances
from .rate_limit import RateLimitMiddleware
from .security_headers import SecurityHeadersMiddleware

__all__ = [
    "AuthMiddleware",
    "get_current_user",
    "set_auth_instances",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware"
]
