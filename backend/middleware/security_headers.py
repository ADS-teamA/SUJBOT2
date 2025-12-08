"""
Security Headers Middleware - HTTP Security Headers

Adds security-hardening HTTP headers to all responses.

Headers added:
- Content-Security-Policy (CSP) - XSS protection
- X-Frame-Options - Clickjacking protection
- X-Content-Type-Options - MIME sniffing protection
- Strict-Transport-Security (HSTS) - Force HTTPS
- X-XSS-Protection - Legacy XSS filter (for older browsers)
- Referrer-Policy - Control referrer information
- Permissions-Policy - Control browser features

Usage:
    # In main.py
    app.add_middleware(SecurityHeadersMiddleware, environment="production")
"""

from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.

    Follows OWASP Secure Headers Project recommendations (2024).
    """

    def __init__(
        self,
        app,
        environment: str = "development",
        enable_hsts: bool = True
    ):
        """
        Initialize security headers middleware.

        Args:
            app: FastAPI application instance
            environment: "development" or "production" (affects HSTS enforcement)
            enable_hsts: Whether to add Strict-Transport-Security header (HTTPS only)
        """
        super().__init__(app)
        self.environment = environment
        self.enable_hsts = enable_hsts and (environment == "production")

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Add security headers to response.

        Flow:
        1. Process request
        2. Add security headers to response
        3. Return response
        """
        response = await call_next(request)

        # Content-Security-Policy (CSP) - Prevents XSS attacks
        # Restrictive policy: only allow same-origin resources
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "  # Allow inline scripts for React
            "style-src 'self' 'unsafe-inline'; "  # Allow inline styles for React
            "img-src 'self' data: https:; "  # Allow images from HTTPS and data URIs
            "font-src 'self' data:; "
            "connect-src 'self' http://localhost:* https://localhost:*; "  # Allow API calls to localhost (dev)
            "frame-ancestors 'none'; "  # Prevent embedding in iframes
            "base-uri 'self'; "
            "form-action 'self'; "
        )

        # X-Frame-Options - Prevents clickjacking attacks
        # DENY = cannot be embedded in any iframe
        response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options - Prevents MIME sniffing
        # nosniff = browser must respect Content-Type header
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Strict-Transport-Security (HSTS) - Force HTTPS in production
        # Only add in production (requires HTTPS)
        if self.enable_hsts:
            # max-age=31536000 = 1 year
            # includeSubDomains = apply to all subdomains
            # preload = submit to HSTS preload list (https://hstspreload.org/)
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # X-XSS-Protection - Legacy XSS filter (for older browsers)
        # 1; mode=block = enable XSS filter and block page if attack detected
        # Note: Modern browsers use CSP instead, but this helps legacy browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy - Control referrer information leakage
        # strict-origin-when-cross-origin = send full URL for same-origin, only origin for cross-origin
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy (formerly Feature-Policy) - Control browser features
        # Disable potentially dangerous features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )

        return response
