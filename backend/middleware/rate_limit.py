"""
Rate Limiting Middleware - Token Bucket Algorithm

Prevents abuse by limiting requests per IP address using token bucket algorithm.

Features:
- In-memory rate limiting (no external dependencies)
- Token bucket algorithm (bursts allowed, sustained rate enforced)
- Configurable limits per route category
- Automatic cleanup of old buckets (memory leak prevention)

Usage:
    # In main.py
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,
        burst_size=10
    )
"""

from typing import Dict, Tuple, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket for rate limiting a single client.

    Allows bursts up to bucket capacity, then enforces sustained rate.

    Example:
        - Capacity: 10 tokens
        - Refill rate: 60 tokens/minute (1 token/second)
        - Client can make 10 requests instantly (burst)
        - Then limited to 1 request/second sustained
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens (burst size)
            refill_rate: Tokens added per second (requests_per_minute / 60)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)  # Start with full bucket
        self.last_refill = datetime.utcnow()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume (default: 1 for 1 request)

        Returns:
            True if tokens available (request allowed)
            False if insufficient tokens (request denied)
        """
        # Refill tokens based on time elapsed
        now = datetime.utcnow()
        elapsed = (now - self.last_refill).total_seconds()
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate)
        )
        self.last_refill = now

        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            return False

    def time_until_available(self) -> float:
        """
        Get seconds until next token available.

        Returns:
            Seconds to wait before retrying
        """
        if self.tokens >= 1:
            return 0.0
        tokens_needed = 1 - self.tokens
        return tokens_needed / self.refill_rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket per IP.

    Configurable limits:
    - Default: 60 requests/minute per IP
    - Auth routes: 10 requests/minute (brute force protection)
    - Public routes: No limit (health checks, docs)
    """

    # Route-specific limits (requests per minute)
    ROUTE_LIMITS = {
        "/auth/login": 10,      # Brute force protection
        "/auth/register": 5,    # Slow account creation abuse
        "/auth/me": 120,        # Session validation (React Admin calls frequently)
        "/auth/logout": 30,     # Logout endpoint
        "/chat/stream": 30,     # Moderate chat usage
        "/citations": 300,      # High limit for citation lookups (UI triggers many)
    }

    DEFAULT_LIMIT = 60  # requests/minute for other routes

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        cleanup_interval: int = 300  # 5 minutes
    ):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application instance
            requests_per_minute: Default rate limit (sustain rate)
            burst_size: Maximum burst size (bucket capacity)
            cleanup_interval: Seconds between bucket cleanup (prevent memory leak)
        """
        super().__init__(app)
        self.default_rpm = requests_per_minute
        self.burst_size = burst_size
        self.buckets: Dict[str, TokenBucket] = {}
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = datetime.utcnow()

    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.

        Flow:
        1. Get client IP address
        2. Get rate limit for route
        3. Check/create token bucket
        4. Try to consume token
        5. Allow or deny request
        """
        # Skip rate limiting for public routes
        if self._is_public_route(request.url.path):
            return await call_next(request)

        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)

        # Get rate limit for this route
        rpm, bucket_key = self._get_rate_limit(request.url.path, client_ip)

        # Get or create token bucket
        bucket = self._get_bucket(bucket_key, rpm)

        # Try to consume token
        if not bucket.consume():
            # Rate limit exceeded
            retry_after = int(bucket.time_until_available()) + 1
            logger.warning(
                f"Rate limit exceeded for {client_ip} on {request.url.path} "
                f"(retry after {retry_after}s)"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )

        # Cleanup old buckets periodically
        await self._cleanup_buckets()

        # Allow request
        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request.

        Supports:
        - X-Forwarded-For header (reverse proxy)
        - X-Real-IP header (nginx)
        - Direct connection

        Args:
            request: FastAPI request object

        Returns:
            Client IP address string
        """
        # Check reverse proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP (client) from chain
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        return request.client.host if request.client else "unknown"

    def _get_rate_limit(self, path: str, client_ip: str) -> Tuple[int, str]:
        """
        Get rate limit for route and construct bucket key.

        Args:
            path: Request path (e.g., "/auth/login")
            client_ip: Client IP address

        Returns:
            Tuple of (requests_per_minute, bucket_key)
        """
        # Check route-specific limits
        for route_prefix, limit in self.ROUTE_LIMITS.items():
            if path.startswith(route_prefix):
                # Bucket key: route + IP (separate limits per route)
                return limit, f"{route_prefix}:{client_ip}"

        # Default limit
        return self.DEFAULT_LIMIT, f"default:{client_ip}"

    def _get_bucket(self, bucket_key: str, rpm: int) -> TokenBucket:
        """
        Get or create token bucket for client.

        Args:
            bucket_key: Unique key (route:ip)
            rpm: Requests per minute limit

        Returns:
            TokenBucket instance
        """
        if bucket_key not in self.buckets:
            refill_rate = rpm / 60.0  # Convert to tokens/second
            self.buckets[bucket_key] = TokenBucket(
                capacity=self.burst_size,
                refill_rate=refill_rate
            )

        return self.buckets[bucket_key]

    def _is_public_route(self, path: str) -> bool:
        """
        Check if route is public (no rate limiting).

        Args:
            path: Request path

        Returns:
            True if public route, False otherwise
        """
        public_routes = {
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

        if path in public_routes:
            return True

        # Static files and assets
        if path.startswith(("/static/", "/favicon.ico", "/assets/")):
            return True

        # Citations endpoint - read-only metadata, safe to exclude from rate limiting
        # Frontend renders many citation links that trigger individual fetches
        if path.startswith("/citations"):
            return True

        return False

    async def _cleanup_buckets(self):
        """
        Remove old buckets to prevent memory leak.

        Runs every cleanup_interval seconds.
        Removes buckets inactive for >2x cleanup_interval.
        """
        now = datetime.utcnow()
        if (now - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return

        # Find buckets inactive for >10 minutes
        cutoff = now - timedelta(seconds=self.cleanup_interval * 2)
        old_keys = [
            key for key, bucket in self.buckets.items()
            if bucket.last_refill < cutoff
        ]

        # Remove old buckets
        for key in old_keys:
            del self.buckets[key]

        if old_keys:
            logger.info(f"Cleaned up {len(old_keys)} inactive rate limit buckets")

        self.last_cleanup = now
