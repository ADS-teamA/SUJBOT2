"""Custom logging middleware for FastAPI."""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Use uvicorn.error instead of uvicorn.access to avoid format conflicts
logger = logging.getLogger("uvicorn.error")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next):
        """Log request and response details."""
        start_time = time.time()

        # Log request
        logger.info(f"{request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Log response
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"{request.method} {request.url.path} "
            f"completed in {process_time:.2f}ms "
            f"with status {response.status_code}"
        )

        # Add process time header
        response.headers["X-Process-Time"] = str(process_time)

        return response
