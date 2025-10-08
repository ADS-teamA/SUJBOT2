"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import redis
import logging

from app.routers import documents, compliance, query, websocket
from app.core.config import settings
from app.middleware.logging import LoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.VERBOSE_LOGGING else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_redis_connection() -> str:
    """Check Redis connection status."""
    try:
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=1)
        r.ping()
        return "connected"
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return "disconnected"


def check_celery_status() -> str:
    """Check Celery worker status."""
    try:
        from app.core.celery_app import celery_app
        stats = celery_app.control.inspect().stats()
        if stats:
            return "running"
        return "no_workers"
    except Exception as e:
        logger.error(f"Celery check failed: {e}")
        return "error"


def check_faiss_loaded() -> str:
    """Check if FAISS indices are loaded."""
    # In production, check if indices exist
    index_dir = settings.INDEX_DIR
    if os.path.exists(index_dir) and os.listdir(index_dir):
        return "loaded"
    return "not_loaded"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown."""
    # Startup
    logger.info("Starting SUJBOT2 API...")

    # Ensure directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.INDEX_DIR, exist_ok=True)

    # Check Redis connection
    redis_status = check_redis_connection()
    logger.info(f"Redis status: {redis_status}")

    # Check Celery workers
    celery_status = check_celery_status()
    logger.info(f"Celery status: {celery_status}")

    logger.info("API ready!")

    yield

    # Shutdown
    logger.info("Shutting down...")
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.API_TITLE,
    description="Legal Compliance Checking API with RAG - Czech Public Procurement Focus",
    version=settings.API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom logging middleware
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(
    documents.router,
    prefix=f"{settings.API_PREFIX}/documents",
    tags=["documents"]
)
app.include_router(
    compliance.router,
    prefix=f"{settings.API_PREFIX}/compliance",
    tags=["compliance"]
)
app.include_router(
    query.router,
    prefix=f"{settings.API_PREFIX}/query",
    tags=["query"]
)
app.include_router(
    websocket.router,
    prefix="/ws",
    tags=["websocket"]
)

# Serve uploaded files (if needed)
if os.path.exists(settings.UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SUJBOT2 API",
        "version": settings.API_VERSION,
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "description": "Legal compliance checking for Czech public procurement"
    }


@app.get(f"{settings.API_PREFIX}/health")
async def health_check():
    """
    Health check endpoint.

    Returns service status and version information.
    """
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "services": {
            "redis": check_redis_connection(),
            "celery": check_celery_status(),
            "faiss": check_faiss_loaded()
        }
    }
