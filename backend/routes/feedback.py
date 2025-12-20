"""
Message Feedback REST API

Provides endpoints for user feedback on assistant messages.
Stores feedback in PostgreSQL and sends to LangSmith for trace correlation.
All endpoints require JWT authentication and verify message ownership.
"""

import asyncio
import logging
import os
from typing import Dict, Literal, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, status, Path
from pydantic import BaseModel, Field, model_validator

from backend.middleware.auth import get_current_user
from src.storage.postgres_adapter import PostgreSQLStorageAdapter
from src.exceptions import StorageError

router = APIRouter(prefix="/feedback", tags=["feedback"])
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""

    message_id: int = Field(..., ge=1, description="Database message ID")
    run_id: Optional[str] = Field(None, description="LangSmith trace ID for correlation")
    score: Literal[-1, 1] = Field(..., description="1=thumbs up, -1=thumbs down")
    comment: Optional[str] = Field(None, max_length=2000, description="Optional comment")

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": 42,
                "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "score": 1,
                "comment": "Very helpful response!",
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for submitted feedback."""

    id: int = Field(..., ge=1)
    message_id: int = Field(..., ge=1)
    score: Literal[-1, 1]
    comment: Optional[str] = None
    langsmith_synced: bool
    created_at: str


class ExistingFeedbackResponse(BaseModel):
    """Response model for checking existing feedback."""

    has_feedback: bool
    score: Optional[Literal[-1, 1]] = None
    comment: Optional[str] = None

    @model_validator(mode="after")
    def check_consistency(self) -> "ExistingFeedbackResponse":
        """Ensure score/comment are only present when has_feedback is True."""
        if self.has_feedback and self.score is None:
            raise ValueError("score required when has_feedback is True")
        if not self.has_feedback and (self.score is not None or self.comment is not None):
            raise ValueError("score and comment must be None when has_feedback is False")
        return self


# ============================================================================
# Dependency Injection
# ============================================================================

_postgres_adapter: Optional[PostgreSQLStorageAdapter] = None
_langsmith_client = None
_table_created = False
_initialization_lock = asyncio.Lock()


def set_postgres_adapter(adapter: PostgreSQLStorageAdapter):
    """Set the global PostgreSQL adapter instance (called from main.py)."""
    global _postgres_adapter
    _postgres_adapter = adapter


def get_postgres_adapter() -> PostgreSQLStorageAdapter:
    """Dependency for getting PostgreSQL adapter."""
    if _postgres_adapter is None:
        raise RuntimeError("PostgreSQLStorageAdapter not initialized")
    return _postgres_adapter


def _get_langsmith_client():
    """Get or create LangSmith client singleton."""
    global _langsmith_client
    if _langsmith_client is None:
        api_key = os.getenv("LANGSMITH_API_KEY")
        if api_key:
            try:
                from langsmith import Client

                endpoint = os.getenv(
                    "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
                )
                _langsmith_client = Client(api_key=api_key, api_url=endpoint)
                logger.info(f"LangSmith client initialized for feedback (endpoint: {endpoint})")
            except ImportError:
                logger.warning("langsmith package not installed, feedback sync disabled")
            except (OSError, ValueError) as e:
                # OSError: Network issues, ValueError: Invalid configuration
                logger.error(f"Failed to initialize LangSmith client: {e}")
        else:
            logger.info("LANGSMITH_API_KEY not set, feedback sync disabled")
    return _langsmith_client


async def _ensure_feedback_table(adapter: PostgreSQLStorageAdapter):
    """Create feedback table if it doesn't exist (for existing deployments)."""
    global _table_created
    if _table_created:
        return

    async with _initialization_lock:
        # Double-check after acquiring lock to prevent race condition
        if _table_created:
            return

        try:
            async with adapter.pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth.message_feedback (
                        id SERIAL PRIMARY KEY,
                        message_id INTEGER NOT NULL REFERENCES auth.messages(id) ON DELETE CASCADE,
                        user_id INTEGER NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
                        run_id TEXT,
                        score INTEGER NOT NULL CHECK (score IN (-1, 1)),
                        comment TEXT,
                        langsmith_synced BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        UNIQUE(message_id, user_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_message_feedback_message_id
                    ON auth.message_feedback(message_id)
                    """
                )
            _table_created = True
            logger.debug("Feedback table verified/created")
        except asyncpg.PostgresError as e:
            logger.error(f"Failed to ensure feedback table: {e}", exc_info=True)
            raise StorageError(
                message="Failed to create feedback table",
                details={"error_type": type(e).__name__},
                cause=e
            )


# ============================================================================
# Helper Functions
# ============================================================================


async def _verify_message_ownership(
    adapter: PostgreSQLStorageAdapter, message_id: int, user_id: int
) -> bool:
    """
    Verify that the user owns the conversation containing the message.

    Args:
        adapter: PostgreSQL storage adapter
        message_id: ID of the message to check
        user_id: ID of the user to verify ownership for

    Returns:
        True if user owns the message, False otherwise

    Raises:
        StorageError: If database query fails
    """
    try:
        async with adapter.pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM auth.messages m
                    JOIN auth.conversations c ON m.conversation_id = c.id
                    WHERE m.id = $1 AND c.user_id = $2
                )
                """,
                message_id,
                user_id,
            )
            return bool(result)
    except asyncpg.PostgresError as e:
        logger.error(
            f"Failed to verify message ownership: message_id={message_id}, user_id={user_id}",
            exc_info=True
        )
        raise StorageError(
            message="Failed to verify message ownership",
            details={"message_id": message_id, "user_id": user_id},
            cause=e
        )


def _send_to_langsmith(run_id: str, score: int, comment: Optional[str]) -> bool:
    """
    Send feedback to LangSmith (synchronous, may add latency to response).

    Args:
        run_id: LangSmith trace/run ID
        score: User score (-1 or 1)
        comment: Optional comment

    Returns:
        True if sent successfully, False otherwise
    """
    client = _get_langsmith_client()
    if not client:
        return False

    try:
        # Map score: 1 (thumbs up) -> 1.0, -1 (thumbs down) -> 0.0
        langsmith_score = 1.0 if score == 1 else 0.0

        client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=langsmith_score,
            comment=comment,
        )
        logger.info(
            f"Feedback sent to LangSmith: run={run_id[:8]}..., "
            f"score={langsmith_score}, has_comment={comment is not None}"
        )
        return True
    except Exception as e:
        # Log at error level since this is data loss for observability
        logger.error(
            f"Failed to send feedback to LangSmith: {type(e).__name__}: {e}",
            extra={"run_id": run_id[:8] if run_id else None, "score": score}
        )
        return False


# ============================================================================
# Endpoints
# ============================================================================


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    request: FeedbackRequest,
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter),
):
    """
    Submit feedback for an assistant message.

    - Stores feedback in PostgreSQL
    - Sends to LangSmith if run_id is provided
    - Returns 409 if feedback already submitted for this message

    Args:
        request: Feedback data (message_id, score, optional comment)
        user: Authenticated user from JWT token
        adapter: PostgreSQL storage adapter

    Returns:
        Created feedback record

    Raises:
        HTTPException 403: If user doesn't own the message's conversation
        HTTPException 409: If feedback already submitted for this message
        HTTPException 500: If database save fails
    """
    # Ensure feedback table exists
    await _ensure_feedback_table(adapter)

    # Verify user owns the message's conversation
    if not await _verify_message_ownership(adapter, request.message_id, user["id"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this message",
        )

    # Insert feedback
    try:
        async with adapter.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO auth.message_feedback (message_id, user_id, run_id, score, comment)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, message_id, score, comment, langsmith_synced, created_at
                """,
                request.message_id,
                user["id"],
                request.run_id,
                request.score,
                request.comment,
            )
    except asyncpg.UniqueViolationError:
        # Duplicate feedback (user already rated this message)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback already submitted for this message",
        )
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to insert feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback",
        )

    # Send to LangSmith (synchronous, may add latency)
    langsmith_synced = False
    if request.run_id:
        langsmith_synced = _send_to_langsmith(
            request.run_id, request.score, request.comment
        )

        # Update sync status in database
        if langsmith_synced:
            try:
                async with adapter.pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE auth.message_feedback
                        SET langsmith_synced = TRUE
                        WHERE id = $1
                        """,
                        row["id"],
                    )
            except asyncpg.PostgresError as e:
                # Non-critical: flag update failed but feedback was saved
                logger.warning(f"Failed to update langsmith_synced flag: {e}")

    logger.info(
        f"Feedback submitted: user={user['id']}, message={request.message_id}, "
        f"score={request.score}, langsmith_synced={langsmith_synced}"
    )

    return FeedbackResponse(
        id=row["id"],
        message_id=row["message_id"],
        score=row["score"],
        comment=row["comment"],
        langsmith_synced=langsmith_synced,
        created_at=row["created_at"].isoformat(),
    )


@router.get("/{message_id}", response_model=ExistingFeedbackResponse)
async def get_feedback(
    message_id: int = Path(..., ge=1, description="Message ID to check"),
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter),
):
    """
    Check if user has submitted feedback for a message.

    Used by frontend to restore feedback state on page refresh.
    Only returns feedback for messages the user owns.

    Args:
        message_id: ID of the message to check
        user: Authenticated user from JWT token
        adapter: PostgreSQL storage adapter

    Returns:
        Existing feedback if any, or has_feedback=false
    """
    # Ensure feedback table exists
    await _ensure_feedback_table(adapter)

    # Verify user owns the message's conversation (security check)
    # Return empty response instead of 403 to avoid leaking message existence
    if not await _verify_message_ownership(adapter, message_id, user["id"]):
        return ExistingFeedbackResponse(has_feedback=False)

    try:
        async with adapter.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT score, comment FROM auth.message_feedback
                WHERE message_id = $1 AND user_id = $2
                """,
                message_id,
                user["id"],
            )

        if row:
            return ExistingFeedbackResponse(
                has_feedback=True,
                score=row["score"],
                comment=row["comment"],
            )
        else:
            return ExistingFeedbackResponse(has_feedback=False)

    except asyncpg.PostgresError as e:
        logger.error(f"Failed to get feedback for message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feedback",
        )
