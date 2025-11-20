"""
Conversation Management REST API

Provides CRUD endpoints for user conversations and messages.
All endpoints require JWT authentication and verify conversation ownership.
"""

from datetime import datetime
from typing import Dict, List, Optional
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from backend.middleware.auth import get_current_user
from src.storage.postgres_adapter import PostgreSQLStorageAdapter

router = APIRouter(prefix="/conversations", tags=["conversations"])
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class MessageResponse(BaseModel):
    """Response model for a single message."""
    id: int
    role: str
    content: str
    metadata: Optional[Dict]
    created_at: str

class ConversationCreate(BaseModel):
    """Request model for creating a new conversation."""
    title: Optional[str] = Field(default="New Conversation", max_length=500)

class ConversationResponse(BaseModel):
    """Response model for conversation metadata."""
    id: str
    user_id: int
    title: str
    messages: List[MessageResponse] = Field(default_factory=list)
    created_at: str
    updated_at: str

class MessageCreate(BaseModel):
    """Request model for appending a message to conversation."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=500000)
    metadata: Optional[Dict] = None

class TitleUpdate(BaseModel):
    """Request model for updating conversation title."""
    title: str = Field(..., min_length=1, max_length=500)

# ============================================================================
# Dependency Injection
# ============================================================================

_postgres_adapter: Optional[PostgreSQLStorageAdapter] = None

def set_postgres_adapter(adapter: PostgreSQLStorageAdapter):
    """Set the global PostgreSQL adapter instance (called from main.py)."""
    global _postgres_adapter
    _postgres_adapter = adapter

def get_postgres_adapter() -> PostgreSQLStorageAdapter:
    """Dependency for getting PostgreSQL adapter."""
    if _postgres_adapter is None:
        raise RuntimeError("PostgreSQLStorageAdapter not initialized")
    return _postgres_adapter

# ============================================================================
# Endpoints
# ============================================================================

@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=1000),
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
):
    """
    List all conversations for the authenticated user.

    Args:
        limit: Maximum number of conversations to return (default 50, max 1000)
        user: Authenticated user from JWT token
        adapter: PostgreSQL storage adapter

    Returns:
        List of conversations ordered by most recent update
    """
    try:
        conversations = await adapter.get_user_conversations(user["id"], limit=limit)

        # Convert to response model with empty messages (use GET /conversations/{id}/messages to fetch messages)
        return [
            ConversationResponse(
                id=conv["id"],
                user_id=user["id"],
                title=conv["title"] or "New Conversation",
                messages=[],  # Messages loaded separately via GET /{id}/messages
                created_at=conv["created_at"].isoformat() if hasattr(conv["created_at"], "isoformat") else str(conv["created_at"]),
                updated_at=conv["updated_at"].isoformat() if hasattr(conv["updated_at"], "isoformat") else str(conv["updated_at"])
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Failed to list conversations for user {user['id']}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations. Please try again later."
        )

@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    data: ConversationCreate,
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
):
    """
    Create a new conversation for the authenticated user.

    Args:
        data: Conversation creation data (title)
        user: Authenticated user from JWT token
        adapter: PostgreSQL storage adapter

    Returns:
        Created conversation metadata
    """
    # Generate conversation ID (UUID4 without prefix, 32 chars)
    conversation_id = uuid.uuid4().hex

    try:
        # Create conversation in database
        await adapter.create_conversation(
            conversation_id=conversation_id,
            user_id=user["id"],
            title=data.title
        )

        # Return created conversation with empty messages list
        now = datetime.now().isoformat()
        return ConversationResponse(
            id=conversation_id,
            user_id=user["id"],
            title=data.title or "New Conversation",
            messages=[],  # New conversations start with no messages
            created_at=now,
            updated_at=now
        )
    except Exception as e:
        logger.error(f"Failed to create conversation for user {user['id']}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation. Please try again later."
        )

@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
):
    """
    Get message history for a conversation.

    Args:
        conversation_id: Conversation ID
        limit: Maximum messages to return (default 100, max 1000)
        user: Authenticated user from JWT token
        adapter: PostgreSQL storage adapter

    Returns:
        List of messages ordered by creation time (ascending)

    Raises:
        HTTPException 403: User does not own this conversation
    """
    try:
        # Verify ownership
        owns = await adapter.verify_conversation_ownership(conversation_id, user["id"])
        if not owns:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this conversation"
            )

        # Get messages
        messages = await adapter.get_conversation_history(conversation_id, limit=limit)

        # Convert to response model
        return [
            MessageResponse(
                id=msg["id"],
                role=msg["role"],
                content=msg["content"],
                metadata=msg.get("metadata"),
                created_at=msg["created_at"].isoformat() if hasattr(msg["created_at"], "isoformat") else str(msg["created_at"])
            )
            for msg in messages
        ]
    except HTTPException:
        # Re-raise HTTP exceptions (403) without logging as errors
        raise
    except Exception as e:
        logger.error(f"Failed to get messages for conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve messages. Please try again later."
        )

@router.post("/{conversation_id}/messages", response_model=Dict, status_code=status.HTTP_201_CREATED)
async def append_message(
    conversation_id: str,
    message: MessageCreate,
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
):
    """
    Append a message to a conversation.

    Args:
        conversation_id: Conversation ID
        message: Message data (role, content, metadata)
        user: Authenticated user from JWT token
        adapter: PostgreSQL storage adapter

    Returns:
        Created message ID

    Raises:
        HTTPException 403: User does not own this conversation
    """
    try:
        # Verify ownership
        owns = await adapter.verify_conversation_ownership(conversation_id, user["id"])
        if not owns:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this conversation"
            )

        # Append message
        message_id = await adapter.append_message(
            conversation_id=conversation_id,
            role=message.role,
            content=message.content,
            metadata=message.metadata
        )

        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "message": "Message appended successfully"
        }
    except HTTPException:
        # Re-raise HTTP exceptions (403) without logging as errors
        raise
    except Exception as e:
        logger.error(f"Failed to append message to conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to append message. Please try again later."
        )

@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
):
    """
    Delete a conversation and all its messages.

    Args:
        conversation_id: Conversation ID
        user: Authenticated user from JWT token
        adapter: PostgreSQL storage adapter

    Raises:
        HTTPException 403: User does not own this conversation
        HTTPException 404: Conversation not found
    """
    try:
        # Delete conversation (includes ownership check)
        deleted = await adapter.delete_conversation(conversation_id, user["id"])

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or you do not have access to it"
            )

        return None  # 204 No Content
    except HTTPException:
        # Re-raise HTTP exceptions (404) without logging as errors
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation. Please try again later."
        )

@router.patch("/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: str,
    data: TitleUpdate,
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
):
    """
    Update conversation title.

    Args:
        conversation_id: Conversation ID
        data: Title update data
        user: Authenticated user from JWT token
        adapter: PostgreSQL storage adapter

    Returns:
        Success message

    Raises:
        HTTPException 403: User does not own this conversation
    """
    try:
        # Verify ownership
        owns = await adapter.verify_conversation_ownership(conversation_id, user["id"])
        if not owns:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this conversation"
            )

        # Update title
        async with adapter.pool.acquire() as conn:
            await conn.execute(
                "UPDATE auth.conversations SET title = $1, updated_at = NOW() WHERE id = $2",
                data.title, conversation_id
            )

        return {"message": "Title updated successfully", "title": data.title}
    except HTTPException:
        # Re-raise HTTP exceptions (403) without logging as errors
        raise
    except Exception as e:
        logger.error(f"Failed to update title for conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update conversation title. Please try again later."
        )
