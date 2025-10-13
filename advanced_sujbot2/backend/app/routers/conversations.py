"""
Conversation API Router - REST endpoints for chat persistence

Provides ChatGPT-like conversation management:
- GET /conversations - List all conversations
- POST /conversations - Create new conversation
- GET /conversations/{id} - Get full conversation with messages
- PATCH /conversations/{id} - Update conversation (title, archive, favorite)
- DELETE /conversations/{id} - Delete conversation
- POST /conversations/{id}/messages - Add message to conversation
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from uuid import UUID
import logging

from app.models.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationListItem,
    ConversationListResponse,
    MessageCreate,
    MessageResponse,
    MessageBatchCreate
)
from app.services.conversation_service import get_conversation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


# ============================================================================
# Conversation Endpoints
# ============================================================================

@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    conversation_data: ConversationCreate
):
    """
    Create a new conversation.

    Returns the created conversation with metadata.
    """
    try:
        service = get_conversation_service()
        conversation = await service.create_conversation(conversation_data)
        logger.info(f"Created conversation: {conversation.id}")
        return conversation
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_archived: bool = Query(False)
):
    """
    List conversations (most recent first).

    Query parameters:
    - limit: Maximum number of conversations (1-100, default 50)
    - offset: Number of conversations to skip (pagination)
    - include_archived: Include archived conversations (default false)
    """
    try:
        service = get_conversation_service()
        conversations, total = await service.list_conversations(
            limit=limit,
            offset=offset,
            include_archived=include_archived
        )

        return ConversationListResponse(
            conversations=conversations,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID
):
    """
    Get full conversation with all messages.

    Returns 404 if conversation not found.
    """
    try:
        service = get_conversation_service()
        conversation = await service.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: UUID,
    update_data: ConversationUpdate
):
    """
    Update conversation metadata.

    Can update:
    - title: Change conversation title
    - is_archived: Archive/unarchive conversation
    - is_favorite: Mark as favorite
    - document_ids: Update document context

    Returns 404 if conversation not found.
    """
    try:
        service = get_conversation_service()
        conversation = await service.update_conversation(conversation_id, update_data)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        logger.info(f"Updated conversation: {conversation_id}")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID
):
    """
    Delete a conversation and all its messages.

    Returns 204 No Content on success.
    Returns 404 if conversation not found.
    """
    try:
        service = get_conversation_service()
        deleted = await service.delete_conversation(conversation_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")

        logger.info(f"Deleted conversation: {conversation_id}")
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Message Endpoints
# ============================================================================

@router.post("/{conversation_id}/messages", response_model=MessageResponse, status_code=201)
async def add_message(
    conversation_id: UUID,
    message_data: MessageCreate
):
    """
    Add a message to a conversation.

    The sequence number should be managed by the client to maintain order.
    Typically: increment by 1 for each message in the conversation.
    """
    try:
        service = get_conversation_service()
        message = await service.add_message(conversation_id, message_data)
        logger.info(f"Added message to conversation {conversation_id}")
        return message
    except Exception as e:
        logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conversation_id}/messages/batch", response_model=List[MessageResponse], status_code=201)
async def add_messages_batch(
    conversation_id: UUID,
    batch_data: MessageBatchCreate
):
    """
    Add multiple messages to a conversation in a single transaction.

    Useful for saving entire conversations at once (e.g., when syncing after refresh).
    """
    try:
        service = get_conversation_service()
        messages = await service.add_messages_batch(conversation_id, batch_data.messages)
        logger.info(f"Added {len(messages)} messages to conversation {conversation_id}")
        return messages
    except Exception as e:
        logger.error(f"Failed to add messages batch to conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
