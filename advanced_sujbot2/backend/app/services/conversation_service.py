"""
Conversation Service - Handles persistence of chat conversations to PostgreSQL

Provides ChatGPT-like conversation management:
- Create new conversations
- Load existing conversations
- Add messages to conversations
- List conversations
- Update conversation metadata (title, archive, favorite)
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
import asyncpg
import os

from app.models.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationListItem,
    MessageCreate,
    MessageResponse
)

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversations in PostgreSQL"""

    def __init__(self):
        """Initialize conversation service with PostgreSQL connection"""
        self.pool: Optional[asyncpg.Pool] = None
        self._initialize_task = None

    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        if self.pool is not None:
            return

        try:
            # Get PostgreSQL credentials from environment
            db_config = {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DB", "sujbot2"),
                "user": os.getenv("POSTGRES_USER", "sujbot_app"),
                "password": os.getenv("POSTGRES_PASSWORD", ""),
                "min_size": 5,
                "max_size": 20,
                "command_timeout": 60
            }

            logger.info(f"Initializing conversation service with PostgreSQL at {db_config['host']}:{db_config['port']}")

            self.pool = await asyncpg.create_pool(**db_config)

            # Verify schema exists
            await self._ensure_schema()

            logger.info("Conversation service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize conversation service: {e}")
            raise

    async def _ensure_schema(self):
        """Ensure conversation schema exists in database"""
        try:
            async with self.pool.acquire() as conn:
                # Check if conversations table exists
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM pg_tables
                        WHERE schemaname = 'public'
                          AND tablename = 'conversations'
                    );
                """)

                if not exists:
                    logger.warning("Conversations table does not exist. Run database/schema_conversations.sql")
                else:
                    logger.info("Conversations schema verified")

        except Exception as e:
            logger.error(f"Failed to verify schema: {e}")
            raise

    async def close(self):
        """Close PostgreSQL connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None

    # ========================================================================
    # Conversation Operations
    # ========================================================================

    async def create_conversation(
        self,
        create_data: ConversationCreate
    ) -> ConversationResponse:
        """
        Create a new conversation.

        Args:
            create_data: Conversation creation data

        Returns:
            Created conversation with metadata
        """
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                # Insert conversation
                row = await conn.fetchrow("""
                    INSERT INTO conversations (title, document_ids, session_id)
                    VALUES ($1, $2, $3)
                    RETURNING id, title, created_at, updated_at, message_count,
                              is_archived, is_favorite, document_ids, session_id
                """, create_data.title, create_data.document_ids, create_data.session_id)

                return ConversationResponse(
                    id=row['id'],
                    title=row['title'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    message_count=row['message_count'],
                    is_archived=row['is_archived'],
                    is_favorite=row['is_favorite'],
                    document_ids=row['document_ids'],
                    session_id=row['session_id'],
                    messages=[]
                )

        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise

    async def get_conversation(
        self,
        conversation_id: UUID
    ) -> Optional[ConversationResponse]:
        """
        Get full conversation with all messages.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation with messages, or None if not found
        """
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                # Get conversation metadata
                conv_row = await conn.fetchrow("""
                    SELECT id, title, created_at, updated_at, message_count,
                           is_archived, is_favorite, document_ids, session_id
                    FROM conversations
                    WHERE id = $1
                """, conversation_id)

                if not conv_row:
                    return None

                # Get all messages for this conversation
                message_rows = await conn.fetch("""
                    SELECT id, conversation_id, type, content, created_at,
                           sequence, intent, pipeline, sources, metadata
                    FROM messages
                    WHERE conversation_id = $1
                    ORDER BY sequence ASC
                """, conversation_id)

                # Build messages list
                messages = [
                    MessageResponse(
                        id=row['id'],
                        conversation_id=row['conversation_id'],
                        type=row['type'],
                        content=row['content'],
                        created_at=row['created_at'],
                        sequence=row['sequence'],
                        intent=row['intent'],
                        pipeline=row['pipeline'],
                        sources=row['sources'],
                        metadata=row['metadata'] or {}
                    )
                    for row in message_rows
                ]

                return ConversationResponse(
                    id=conv_row['id'],
                    title=conv_row['title'],
                    created_at=conv_row['created_at'],
                    updated_at=conv_row['updated_at'],
                    message_count=conv_row['message_count'],
                    is_archived=conv_row['is_archived'],
                    is_favorite=conv_row['is_favorite'],
                    document_ids=conv_row['document_ids'],
                    session_id=conv_row['session_id'],
                    messages=messages
                )

        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            raise

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False
    ) -> Tuple[List[ConversationListItem], int]:
        """
        List conversations (most recent first).

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            include_archived: Include archived conversations

        Returns:
            (conversations, total_count)
        """
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                # Build WHERE clause
                where_clause = "" if include_archived else "WHERE is_archived = FALSE"

                # Get total count
                total = await conn.fetchval(f"""
                    SELECT COUNT(*) FROM conversations {where_clause}
                """)

                # Get conversations
                rows = await conn.fetch(f"""
                    SELECT * FROM conversation_list
                    {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT $1 OFFSET $2
                """, limit, offset)

                conversations = [
                    ConversationListItem(
                        id=row['id'],
                        title=row['title'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        message_count=row['message_count'],
                        is_archived=row['is_archived'],
                        is_favorite=row['is_favorite'],
                        document_ids=row['document_ids'],
                        latest_message_preview=row['latest_message_preview'],
                        latest_message_type=row['latest_message_type']
                    )
                    for row in rows
                ]

                return conversations, total

        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            raise

    async def update_conversation(
        self,
        conversation_id: UUID,
        update_data: ConversationUpdate
    ) -> Optional[ConversationResponse]:
        """
        Update conversation metadata.

        Args:
            conversation_id: Conversation ID
            update_data: Fields to update

        Returns:
            Updated conversation, or None if not found
        """
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                # Build UPDATE query dynamically
                updates = []
                params = []
                param_index = 1

                if update_data.title is not None:
                    updates.append(f"title = ${param_index}")
                    params.append(update_data.title)
                    param_index += 1

                if update_data.is_archived is not None:
                    updates.append(f"is_archived = ${param_index}")
                    params.append(update_data.is_archived)
                    param_index += 1

                if update_data.is_favorite is not None:
                    updates.append(f"is_favorite = ${param_index}")
                    params.append(update_data.is_favorite)
                    param_index += 1

                if update_data.document_ids is not None:
                    updates.append(f"document_ids = ${param_index}")
                    params.append(update_data.document_ids)
                    param_index += 1

                if not updates:
                    # No updates - just return current conversation
                    return await self.get_conversation(conversation_id)

                # Always update updated_at
                updates.append("updated_at = CURRENT_TIMESTAMP")

                # Add conversation_id as last param
                params.append(conversation_id)

                query = f"""
                    UPDATE conversations
                    SET {', '.join(updates)}
                    WHERE id = ${param_index}
                    RETURNING id
                """

                result = await conn.fetchval(query, *params)

                if result:
                    return await self.get_conversation(conversation_id)
                else:
                    return None

        except Exception as e:
            logger.error(f"Failed to update conversation {conversation_id}: {e}")
            raise

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        """
        Delete a conversation (and all its messages via CASCADE).

        Args:
            conversation_id: Conversation ID

        Returns:
            True if deleted, False if not found
        """
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM conversations WHERE id = $1
                """, conversation_id)

                return result == "DELETE 1"

        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            raise

    # ========================================================================
    # Message Operations
    # ========================================================================

    async def add_message(
        self,
        conversation_id: UUID,
        message_data: MessageCreate
    ) -> MessageResponse:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation ID
            message_data: Message data

        Returns:
            Created message
        """
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO messages (
                        conversation_id, type, content, sequence,
                        intent, pipeline, sources, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id, conversation_id, type, content, created_at,
                              sequence, intent, pipeline, sources, metadata
                """,
                    conversation_id,
                    message_data.type,
                    message_data.content,
                    message_data.sequence,
                    message_data.intent,
                    message_data.pipeline,
                    message_data.sources,
                    message_data.metadata or {}
                )

                return MessageResponse(
                    id=row['id'],
                    conversation_id=row['conversation_id'],
                    type=row['type'],
                    content=row['content'],
                    created_at=row['created_at'],
                    sequence=row['sequence'],
                    intent=row['intent'],
                    pipeline=row['pipeline'],
                    sources=row['sources'],
                    metadata=row['metadata'] or {}
                )

        except Exception as e:
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            raise

    async def add_messages_batch(
        self,
        conversation_id: UUID,
        messages: List[MessageCreate]
    ) -> List[MessageResponse]:
        """
        Add multiple messages to a conversation in a single transaction.

        Args:
            conversation_id: Conversation ID
            messages: List of messages to add

        Returns:
            List of created messages
        """
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    created_messages = []

                    for msg_data in messages:
                        row = await conn.fetchrow("""
                            INSERT INTO messages (
                                conversation_id, type, content, sequence,
                                intent, pipeline, sources, metadata
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            RETURNING id, conversation_id, type, content, created_at,
                                      sequence, intent, pipeline, sources, metadata
                        """,
                            conversation_id,
                            msg_data.type,
                            msg_data.content,
                            msg_data.sequence,
                            msg_data.intent,
                            msg_data.pipeline,
                            msg_data.sources,
                            msg_data.metadata or {}
                        )

                        created_messages.append(MessageResponse(
                            id=row['id'],
                            conversation_id=row['conversation_id'],
                            type=row['type'],
                            content=row['content'],
                            created_at=row['created_at'],
                            sequence=row['sequence'],
                            intent=row['intent'],
                            pipeline=row['pipeline'],
                            sources=row['sources'],
                            metadata=row['metadata'] or {}
                        ))

                    return created_messages

        except Exception as e:
            logger.error(f"Failed to add messages batch to conversation {conversation_id}: {e}")
            raise


# ============================================================================
# Global Singleton
# ============================================================================

_conversation_service: Optional[ConversationService] = None


def get_conversation_service() -> ConversationService:
    """Get or create global conversation service instance"""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service
