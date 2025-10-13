"""WebSocket endpoints for real-time chat."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict
import json
import logging
from datetime import datetime

from app.services.chat_service import ChatService
from app.core.dependencies import get_chat_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manager for WebSocket connections."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Accept new WebSocket connection.

        Args:
            websocket: WebSocket instance
            client_id: Unique client identifier
        """
        logger.info(f"ConnectionManager.connect() called for client {client_id}")
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} accepted and stored")

    def disconnect(self, client_id: str):
        """
        Remove connection.

        Args:
            client_id: Client identifier
        """
        self.active_connections.pop(client_id, None)

    async def send_message(self, client_id: str, message: dict):
        """
        Send message to specific client.

        Args:
            client_id: Client identifier
            message: Message dictionary
        """
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_json(message)

    async def stream_response(self, client_id: str, response_stream):
        """
        Stream chunks to client.

        Handles special status messages in format: __STATUS__{json}__STATUS__

        Args:
            client_id: Client identifier
            response_stream: Async iterator of response chunks
        """
        websocket = self.active_connections.get(client_id)
        if not websocket:
            return

        try:
            async for chunk in response_stream:
                # Check if chunk contains pipeline status
                if "__STATUS__" in chunk:
                    # Extract status JSON
                    parts = chunk.split("__STATUS__")
                    if len(parts) >= 3:  # Should be: ['', '{json}', '\n']
                        status_json = parts[1]
                        try:
                            status_data = json.loads(status_json)
                            # Send pipeline status as separate message
                            await websocket.send_json(status_data)
                        except json.JSONDecodeError:
                            # If parsing fails, send as regular content
                            await websocket.send_json({
                                "type": "stream_chunk",
                                "content": chunk
                            })
                    continue  # Don't send status markers to frontend

                # Send regular content chunk
                await websocket.send_json({
                    "type": "stream_chunk",
                    "content": chunk
                })

            # Send completion message
            await websocket.send_json({
                "type": "stream_complete"
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })


manager = ConnectionManager()


@router.websocket("")
async def websocket_chat(
    websocket: WebSocket,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    WebSocket endpoint for real-time chat.

    Message format:
    {
        "type": "chat_message",
        "content": "User question...",
        "document_ids": ["doc1", "doc2"]
    }

    Response format:
    {
        "type": "stream_chunk",
        "content": "Response chunk..."
    }
    """
    # Generate client ID
    client_id = str(id(websocket))
    logger.info(f"WebSocket handler started for client {client_id}")

    await manager.connect(websocket, client_id)
    logger.info(f"Client {client_id} connected, waiting for messages...")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"Received WebSocket message: type={message.get('type')}, content={message.get('content', '')[:50]}...")

            if message["type"] == "chat_message":
                # Send acknowledgment
                await manager.send_message(client_id, {
                    "type": "message_received",
                    "timestamp": datetime.now().isoformat()
                })

                logger.info(f"Processing query with {len(message.get('document_ids', []))} documents")

                # Process query and stream response
                response_stream = chat_service.process_query_stream(
                    query=message["content"],
                    document_ids=message.get("document_ids", [])
                )

                await manager.stream_response(client_id, response_stream)
                logger.info("Query processing complete")

            elif message["type"] == "ping":
                # Heartbeat
                await manager.send_message(client_id, {"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await manager.send_message(client_id, {
            "type": "error",
            "message": str(e)
        })
        manager.disconnect(client_id)
