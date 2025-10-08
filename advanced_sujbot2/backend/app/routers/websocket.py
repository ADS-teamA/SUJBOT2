"""WebSocket endpoints for real-time chat."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict
import json
from datetime import datetime

from app.services.chat_service import ChatService
from app.core.dependencies import get_chat_service

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
        await websocket.accept()
        self.active_connections[client_id] = websocket

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

        Args:
            client_id: Client identifier
            response_stream: Async iterator of response chunks
        """
        websocket = self.active_connections.get(client_id)
        if not websocket:
            return

        try:
            async for chunk in response_stream:
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

    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "chat_message":
                # Send acknowledgment
                await manager.send_message(client_id, {
                    "type": "message_received",
                    "timestamp": datetime.now().isoformat()
                })

                # Process query and stream response
                response_stream = chat_service.process_query_stream(
                    query=message["content"],
                    document_ids=message.get("document_ids", [])
                )

                await manager.stream_response(client_id, response_stream)

            elif message["type"] == "ping":
                # Heartbeat
                await manager.send_message(client_id, {"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")

    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.send_message(client_id, {
            "type": "error",
            "message": str(e)
        })
        manager.disconnect(client_id)
